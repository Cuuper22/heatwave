import csv
import math
import sys
import threading
import time
import types

import numpy as np
import pytest

from heatwave.laptop import telemetry as tm
from heatwave.laptop.telemetry import (
    FakeBackend, FakeParams, NvidiaSmiBackend, NvmlBackend, Sampler, bin_series,
    parse_smi_number, parse_smi_stream_line,
)


def test_parse_na_values():
    assert math.isnan(parse_smi_number("[N/A]"))
    assert math.isnan(parse_smi_number("[Not Supported]"))
    assert parse_smi_number(" 42.5 ") == 42.5
    row = parse_smi_stream_line("55.2, 61, 1500, 7000, 90, 40, [N/A], [N/A], 80.00, 2000")
    assert row["power_w"] == 55.2 and row["temp_c"] == 61
    assert np.isnan(row["fan_pct"]) and np.isnan(row["power_limit_w"])
    assert row["enforced_limit_w"] == 80.0 and row["mem_used_mib"] == 2000


class StubNvml(types.SimpleNamespace):
    """Stand-in for pynvml. Power-limit getters raise like on a laptop."""


def make_stub():
    n = StubNvml()
    n.NVML_TEMPERATURE_GPU = 0
    n.NVML_CLOCK_SM = 1
    n.NVML_CLOCK_MEM = 2
    n.nvmlInit = lambda: None
    n.nvmlShutdown = lambda: None
    n.nvmlDeviceGetHandleByIndex = lambda i: "h"
    n.nvmlDeviceGetName = lambda h: b"Stub GPU"
    n.nvmlDeviceGetPowerManagementLimitConstraints = lambda h: (1000, 105000)
    n.nvmlDeviceGetPowerManagementDefaultLimit = lambda h: 80000
    n.nvmlDeviceGetMaxClockInfo = lambda h, k: 1900
    n.nvmlDeviceGetSupportedMemoryClocks = lambda h: [7000]
    n.nvmlDeviceGetSupportedGraphicsClocks = lambda h, m: [210, 300, 1900]
    n.nvmlDeviceGetPowerUsage = lambda h: 55500
    n.nvmlDeviceGetTotalEnergyConsumption = lambda h: 123000
    n.nvmlDeviceGetTemperature = lambda h, k: 66
    n.nvmlDeviceGetClockInfo = lambda h, k: 1500 if k == 1 else 7000
    n.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=91, memory=33)
    n.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=2 * 2**30)

    def nope(*a):
        raise RuntimeError("Not Supported")

    n.nvmlDeviceGetFanSpeed = nope
    n.nvmlDeviceGetPowerManagementLimit = nope
    n.nvmlDeviceGetEnforcedPowerLimit = lambda h: 80000
    return n


def test_nvml_backend_maps_units_and_handles_unsupported():
    b = NvmlBackend(0, nvml=make_stub())
    info = b.info()
    assert info.name == "Stub GPU"
    assert (info.min_limit_w, info.default_limit_w, info.max_limit_w) == (1.0, 80.0, 105.0)
    assert info.clock_max_mhz == 1900 and info.clock_min_mhz == 210
    assert np.isnan(info.power_limit_w)
    r = b.read()
    assert r["power_w"] == 55.5 and r["energy_j"] == 123.0 and r["temp_c"] == 66
    assert r["sm_clock_mhz"] == 1500 and r["util_gpu_pct"] == 91
    assert np.isnan(r["fan_pct"]) and np.isnan(r["power_limit_w"])
    assert r["enforced_limit_w"] == 80.0 and r["mem_used_mib"] == 2048


def test_smi_backend_streams_and_parses():
    script = ("import sys,time\n"
              "for i in range(100):\n"
              "    print('50.%d, 60, 1400, 7000, 80, 30, [N/A], [N/A], 80.00, 1000' % i, flush=True)\n"
              "    time.sleep(0.05)\n")
    info_script = "print('Stub, 80.00, 1.00, 105.00, [N/A], 1900')"
    b = NvidiaSmiBackend(stream_cmd=[sys.executable, "-c", script],
                         info_cmd=[sys.executable, "-c", info_script])
    try:
        r = b.read()
        assert 50 <= r["power_w"] < 51 and r["temp_c"] == 60
        info = b.info()
        assert info.default_limit_w == 80 and info.max_limit_w == 105 and np.isnan(info.power_limit_w)
    finally:
        b.close()


def test_make_backend_fake():
    assert tm.make_backend("fake").data_kind == "simulated"


# --- simulated RC model ----------------------------------------------------------------
class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def settle(limit_w=None, seconds=600):
    clk = Clock()
    b = FakeBackend(FakeParams(power_noise_w=0, temp_noise_c=0, smi_mode="power"), clock=clk)
    b.set_activity(1.0)
    if limit_w:
        assert b.smi(["nvidia-smi", "-pl", str(limit_w)])[0] == 0
    clk.t = seconds
    return b.read()


def test_rc_model_steady_state_and_limit_effect():
    p = FakeParams()
    full = settle()
    low = settle(limit_w=50)
    # steady state: T = ambient + R * P, with P = min(demand, limit)
    assert full["temp_c"] == pytest.approx(p.ambient_c + p.r_k_per_w * p.default_limit_w, abs=0.1)
    assert low["temp_c"] == pytest.approx(p.ambient_c + p.r_k_per_w * 50, abs=0.1)
    assert low["power_w"] == pytest.approx(50, abs=0.01)
    assert low["temp_c"] < full["temp_c"]


def test_fake_smi_modes():
    b = FakeBackend(FakeParams(smi_mode="clock"))
    assert b.smi(["nvidia-smi", "-pl", "70"])[0] != 0
    assert b.smi(["nvidia-smi", "-lgc", "210,1000"])[0] == 0
    assert b.smi(["nvidia-smi", "-rgc"])[0] == 0
    b = FakeBackend(FakeParams(smi_mode="power"))
    assert b.smi(["nvidia-smi", "-pl", "70"])[0] == 0
    assert b.smi(["nvidia-smi", "-pl", "500"])[0] != 0
    b = FakeBackend(FakeParams(smi_mode="noadmin"))
    assert "Insufficient" in b.smi(["nvidia-smi", "-pl", "70"])[2]


def test_bin_series():
    t = np.array([0.1, 0.5, 1.2, 3.9])
    v = np.array([1.0, 3.0, 5.0, 7.0])
    out = bin_series(t, v, 4)
    assert out[0] == 2.0 and out[1] == 5.0 and np.isnan(out[2]) and out[3] == 7.0


def test_sampler_writes_labeled_csv(tmp_path):
    stop = threading.Event()
    s = Sampler(FakeBackend(), tmp_path / "t.csv", stop, hz=20.0)
    s.start()
    s.started.wait(2)
    time.sleep(1.6)
    stop.set()
    s.join(5)
    assert s.error is None
    with open(tmp_path / "t.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == list(tm.CSV_COLUMNS)
    assert len(rows) >= 25
    assert {r["data_kind"] for r in rows} == {"simulated"}
    assert rows[0]["t_utc"].endswith("+00:00")
    mono = [float(r["t_mono_s"]) for r in rows]
    assert mono == sorted(mono)
    assert s.window(1.0) and s.latest() is not None
