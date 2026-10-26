"""Turn a raw laptop collection run (scripts/laptop_collect.py output) into processed series.

Usage: python scripts/process_laptop_run.py <raw_run_dir> <name> [out_root]

Writes data/processed/laptop_<name>_10hz.parquet (native ~10 Hz samples)
and laptop_<name>_1s.parquet (1 s means), both MEASURED, with these
known-future covariates built from the load generator's own event log:
  - training_on      1 while the load is in a training interval, else 0
                     (the scripted schedule is known in advance)
  - ckpt_scheduled   1 during a checkpoint write interval
  - segment          integer code of the scripted segment
plus the events table laptop_<name>_events.parquet.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from heatwave.data.schema import PROCESSED, save_series as _save  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(raw: Path, name: str, root: Path = PROCESSED) -> None:
    def save_series(df, nm, meta):
        return _save(df, nm, meta, root=root)

    tel = pd.read_csv(raw / "telemetry.csv")
    ev = pd.read_csv(raw / "events.csv")
    meta = json.loads((raw / "meta.json").read_text())
    tel["time"] = pd.to_datetime(tel["utc"], utc=True, format="ISO8601")
    ev["time"] = pd.to_datetime(ev["utc"], utc=True, format="ISO8601")
    tel = tel.set_index("time").drop(columns=["utc"])
    num = ["power_w", "energy_mj", "temp_c", "sm_clock_mhz", "util_gpu_pct",
           "util_mem_pct", "enforced_limit_w", "clock_event_reasons"]
    for c in num:
        tel[c] = pd.to_numeric(tel[c], errors="coerce")

    # Segment intervals from segment_start events (the last one is "end").
    segs = ev[ev.kind == "segment_start"].reset_index(drop=True)
    sched = {s[0]: s for s in meta["schedule"]}
    seg_names = [s[0] for s in meta["schedule"]]
    tel["segment"] = -1
    tel["training_on"] = 0.0
    for i in range(len(segs) - 1):
        a, b, nm = segs.time[i], segs.time[i + 1], segs.segment[i]
        m = (tel.index >= a) & (tel.index < b)
        tel.loc[m, "segment"] = seg_names.index(nm)
        mode = sched[nm][2]
        if mode == "train":
            tel.loc[m, "training_on"] = 1.0
        elif mode == "square":
            el = (tel.index[m] - a).total_seconds()
            tel.loc[m, "training_on"] = ((el // 60) % 2 == 0).astype(float)
    tel["ckpt_scheduled"] = 0.0
    cs, ce = list(ev[ev.kind == "ckpt_start"].time), list(ev[ev.kind == "ckpt_end"].time)
    for a, b in zip(cs, ce):
        tel.loc[(tel.index >= a) & (tel.index <= b), "ckpt_scheduled"] = 1.0

    # Power from the cumulative energy counter: it updates about every 94 ms,
    # about 5x faster than the board power reading (~0.5 s), and resolves
    # the checkpoint dips that the power reading smooths away.
    e_j = tel["energy_mj"].to_numpy() / 1000.0
    mono = tel["mono_s"].to_numpy()
    ch = np.flatnonzero(np.diff(e_j) != 0) + 1
    pe = np.full(len(tel), np.nan)
    pe[ch[1:]] = np.diff(e_j[ch]) / np.diff(mono[ch])
    tel["power_energy_w"] = pd.Series(pe, index=tel.index).bfill(limit=5)

    cols = {
        "mono_s": {"unit": "s", "description": "host monotonic clock"},
        "power_w": {"unit": "W", "description": "GPU board power (NVML nvmlDeviceGetPowerUsage)"},
        "energy_mj": {"unit": "mJ", "description": "cumulative GPU energy counter (NVML)"},
        "power_energy_w": {"unit": "W", "description": "power from the NVML cumulative energy counter (differenced at counter updates)"},
        "temp_c": {"unit": "degC", "description": "GPU core temperature (NVML)"},
        "sm_clock_mhz": {"unit": "MHz", "description": "SM clock"},
        "util_gpu_pct": {"unit": "%", "description": "GPU utilization"},
        "util_mem_pct": {"unit": "%", "description": "memory controller utilization"},
        "enforced_limit_w": {"unit": "W", "description": "enforced power limit"},
        "clock_event_reasons": {"unit": "bitmask", "description": "NVML clock event reasons"},
        "segment": {"unit": "code", "description": "index into meta schedule"},
        "training_on": {"unit": "0/1", "description": "scripted load active (known future)"},
        "ckpt_scheduled": {"unit": "0/1", "description": "checkpoint write interval (from load log)"},
    }
    source = {
        "title": f"Laptop GPU collection run {name}",
        "url": "local: scripts/laptop_collect.py on the author's laptop",
        "license": "same as this repo",
        "retrieved_utc": meta.get("ended_utc"),
        "raw_sha256": {f: sha(raw / f) for f in ("telemetry.csv", "events.csv", "meta.json")},
        "device": {k: meta[k] for k in ("gpu", "driver", "limits_w", "default_limit_w", "torch", "params")},
        "schedule": meta["schedule"],
    }
    base = {"columns": cols, "source": source, "data_kind": "measured",
            "producer": "scripts/process_laptop_run.py"}
    save_series(tel, f"laptop_{name}_10hz", {**base, "freq": "100ms (irregular, ~10 Hz)"})
    one = tel.resample("1s").mean()
    # Exact 1 s average power from the energy counter, interpolated at second boundaries.
    edges = one.index.union([one.index[-1] + pd.Timedelta("1s")])
    t_ns = tel.index.asi8.astype(float)
    e_at = np.interp(edges.asi8.astype(float), t_ns, e_j)
    one["power_energy_w"] = np.diff(e_at)
    inside = (edges[:-1] >= tel.index[0]) & (edges[1:] <= tel.index[-1])
    one.loc[~inside, "power_energy_w"] = np.nan
    one["segment"] = tel["segment"].resample("1s").max()
    save_series(one, f"laptop_{name}_1s", {**base, "freq": "1s"})
    evc = {"segment": {"unit": "name", "description": "segment"},
           "kind": {"unit": "name", "description": "event kind"},
           "step": {"unit": "count", "description": "training step"},
           "mono_s": {"unit": "s", "description": "host monotonic clock"}}
    e = ev.set_index("time")[["segment", "kind", "step", "mono_s"]]
    save_series(e, f"laptop_{name}_events", {**base, "columns": evc, "freq": "event"})
    print(f"rows 10hz={len(tel)} 1s={len(one)} events={len(e)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else PROCESSED)
