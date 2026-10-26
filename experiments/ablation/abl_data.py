"""Datasets for the covariate ablations: simulated (e003 sim) and real (public, measured laptop).

Each dataset fixes its evaluation geometry (context, horizon, horizon buckets, origins) in a `Scale` that was declared
before any score was computed. Origins are rolling, on a held-out final segment of each series:
  * simulator: the last three of the seven simulated days (test seeds 100-109, the fixed test split);
  * grid and carbon series: 2026-05-01 onward (fixed test split);
  * laptop: run2 (test split); run1 (train/calibration run, zero-shot model, no fitting) is reported separately.
No parameter of any forecaster is fitted on a test series.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from abl_engine import Batch, Scale
from heatwave.forecast import covariates as cv

# ------------------------------------------------------------------------------------------------ simulator (SIMULATED)
SIM_KINDS = {
    "site_power_1h": dict(
        key="site_mean_power_w_1h", level="site", res=3600.0, start=96, stride=1,
        scale=Scale("sim_site_power_1h", 72, 24, ((1, 3), (4, 12), (13, 24)), 24, "1 h"),
        what="site power, hourly mean (W)",
    ),
    "site_power_1min": dict(
        key="site_power_w_1min", level="site", res=60.0, start=5760, stride=20,
        scale=Scale("sim_site_power_1min", 2880, 240, ((1, 15), (16, 60), (61, 240)), 1440, "1 min"),
        what="site power, 1-minute mean (W)",
    ),
    "rack_temp_1min": dict(
        key="rack_temp_c_1min", level="rack", res=60.0, start=5760, stride=20,
        scale=Scale("sim_rack_temp_1min", 2880, 240, ((1, 15), (16, 60), (61, 240)), 1440, "1 min"),
        what="rack air temperature, 1-minute mean (degC)",
    ),
    "rack_power_1s": dict(
        key="rack_power_w_1s", level="rack", res=1.0, start=345600, stride=900,
        scale=Scale("sim_rack_power_1s", 2048, 256, ((1, 16), (17, 64), (65, 256)), 1, "1 s"),
        what="rack power at 1 s (W)",
    ),
}


_BUILDERS: dict = {}


def get_builder(scenario: dict, seed: int, variant: str) -> "cv.SimCovariateBuilder":
    """One covariate builder (and its plan cache) per simulated world, shared by all datasets and, for the
    `literal` variant (one world for every seed), by all seeds."""
    key = (variant, 0 if variant == "literal" else int(seed))
    if key not in _BUILDERS:
        _BUILDERS[key] = cv.SimCovariateBuilder(scenario, key[1] if variant == "literal" else seed, variant)
    return _BUILDERS[key]


class SimDataset:
    """One simulator quantity over several test seeds. Streams are (seed, site-or-rack)."""

    data_kind = "simulated"

    def __init__(self, kind: str, variant: str, seeds, cache_dir: Path, scenario: dict, *,
                 stride_mult: int = 1, units: Optional[list] = None) -> None:
        spec = SIM_KINDS[kind]
        self.kind, self.variant, self.seeds = kind, variant, list(seeds)
        self.spec, self.scale = spec, spec["scale"]
        self.name = f"{spec['scale'].name}[{variant}]"
        self.what = spec["what"]
        self.series, self.builders = [], []
        for s in self.seeds:
            z = np.load(Path(cache_dir) / f"sim_{variant}_{s}.npz")
            self.series.append(np.asarray(z[spec["key"]], dtype=np.float64))
            self.unit_ids = [str(u) for u in z["rack_ids" if spec["level"] == "rack" else "site_ids"]]
            self.builders.append(get_builder(scenario, s, variant))
        self.units = list(range(len(self.unit_ids))) if units is None else list(units)
        self.n_total = self.series[0].shape[0]
        self.origin_stride = spec["stride"] * stride_mult
        self.origins = np.arange(spec["start"], self.n_total - self.scale.horizon + 1, self.origin_stride)
        self.stream_ids = [f"seed{s}/{self.unit_ids[u]}" for s in self.seeds for u in self.units]
        self.cluster = np.repeat(np.arange(len(self.seeds)), len(self.units))
        self.groups = list(cv.SIM_GROUPS)
        if variant == "literal":
            keys = {b.world_key() for b in self.builders}
            assert len(keys) == 1, "literal variant must share one world"

    @property
    def n_streams(self) -> int:
        return len(self.stream_ids)

    @property
    def n_origins(self) -> int:
        return len(self.origins)

    def subsampling(self) -> dict:
        return {"origin_stride_steps": int(self.origin_stride), "n_origins": int(self.n_origins),
                "seeds": self.seeds, "units": [self.unit_ids[u] for u in self.units]}

    def iter_batches(self, batch_size: int) -> Iterator[Batch]:
        sc = self.scale
        T, H = sc.context, sc.horizon
        res, level = self.spec["res"], self.spec["level"]
        pend = dict(stream=[], origin=[], ctx=[], y=[], cov={g: [] for g in self.groups})
        n_units = len(self.units)
        shared = None
        for oi, o in enumerate(self.origins):
            for si, b in enumerate(self.builders):
                if self.variant == "literal" and shared is not None and shared[0] == o:
                    cov = shared[1]
                else:
                    cov = b.window(level, res, int(o), T, H)
                    if self.variant == "literal":
                        shared = (o, cov)
                for ui, u in enumerate(self.units):
                    y = self.series[si][:, u]
                    pend["stream"].append(si * n_units + ui)
                    pend["origin"].append(oi)
                    pend["ctx"].append(y[o - T:o])
                    pend["y"].append(y[o:o + H])
                    for g in self.groups:
                        pend["cov"][g].append(cov[g][:, u, :])
            while len(pend["stream"]) >= batch_size:
                yield self._take(pend, batch_size)
        if pend["stream"]:
            yield self._take(pend, len(pend["stream"]))

    @staticmethod
    def _take(pend: dict, n: int) -> Batch:
        b = Batch(
            np.array(pend["stream"][:n]), np.array(pend["origin"][:n]), np.stack(pend["ctx"][:n]),
            np.stack(pend["y"][:n]), {g: np.stack(v[:n]) for g, v in pend["cov"].items()},
        )
        for k in ("stream", "origin", "ctx", "y"):
            del pend[k][:n]
        for g in pend["cov"]:
            del pend["cov"][g][:n]
        return b


# --------------------------------------------------------------------------------------- real series on a regular grid
class GridDataset:
    """One real series with covariate groups supplied per origin. One stream."""

    n_streams = 1

    def __init__(self, name: str, scale: Scale, y: np.ndarray, origins: np.ndarray, stride: int, groups: dict,
                 data_kind: str, what: str, notes: dict) -> None:
        self.name, self.scale, self.y = name, scale, np.asarray(y, dtype=np.float64)
        self.origins, self.origin_stride = np.asarray(origins), stride
        self.groups_fn: dict[str, Callable] = groups
        self.groups = list(groups)
        self.data_kind, self.what, self.notes = data_kind, what, notes
        self.stream_ids = [name]
        self.cluster = None

    @property
    def n_origins(self) -> int:
        return len(self.origins)

    def subsampling(self) -> dict:
        return {"origin_stride_steps": int(self.origin_stride), "n_origins": int(self.n_origins), **self.notes}

    def iter_batches(self, batch_size: int) -> Iterator[Batch]:
        T, H = self.scale.context, self.scale.horizon
        for s in range(0, len(self.origins), batch_size):
            ob = self.origins[s:s + batch_size]
            yield Batch(
                np.zeros(len(ob), dtype=int), np.arange(s, s + len(ob)),
                np.stack([self.y[o - T:o] for o in ob]), np.stack([self.y[o:o + H] for o in ob]),
                {g: np.stack([fn(int(o), T, H) for o in ob]) for g, fn in self.groups_fn.items()},
            )


def _origins(grid: pd.DatetimeIndex, start: pd.Timestamp, stride: int, T: int, H: int, end_idx: Optional[int] = None) -> np.ndarray:
    i0 = max(int(grid.searchsorted(start)), T)
    n = len(grid) if end_idx is None else end_idx
    return np.arange(i0, n - H + 1, stride)


TEST_START = pd.Timestamp("2026-05-01", tz="UTC")

SCALE_RT = Scale("caiso_rt_lmp_5min", 576, 288, ((1, 12), (13, 72), (73, 288)), 288, "5 min",
                 ("1-12 (1 h)", "13-72 (1-6 h)", "73-288 (6-24 h)"))
SCALE_CAISO_CARBON = Scale("caiso_carbon_5min", 576, 288, ((1, 12), (13, 72), (73, 288)), 288, "5 min",
                           ("1-12 (1 h)", "13-72 (1-6 h)", "73-288 (6-24 h)"))
SCALE_GB = Scale("gb_carbon_30min", 336, 48, ((1, 6), (7, 24), (25, 48)), 48, "30 min",
                 ("1-6 (3 h)", "7-24 (3-12 h)", "25-48 (12-24 h)"))
SCALE_LAP1 = Scale("laptop_1s", 600, 60, ((1, 5), (6, 20), (21, 60)), 1, "1 s")
SCALE_LAP10 = Scale("laptop_10hz", 600, 100, ((1, 10), (11, 40), (41, 100)), 1, "0.1 s")

WX_COLS = ["temperature_2m_c", "wet_bulb_2m_c", "shortwave_radiation_w_m2", "wind_speed_10m_mps"]


def _reg(s: pd.Series, freq: str) -> pd.Series:
    """Complete regular UTC grid (NaN where missing)."""
    full = pd.date_range(s.index[0], s.index[-1], freq=freq, tz="UTC")
    return s.reindex(full)


def build_real(stride_mult: int = 1) -> dict:
    """All real-data ablation datasets: name -> (GridDataset, config list [(cfg, groups)], group labels)."""
    from heatwave.data import public as P

    out = {}
    wx_day = P.load_weather("bayarea", kind="forecast", vintage="dayahead")
    wx_lon = P.load_weather("london", kind="forecast", vintage="dayahead")
    lead24 = pd.Timedelta("24h")

    def cov_fn(c: cv.IssueTimeCovariate):
        return lambda i, T, H: c.window(i, T, H)

    # ---- CAISO real-time LMP (public): day-ahead price, weather forecast, day-ahead demand forecast
    rt = _reg(P.load_price("caiso", "rtm")["lmp_usd_per_mwh"], "5min")
    grid = rt.index
    dam = P.load_price("caiso", "dam")["lmp_usd_per_mwh"]
    c_da = cv.day_ahead_price_covariate(dam, grid)
    c_wx = cv.archived_forecast_covariate(wx_day, grid, WX_COLS, issue_lead=lead24, label="open-meteo previous_day1 (bayarea)",
                                          note="Open-Meteo previous_day1: forecast issued one day before the valid time")
    dem_df = _demand_frame(P)
    c_dem = _demand_cov(dem_df, grid)
    sc = SCALE_RT
    stride = 12 * stride_mult
    org = _origins(grid, TEST_START, stride, sc.context, sc.horizon)
    out["caiso_rt_lmp"] = (
        GridDataset("caiso_rt_lmp_5min", sc, rt.to_numpy(), org, stride,
                    {"DA": cov_fn(c_da), "WX": cov_fn(c_wx), "DEM": cov_fn(c_dem)}, "public",
                    "CAISO NP15 real-time LMP, 5 min (USD/MWh)", {"test_start": str(TEST_START), "grid_start": str(grid[0]), "grid_end": str(grid[-1])}),
        [("none", ()), ("DA", ("DA",)), ("WX", ("WX",)), ("DEM", ("DEM",)), ("DA+WX+DEM", ("DA", "WX", "DEM"))],
        {"DA": "CAISO day-ahead LMP", "WX": "Bay Area weather forecast (day-ahead vintage)", "DEM": "CAISO day-ahead demand forecast"},
    )

    # ---- CAISO in-state carbon intensity (public): weather forecast, demand forecast
    cc = _reg(P.load_carbon("caiso")["intensity_g_per_kwh"], "5min")
    cc = cc[:rt.index[-1]]
    g2 = cc.index
    c_wx2 = cv.archived_forecast_covariate(wx_day, g2, WX_COLS, issue_lead=lead24, label="open-meteo previous_day1 (bayarea)")
    c_dem2 = _demand_cov(dem_df, g2)
    org2 = _origins(g2, TEST_START, stride, SCALE_CAISO_CARBON.context, SCALE_CAISO_CARBON.horizon)
    out["caiso_carbon"] = (
        GridDataset("caiso_carbon_5min", SCALE_CAISO_CARBON, cc.to_numpy(), org2, stride,
                    {"WX": cov_fn(c_wx2), "DEM": cov_fn(c_dem2)}, "public",
                    "CAISO in-state carbon intensity, 5 min (gCO2/kWh)", {"test_start": str(TEST_START)}),
        [("none", ()), ("WX", ("WX",)), ("DEM", ("DEM",)), ("WX+DEM", ("WX", "DEM"))],
        {"WX": "Bay Area weather forecast (day-ahead vintage)", "DEM": "CAISO day-ahead demand forecast"},
    )

    # ---- GB carbon intensity (public): archived forecast of unknown vintage, weather forecast
    a = P.load_carbon("gb", "actual")["intensity_g_per_kwh"]
    f = P.load_carbon("gb", "forecast")["intensity_g_per_kwh"]
    ga = _reg(a, "30min")
    gf = f.reindex(ga.index)
    g3 = ga.index
    c_f = cv.archived_forecast_covariate(
        pd.DataFrame({"f": gf}), g3, ["f"], issue_lead=lead24, label="GB archived intensity forecast (vintage undocumented)",
        vintage_verified=False, note="assumed issue lead 24 h; the API does not document the vintage")
    c_wx3 = cv.archived_forecast_covariate(wx_lon, g3, WX_COLS, issue_lead=lead24, label="open-meteo previous_day1 (london)")
    stride3 = 2 * stride_mult
    org3 = _origins(g3, TEST_START, stride3, SCALE_GB.context, SCALE_GB.horizon)
    out["gb_carbon"] = (
        GridDataset("gb_carbon_30min", SCALE_GB, ga.to_numpy(), org3, stride3,
                    {"F": cov_fn(c_f), "WX": cov_fn(c_wx3)}, "public",
                    "GB carbon intensity, actual, 30 min (gCO2/kWh)", {"test_start": str(TEST_START)}),
        [("none", ()), ("F", ("F",)), ("WX", ("WX",)), ("F+WX", ("F", "WX"))],
        {"F": "archived GB intensity forecast (vintage undocumented, assumed 24 h lead)", "WX": "London weather forecast (day-ahead vintage)"},
    )
    return out


def _demand_frame(P) -> pd.DataFrame:
    dem = P.load_forecast_covariates("caiso")["demand_forecast_day_ahead_mw"]
    dem = dem.reindex(pd.date_range(dem.index[0], dem.index[-1], freq="5min", tz="UTC"))
    return pd.DataFrame({"demand_forecast_day_ahead_mw": dem.to_numpy()}, index=dem.index).ffill()


def _demand_cov(dem_df: pd.DataFrame, grid: pd.DatetimeIndex) -> cv.IssueTimeCovariate:
    return cv.IssueTimeCovariate(
        grid, _hold(dem_df, grid), cv.publication_instants(grid), "caiso_day_ahead_demand_forecast", vintage_verified=False,
        note="CAISO Outlook day-ahead demand forecast; publication time assumed equal to the day-ahead price (13:00 Pacific, day before)")


def _hold(df: pd.DataFrame, grid: pd.DatetimeIndex) -> np.ndarray:
    pos = np.searchsorted(df.index.asi8, grid.asi8, side="right") - 1
    v = np.full((len(grid), df.shape[1]), np.nan)
    ok = pos >= 0
    v[ok] = df.to_numpy(dtype=float)[pos[ok]]
    return v


# ---------------------------------------------------------------------------------------------- laptop (MEASURED)
def laptop_d_nominal(run: str = "run1") -> float:
    """Nominal checkpoint write duration (s): median logged duration in the calibration run (run1)."""
    from heatwave.data.schema import load_series

    ev = load_series(f"laptop_{run}_events")[0]
    s = ev[ev.kind == "ckpt_start"].index.asi8
    e = ev[ev.kind == "ckpt_end"].index.asi8
    n = min(len(s), len(e))
    return float(np.median((e[:n] - s[:n]) / 1e9))


def build_laptop(run: str, freq: str, quantity: str, d_nom: float, stride_mult: int = 1):
    """Laptop dataset for one run: quantity 'power' (energy-counter power, W) or 'temp' (degC).

    Covariate groups: S = scheduled load-active indicator (schedule + observed segment start), K = checkpoint indicator
    projected from the schedule and the last observed checkpoint, L = logged checkpoint intervals of the whole run
    (contain the realised start and duration: NOT known at the origin; reported only as a labelled comparison).
    """
    from heatwave.data.schema import load_series

    df, meta = load_series(f"laptop_{run}_{'1s' if freq == '1s' else '10hz'}")
    ev = load_series(f"laptop_{run}_events")[0]
    segments = meta["source"]["schedule"]
    sched = cv.laptop_schedule_from_events(segments, ev, d_nom)
    if freq == "1s":
        res_s, rule, scale, stride = 1.0, "1s", SCALE_LAP1, 5 * stride_mult
        grid = df.index
    else:
        res_s, rule, scale, stride = 0.1, "100ms", SCALE_LAP10, 20 * stride_mult
        d = df[["power_energy_w", "temp_c", "training_on", "ckpt_scheduled"]].resample(rule).mean()
        d = d.reindex(pd.date_range(d.index[0], d.index[-1], freq=rule))
        df, grid = d, d.index
    col = {"power": "power_energy_w", "temp": "temp_c"}[quantity]
    y = df[col].interpolate(limit=20, limit_direction="both").to_numpy()
    # covariates are passed as 0/1 indicators (the model's indicator path): load active = at least half of the bin,
    # checkpoint = any overlap with the bin
    on = (df["training_on"].interpolate(limit=20, limit_direction="both").to_numpy() >= 0.5).astype(float)
    ck = (df["ckpt_scheduled"].interpolate(limit=20, limit_direction="both").to_numpy() > 0.0).astype(float)
    # events are timestamped in UTC with sub-second precision; grid bins start on the grid timestamps
    gidx = pd.DatetimeIndex(grid)

    def fn_S(i, T, H):
        fut = (cv.laptop_training_on_future(sched, gidx, res_s, i, H) >= 0.5).astype(float)
        return np.concatenate([on[i - T:i], fut])[None, :]

    def fn_K(i, T, H):
        fut = (cv.laptop_ckpt_future(sched, gidx, res_s, i, H) > 0.0).astype(float)
        return np.concatenate([ck[i - T:i], fut])[None, :]

    def fn_L(i, T, H):
        return ck[i - T:i + H][None, :]

    org = np.arange(scale.context, len(y) - scale.horizon + 1, stride)
    name = f"laptop_{quantity}_{freq}_{run}"
    ds = GridDataset(name, scale, y, org, stride, {"S": fn_S, "K": fn_K, "L": fn_L}, "measured",
                     f"laptop RTX 3060 Laptop GPU {col}, {freq} ({'test run' if run == 'run2' else 'calibration run, zero-shot model'})",
                     {"run": run, "ckpt_nominal_duration_s": d_nom})
    cfgs = [("none", ()), ("S", ("S",)), ("K", ("K",)), ("S+K", ("S", "K")), ("S+L_realised", ("S", "L"))]
    labels = {"S": "scheduled load-active indicator (pre-written schedule)",
              "K": "checkpoint indicator projected from the schedule and last observed checkpoint",
              "L": "logged checkpoint intervals of the whole run (contain realised start and duration; not known at the origin)"}
    return ds, cfgs, labels
