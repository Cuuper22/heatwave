#!/usr/bin/env python
"""Collect the 2026 high-resolution AI-training power dataset on 8-GPU H100 and B200 nodes (20 ms sampling).

Source: Elsayed, Al-Obaidi, Farag, "Characterization of high-resolution AI data center training workloads on single and
multiple GPU nodes", Scientific Data 13, 1268 (2026), doi:10.1038/s41597-026-07496-6; data: figshare
doi:10.6084/m9.figshare.31654879 (CC BY 4.0, per the figshare record).
Only the Node_Dataset (32 unique sessions: 16 H100 + 16 B200 8-GPU nodes, 15 minutes each, per-GPU power / temperature /
utilization / memory via pynvml) is processed. The Single_Machine_Dataset (RTX 3060 workstation, 40 sessions) stays in the raw
zip and is not processed.

Raw   : data/raw/elsayed2026/ds.zip (gitignored, 210 MB); sha256 in data/processed/public_gpu_node/raw_manifest.json
Output: data/processed/public_gpu_node_20ms/<session>.*   20 ms grid: per-GPU power_w, node_power_w
        data/processed/public_gpu_node_1s/<session>.*     1 s means (and power max) of every column
        data/processed/public_gpu_node_sessions.parquet   one row per session (hardware, task, hyperparameters from the file name)
The recorded timestamps carry no time zone; they are interpreted as UTC (not verified by the dataset documentation).
Usage : python scripts/fetch_node_power_h100_b200.py
"""

from __future__ import annotations

import hashlib
import io
import re
import sys
import zipfile

import numpy as np
import pandas as pd

from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, Manifest, download, utc_now
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "elsayed2026"
MAN = PROCESSED_ROOT / "public_gpu_node"
OUT20 = PROCESSED_ROOT / "public_gpu_node_20ms"
OUT1 = PROCESSED_ROOT / "public_gpu_node_1s"
URL = "https://ndownloader.figshare.com/files/62953543"
LANDING = "https://doi.org/10.6084/m9.figshare.31654879"
LICENSE = "CC BY 4.0 (figshare record license field; https://creativecommons.org/licenses/by/4.0/)"
PER_GPU = {
    "utilization_percent": ("util_pct", "%", "GPU utilization"),
    "mem_utilization": ("mem_util_pct", "%", "GPU memory-controller utilization"),
    "Power_TDP": ("power_pct_limit", "%", "board power as percent of the power limit (as reported in the dataset)"),
    "mem_used_MB": ("mem_used_mb", "MB", "GPU memory in use"),
    "power_W": ("power_w", "W", "board power draw (pynvml)"),
    "temp_C": ("temp_c", "degC", "GPU temperature (pynvml)"),
}


def parse_name(stem: str) -> dict:
    d = {"hardware": re.search(r"(H100|B200)", stem).group(1), "task": "image_generation" if stem.startswith("Image") else "llm_training"}
    for key, pat in [("model_size", r"ModelSize(\d+)"), ("image_size", r"ImageSize(\d+)"), ("batch_size", r"BatchSize(\d+)"),
                     ("llama_params", r"Lamma(\d+B)"), ("seq_length", r"SeqLength(\d+)"), ("deepspeed_stage", r"Deepspeedds_z(\d)")]:
        m = re.search(pat, stem)
        d[key] = m.group(1) if m else None
    return d


def to_grid(raw: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    t = pd.to_datetime(raw["timestamp"])
    t0 = t.iloc[0].floor(f"{step_ms}ms")
    k = np.rint((t - t0).dt.total_seconds().to_numpy() * 1000.0 / step_ms).astype("int64")
    g = raw.drop(columns=["timestamp"]).groupby(k).mean()
    g = g.reindex(np.arange(0, int(k.max()) + 1))
    g.index = pd.DatetimeIndex(t0.tz_localize("UTC") + pd.to_timedelta(g.index * step_ms, unit="ms"), name="time")
    return g


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "ds.zip"
    download(URL, zpath, expected_size=209685111)
    man = Manifest(MAN / "raw_manifest.json")
    man.record("ds.zip", zpath, URL)
    man.save()
    ds_sha = man.sha("ds.zip")
    src = {"title": "High-resolution AI Data Center Training Workloads Dataset (Elsayed et al., Sci Data 2026), Node_Dataset", "url": LANDING,
           "license": LICENSE, "retrieved_utc": man.data["ds.zip"]["retrieved_utc"], "raw_sha256": {"ds.zip": ds_sha}}
    seen: dict[str, str] = {}
    rows = []
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if "/Node_Dataset/" in n and n.lower().endswith(".csv"))
        for n in names:
            b = z.read(n)
            sha = hashlib.sha256(b).hexdigest()
            stem = n.rsplit("/", 1)[-1][:-4]
            if sha in seen:
                continue
            seen[sha] = stem
            raw = pd.read_csv(io.BytesIO(b))
            gpus = sorted({int(m.group(1)) for c in raw.columns if (m := re.match(r"gpu(\d+)_", c))})
            ren, units = {}, {}
            for k in gpus:
                for suf, (nm, u, desc) in PER_GPU.items():
                    ren[f"gpu{k}_{suf}"] = f"g{k}_{nm}"
                    units[f"g{k}_{nm}"] = (u, f"GPU {k}: {desc}")
            ren.update({"cpu_utilization_percent": "cpu_util_pct", "cpu_freq_MHz": "cpu_freq_mhz"})
            units.update({"cpu_util_pct": ("%", "host CPU utilization (psutil)"), "cpu_freq_mhz": ("MHz", "host CPU frequency (psutil)")})
            keep = ["timestamp"] + list(ren)
            d = raw[keep].rename(columns=ren)
            # 20 ms grid: per-GPU power and node sum only
            g = to_grid(d, 20)
            pcols = [f"g{k}_power_w" for k in gpus]
            f20 = g[pcols].copy()
            f20["node_power_w"] = f20[pcols].sum(axis=1, min_count=len(pcols))
            u20 = {c: ("W", units[c][1]) for c in pcols}
            u20["node_power_w"] = ("W", "sum of the GPUs' power_w (NaN if any GPU is missing in that 20 ms slot)")
            base_meta = {"data_kind": "public", "producer": "scripts/fetch_node_power_h100_b200.py", "source": src,
                         "session": {"file": n.split("/", 2)[-1], **parse_name(stem), "n_gpus": len(gpus)},
                         "notes": "Timestamps carry no time zone in the dataset; interpreted as UTC. Samples are about every 20 ms with jitter; "
                                  "each sample is assigned to its nearest 20 ms slot (slots without a sample are NaN, slots with two are averaged)."}
            save_series(f20.astype("float32"), stem, {**base_meta, "freq": "20ms", "columns": {c: {"unit": u, "description": dsc} for c, (u, dsc) in u20.items()}}, root=OUT20)
            # 1 s means of everything, plus 1 s max of power
            gg = g.copy()
            gg["node_power_w"] = gg[pcols].sum(axis=1, min_count=len(pcols))
            one = gg.resample("1s").mean()
            for c in pcols + ["node_power_w"]:
                one[c.replace("_power_w", "_power_max_w") if c != "node_power_w" else "node_power_max_w"] = gg[c].resample("1s").max()
            cm = {c: {"unit": u, "description": dsc} for c, (u, dsc) in units.items() if c in one.columns}
            cm["node_power_w"] = {"unit": "W", "description": "sum of the GPUs' power_w (1 s mean of the 20 ms sums)"}
            cm["node_power_max_w"] = {"unit": "W", "description": "maximum over the 1 s of the 20 ms node power sums"}
            for c in pcols:
                cm[c.replace("_power_w", "_power_max_w")] = {"unit": "W", "description": f"GPU {c.split('_')[0][1:]}: maximum board power over the 1 s"}
            save_series(one.astype("float32")[list(cm)], stem, {**base_meta, "freq": "1s", "columns": cm}, root=OUT1)
            rows.append({"session": stem, **parse_name(stem), "n_gpus": len(gpus), "rows_20ms": int(len(f20)), "rows_1s": int(len(one)),
                         "start_utc": f20.index[0].isoformat(), "raw_file": n.split("/", 2)[-1], "raw_content_sha256": sha,
                         "frac_missing_20ms_node_power": float(f20["node_power_w"].isna().mean())})
            print(stem, len(f20), flush=True)
    pd.DataFrame(rows).to_parquet(PROCESSED_ROOT / "public_gpu_node_sessions.parquet")
    print(f"{len(rows)} unique sessions (identical files in several folders were stored once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
