#!/usr/bin/env python
"""Collect the 2026 NLR (National Laboratory of the Rockies) dataset of generative-AI workload power profiles on H100 nodes.

Source: Vercellino et al., "Dataset of Generative AI Workload Power Profiles", NLR Data Catalog, doi:10.7799/3025227
(https://data.nlr.gov/submissions/312); paper: arXiv:2604.07345. License: NLR Data Catalog licence, free use and copying
provided the licence notice is kept and DOE/NLR/ALLIANCE are credited (https://data.nlr.gov/node/312/license).
Hardware (paper): nodes of four NVIDIA H100 SXM GPUs and two AMD EPYC 9554 CPUs on the Kestrel system.

We process the dataset authors' aggregated files (01_aggregated_datasets): total power (W) = sum over all GPUs (NVML) and CPUs (RAPL)
of all nodes of one job, at 0.2 s (training) or 0.1 s (inference).
  training                  : all 41 runs (Llama-2 70B LoRA fine-tuning and Stable Diffusion; 2, 4, 8, 16 nodes)
  inference_online_finite   : a seeded sample of 60 of the 1026 vLLM serving runs (Llama-3 70B; request rate varies)
The aggregated files carry a relative time axis only (seconds from job start). The series index is that offset added to
1970-01-01T00:00:00Z: it is NOT a calendar time. Absolute start times of the inference runs are kept in the sessions table
(time zone not stated by the authors).

Raw   : data/raw/nlr_genai_power/dataset.zip (gitignored, 1.0 GB); sha256 in data/processed/public_gpu_nlr/raw_manifest.json
Output: data/processed/public_gpu_nlr/<run>.*   and data/processed/public_gpu_nlr_runs.parquet
Usage : python scripts/fetch_nlr_genai_power.py [--seed 0] [--n-inference 60]
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile

import numpy as np
import pandas as pd

from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, Manifest, download
from heatwave.data.schema import save_series

RAW = RAW_ROOT / "nlr_genai_power"
OUT = PROCESSED_ROOT / "public_gpu_nlr"
URL = "https://data.nlr.gov/system/files/312/1775845072-dataset.zip"
LANDING = "https://doi.org/10.7799/3025227"
LICENSE = "NLR Data Catalog licence (https://data.nlr.gov/node/312/license): free use/copy with notice retained and credit to DOE/NLR/ALLIANCE"
AGG = "01_aggregated_datasets/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-inference", type=int, default=60)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "dataset.zip"
    download(URL, zpath)
    man = Manifest(OUT / "raw_manifest.json")
    man.record("dataset.zip", zpath, URL)
    man.save()
    src = {"title": "Dataset of Generative AI Workload Power Profiles (Vercellino et al., NLR 2026), aggregated datasets", "url": LANDING,
           "paper": "https://arxiv.org/abs/2604.07345", "license": LICENSE,
           "retrieved_utc": man.data["dataset.zip"]["retrieved_utc"], "raw_sha256": {"dataset.zip": man.sha("dataset.zip")}}
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    rows = []
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())

        def read_parquet(rel: str) -> pd.DataFrame:
            return pd.read_parquet(io.BytesIO(z.read(AGG + rel)))

        def save(run: str, p: pd.DataFrame, step_ms: int, extra: dict, kind: str) -> None:
            idx = np.rint(p.index.to_numpy(dtype="float64") * 1000.0 / step_ms).astype("int64")
            s = p["power[W]"].groupby(idx).mean().reindex(np.arange(idx.min(), idx.max() + 1))
            df = pd.DataFrame({"total_power_w": s.to_numpy(dtype="float32")},
                              index=pd.DatetimeIndex(epoch + pd.to_timedelta(s.index.to_numpy() * step_ms, unit="ms"), name="time"))
            save_series(df, run, {
                "freq": f"{step_ms}ms",
                "columns": {"total_power_w": {"unit": "W", "description": "sum over all GPUs (NVML) and CPUs (RAPL) of all nodes of the job, as aggregated by the dataset authors"}},
                "source": src, "data_kind": "public", "producer": "scripts/fetch_nlr_genai_power.py",
                "time_origin": "relative: index = 1970-01-01T00:00:00Z + seconds since job start; not a calendar time",
                "run": extra,
                "notes": "Hardware per node: 4x NVIDIA H100 SXM + 2x AMD EPYC 9554 (paper arXiv:2604.07345)."}, root=OUT)
            rows.append({"run": run, "kind": kind, "step_ms": step_ms, "rows": int(len(df)), "frac_missing": float(df.total_power_w.isna().mean()), **extra})

        m = pd.read_csv(io.BytesIO(z.read(AGG + "training/metadata.csv")))
        for r in m.itertuples():
            p = read_parquet(r.path_save)
            run = f"nlr_train_{r.model}_{int(r.nodes)}n_slurm{int(r.slurmid)}"
            save(run, p, 200, {"model": r.model, "nodes": int(r.nodes), "gpus": int(r.nodes) * 4, "slurm_id": int(r.slurmid), "repeat": int(r.repeat)}, "training")
        f = pd.read_csv(io.BytesIO(z.read(AGG + "inference_online_finite_llama3_70b/metadata.csv")))
        f = f[f.path_run.notna()].reset_index(drop=True)
        rng = np.random.default_rng(a.seed)
        pick = np.sort(rng.choice(len(f), size=min(a.n_inference, len(f)), replace=False))
        for i in pick:
            r = f.iloc[i]
            idx_file = r.path_run.split("inference_online_finite_llama3_70b/")[-1]
            p = read_parquet("inference_online_finite_llama3_70b/" + idx_file)
            run = f"nlr_infer_online_{idx_file.split('/')[-1][:-8]}"
            save(run, p, 100, {"model": "llama3_70b_vllm_online", "nodes": None, "gpus": None, "request_rate": float(r["request_rate_x"]),
                                "num_prompts": int(r["num-prompts"]), "start_time_unspecified_tz": str(r["start_time"]),
                                "mean_power_w_authors": float(r["mean_power[W]"]), "peak_power_w_authors": float(r["peak_power[W]"])}, "inference_online")
    pd.DataFrame(rows).to_parquet(PROCESSED_ROOT / "public_gpu_nlr_runs.parquet")
    print(f"saved {len(rows)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
