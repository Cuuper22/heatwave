#!/usr/bin/env python
"""Collect two older public GPU-cluster job traces: Microsoft Philly (2017) and SenseTime Helios (2020).

These hosts were blocked from the project VM, so this script was run on the
author's laptop and the processed outputs were copied back (see data/SOURCES.md).

  Philly : https://github.com/msr-fiddle/philly-traces (trace-data.tar.gz via Git LFS,
           https://media.githubusercontent.com/media/msr-fiddle/philly-traces/master/trace-data.tar.gz).
           Jeon et al., "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training
           Workloads", USENIX ATC 2019, arXiv:1901.05758. Licence: repository LICENSE file.
  Helios : https://github.com/S-Lab-System-Group/HeliosData (data.zip). Hu et al.,
           "Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU
           Datacenters", SC 2021, doi:10.1145/3458817.3476223. Licence: repository LICENSE.txt.

Outputs (job tables only; Philly per-minute GPU utilization is not kept):
  data/processed/public_job_traces/philly_jobs.parquet
      jobid, status, vc, user, submitted, first_start, last_end, n_attempts, n_gpus (of the last attempt)
  data/processed/public_job_traces/helios_<cluster>_jobs.parquet  (Earth, Saturn, Uranus, Venus)
      GPU jobs only (gpu_num > 0); columns job_id, vc, gpu_num, node_num, state, submit_time,
      start_time, end_time (parsed, naive as published), duration, queue; zstd-compressed
  data/processed/public_job_traces/extra_manifest.json  (URLs, sha256 of downloads, licences, row counts)
Usage: python scripts/fetch_job_traces_extra.py [--skip-philly]
       python scripts/fetch_job_traces_extra.py --compact-only <dir>   (re-apply the compaction to
       full Helios tables written by an earlier version of this script; used once on the VM)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "job_traces_extra"
OUT = ROOT / "data" / "processed" / "public_job_traces"
PHILLY = "https://media.githubusercontent.com/media/msr-fiddle/philly-traces/master/trace-data.tar.gz"
PHILLY_LIC = "https://raw.githubusercontent.com/msr-fiddle/philly-traces/master/LICENSE"
HELIOS = "https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/master/data.zip"
HELIOS_LIC = "https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/master/LICENSE.txt"


def get(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def philly(man: dict) -> None:
    tgz = RAW / "philly-trace-data.tar.gz"
    man["philly"] = {"url": PHILLY, "sha256": get(PHILLY, tgz), "bytes": tgz.stat().st_size}
    lic = requests.get(PHILLY_LIC, timeout=60).text
    man["philly"]["licence_first_line"] = first_line(lic)
    rows = []
    with tarfile.open(tgz, "r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("cluster_job_log"))
        jobs = json.load(tf.extractfile(member))
    for j in jobs:
        att = j.get("attempts") or []
        starts = [a.get("start_time") for a in att if a.get("start_time")]
        ends = [a.get("end_time") for a in att if a.get("end_time")]
        last = att[-1] if att else {}
        n_gpus = sum(len(d.get("gpus", [])) for d in last.get("detail", []))
        rows.append({"jobid": j.get("jobid"), "status": j.get("status"), "vc": j.get("vc"), "user": j.get("user"),
                     "submitted": j.get("submitted_time"), "first_start": min(starts) if starts else None,
                     "last_end": max(ends) if ends else None, "n_attempts": len(att), "n_gpus": n_gpus})
    df = pd.DataFrame(rows)
    for c in ("submitted", "first_start", "last_end"):
        df[c] = pd.to_datetime(df[c], errors="coerce")  # Philly local time (US Pacific), kept naive as published
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "philly_jobs.parquet", compression="zstd")
    man["philly"]["rows"] = int(len(df))
    man["philly"]["member"] = member.name
    man["philly"]["timezone_note"] = "timestamps kept naive as published (cluster local time)"


HELIOS_COLS = ["job_id", "vc", "gpu_num", "node_num", "state", "submit_time", "start_time",
               "end_time", "duration", "queue"]


def compact_helios(df: pd.DataFrame) -> pd.DataFrame:
    """Keep GPU jobs and the scheduling columns; parse times (naive, as published)."""
    out = df.loc[df["gpu_num"] > 0, HELIOS_COLS].copy()
    for c in ("submit_time", "start_time", "end_time"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in ("vc", "state"):
        out[c] = out[c].astype("category")
    return out.reset_index(drop=True)


def helios(man: dict) -> None:
    z = RAW / "helios-data.zip"
    man["helios"] = {"url": HELIOS, "sha256": get(HELIOS, z), "bytes": z.stat().st_size, "clusters": {}}
    lic = requests.get(HELIOS_LIC, timeout=60).text
    man["helios"]["licence_first_line"] = first_line(lic)
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith("cluster_log.csv"):
                cluster = Path(name).parent.name
                df = pd.read_csv(io.BytesIO(zf.read(name)))
                gpu = compact_helios(df)
                OUT.mkdir(parents=True, exist_ok=True)
                gpu.to_parquet(OUT / f"helios_{cluster.lower()}_jobs.parquet", compression="zstd")
                man["helios"]["clusters"][cluster] = {"all_jobs": int(len(df)), "gpu_jobs": int(len(gpu))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-philly", action="store_true")
    ap.add_argument("--compact-only", default=None)
    a = ap.parse_args()
    if a.compact_only:
        src = Path(a.compact_only)
        man = json.loads((src / "extra_manifest.json").read_text())
        for f in sorted(src.glob("helios_*_jobs.parquet")):
            df = pd.read_parquet(f)
            cluster = f.stem.split("_")[1].capitalize()
            gpu = compact_helios(df) if "user" in df.columns else df
            OUT.mkdir(parents=True, exist_ok=True)
            gpu.to_parquet(OUT / f.name, compression="zstd")
            man["helios"]["clusters"][cluster] = {"all_jobs": int(len(df)), "gpu_jobs": int(len(gpu))}
        pd.read_parquet(src / "philly_jobs.parquet").to_parquet(OUT / "philly_jobs.parquet", compression="zstd")
        man["compacted_on"] = "project VM (--compact-only)"
        (OUT / "extra_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        print(json.dumps(man, indent=1))
        return
    man = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fetched_from": "author's laptop (these hosts were blocked from the project VM)"}
    helios(man)
    if not a.skip_philly:
        philly(man)
    (OUT / "extra_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
