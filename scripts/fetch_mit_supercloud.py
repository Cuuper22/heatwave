#!/usr/bin/env python
"""Collect and process a subset of the MIT Supercloud GPU telemetry (nvidia-smi, ~9.4 Hz).

Re-runnable: raw files already present with the listed size are not downloaded again, their sha256
is checked against the committed manifest, and processed series whose raw sha256 is unchanged are
skipped.

Outputs
  data/raw/mit_supercloud/...                       raw files (gitignored)
  data/processed/public_gpu_mit/raw_manifest.json   sha256 of every raw file used (committed)
  data/processed/public_gpu_mit/gpu_<job>_<node>.parquet + .meta.json   1 s series, one per (job, node)
  data/processed/public_gpu_mit_jobs.parquet + .table.json  jobs table (one row per job); description in public_gpu_mit_jobs.table.json

Usage: python scripts/fetch_mit_supercloud.py [--seed 0] [--n-labelled 200] [--n-other 260]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from heatwave.data.public import mit_supercloud as mit
from heatwave.data.public._common import (
    PROCESSED_ROOT, RAW_ROOT, Manifest, download, load_json, run_pool, sha256_file, utc_now, write_json,
)
from heatwave.data.schema import save_series
from heatwave.provenance import write_result

RAW = RAW_ROOT / "mit_supercloud"
OUT = PROCESSED_ROOT / "public_gpu_mit"
META_FILES = ["slurm-log.csv", "labelled_jobids.csv", "labelled_job_stats.csv", "tres-mapping.txt", "README.md", "LICENSE"]


def _process_one(args: tuple[str, str, int, str, str]):
    """Worker: raw csv -> processed series. Returns a jobs-table row dict or None."""
    key, raw_sha, job_id, node, retrieved = args
    name = mit.series_name(job_id, node)
    meta_path = OUT / f"{name}.meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old["source"]["raw_sha256"].get(key) == raw_sha:
            if old["job"]["rows_1s"] < mit.MIN_ROWS:
                return None
            return old["job"] | {"series": name}
    df = mit.process_csv(RAW / key)
    if df is None:
        return None
    gpus = sorted({int(c[1:].split("_", 1)[0]) for c in df.columns if c.startswith("g") and c[1].isdigit()})
    obs = df["node_power_w"].isna().mean()
    job = {
        "job_id": job_id, "node_id": node, "n_gpus_on_node": len(gpus),
        "start_utc": df.index[0].isoformat(), "end_utc": df.index[-1].isoformat(),
        "rows_1s": int(len(df)), "frac_missing_node_power": float(obs),
    }
    meta = {
        "freq": "1s",
        "columns": mit.column_meta(gpus),
        "source": {
            "title": "The MIT Supercloud Dataset (labelled part, 202201 release), per-job nvidia-smi GPU time series",
            "url": mit.BUCKET + mit.PREFIX + key.split("202201/")[-1] if "202201/" in key else mit.BUCKET + key,
            "paper": mit.PAPER_URL,
            "license": f"{mit.LICENSE_NAME} ({mit.LICENSE_URL})",
            "retrieved_utc": retrieved,
            "raw_sha256": {key: raw_sha},
        },
        "data_kind": "public",
        "producer": "scripts/fetch_mit_supercloud.py",
        "job": job,
        "notes": (
            "Raw nvidia-smi samples (about every 106 ms) binned to 1 s: mean of samples in the second "
            "(power_max_w: maximum). Seconds with no sample are NaN. Hardware: two NVIDIA V100 32 GB per node."
        ),
    }
    save_series(df, name, meta, root=OUT)
    return job | {"series": name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-labelled", type=int, default=200)
    ap.add_argument("--n-other", type=int, default=260)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(OUT / "raw_manifest.json")

    # 1. small metadata files
    for f in META_FILES:
        dest = RAW / f
        download(mit.BUCKET + mit.PREFIX + f, dest)
        manifest.record(f, dest, mit.BUCKET + mit.PREFIX + f)
    slurm = pd.read_csv(RAW / "slurm-log.csv")
    labelled = pd.read_csv(RAW / "labelled_jobids.csv")

    # 2. bucket listing of the GPU time-series files (cached)
    listing_path = RAW / "gpu_listing.json.gz"
    if listing_path.exists():
        rows = json.loads(gzip.open(listing_path).read())
    else:
        rows = []
        for i in range(100):
            keys, _ = mit.list_prefix(f"{mit.PREFIX}gpu/{i:04d}/", delimiter=None)
            rows += keys
        with gzip.open(listing_path, "wt") as fh:
            fh.write(json.JSONEncoder().encode(rows))
    files = pd.DataFrame(rows, columns=["key", "size"])
    ids = files.key.map(mit.parse_key)
    files["id_job"] = [a for a, _ in ids]
    files["node"] = [b for _, b in ids]
    print(f"bucket: {len(files)} GPU files, {files['size'].sum() / 1e9:.1f} GB, {files.id_job.nunique()} jobs")

    # 3. seeded selection
    sel = mit.select_files(files, slurm, labelled, n_labelled=args.n_labelled, n_other=args.n_other, seed=args.seed)
    print(f"selected: {sel.id_job.nunique()} jobs, {len(sel)} files, {sel['size'].sum() / 1e9:.2f} GB")

    # 4. download (4 parallel connections)
    def fetch(row):
        dest = RAW / row.key
        download(mit.BUCKET + row.key, dest, expected_size=int(row.size))
        return row.key, manifest.record(row.key, dest, mit.BUCKET + row.key)

    res = run_pool(fetch, list(sel.itertuples()), workers=4)
    keep_keys = set(META_FILES) | set(sel.key)
    manifest.data = {k: v for k, v in manifest.data.items() if k in keep_keys}
    manifest.save()
    shas = dict(res)

    # 5. process
    work = [(r.key, shas[r.key], int(r.id_job), r.node, manifest.data[r.key]["retrieved_utc"]) for r in sel.itertuples()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        out = [o for o in ex.map(_process_one, work, chunksize=4) if o]
    series = pd.DataFrame(out)
    # remove processed series left over from an earlier selection
    keep = set(series.series)
    for p in OUT.glob("gpu_*.parquet"):
        if p.stem not in keep:
            p.unlink()
            p.with_suffix(".meta.json").unlink(missing_ok=True)

    # 6. jobs table (one row per job; nodes of multi-node jobs listed)
    s = slurm.set_index("id_job")
    lab = labelled.set_index("id_job").model
    rows_j = []
    for jid, g in series.groupby("job_id"):
        r = s.loc[jid]
        rows_j.append({
            "job_id": int(jid),
            "n_nodes": int(len(g)),
            "n_gpus": int(g.n_gpus_on_node.sum()),
            "node_ids": sorted(g.node_id),
            "series": sorted(g.series),
            "start_utc": pd.Timestamp(g.start_utc.min()),
            "end_utc": pd.Timestamp(g.end_utc.max()),
            "duration_s": float((pd.Timestamp(g.end_utc.max()) - pd.Timestamp(g.start_utc.min())).total_seconds()),
            "slurm_start_utc": pd.to_datetime(int(r.time_start), unit="s", utc=True),
            "slurm_end_utc": pd.to_datetime(int(r.time_end), unit="s", utc=True),
            "slurm_nodes_alloc": int(r.nodes_alloc),
            "cpus_req": int(r.cpus_req),
            "job_type": r.job_type,
            "model": lab.get(jid, None),
            "frac_missing_node_power": float(g.frac_missing_node_power.mean()),
        })
    jobs = pd.DataFrame(rows_j).sort_values("job_id").reset_index(drop=True)
    for c in ("start_utc", "end_utc"):
        jobs[c] = pd.to_datetime(jobs[c], utc=True)
    jobs.to_parquet(PROCESSED_ROOT / "public_gpu_mit_jobs.parquet")
    write_json(PROCESSED_ROOT / "public_gpu_mit_jobs.table.json", {
        "name": "public_gpu_mit_jobs",
        "description": "One row per selected MIT Supercloud GPU job; series names point into data/processed/public_gpu_mit/.",
        "columns": {c: str(jobs[c].dtype) for c in jobs.columns},
        "source": {"title": "MIT Supercloud Dataset slurm-log.csv + labelled_jobids.csv + GPU time series", "url": mit.BUCKET + mit.PREFIX,
                    "license": f"{mit.LICENSE_NAME} ({mit.LICENSE_URL})", "retrieved_utc": utc_now(),
                    "raw_sha256": {k: manifest.sha(k) for k in META_FILES}},
        "data_kind": "public",
        "producer": "scripts/fetch_mit_supercloud.py",
        "selection": {"seed": args.seed, "n_labelled": args.n_labelled, "n_other": args.n_other,
                       "min_duration_s": 1800, "max_duration_s": 8 * 3600, "max_file_bytes": 25_000_000},
    })
    write_result(PROCESSED_ROOT.parent.parent / "results" / "data" / "public_mit_selection.json", {
        "bucket_gpu_files": int(len(files)), "bucket_gpu_jobs": int(files.id_job.nunique()), "bucket_gpu_bytes": int(files["size"].sum()),
        "slurm_log_rows": int(len(slurm)), "labelled_jobs_in_labels_file": int(len(labelled)),
        "selected_jobs": int(sel.id_job.nunique()), "selected_files": int(len(sel)), "selected_raw_bytes": int(sel["size"].sum()),
        "selected_fraction_of_bucket_gpu_jobs": float(sel.id_job.nunique() / files.id_job.nunique()),
        "kept_jobs_after_min_30min_telemetry": int(len(jobs)), "kept_series": int(len(series)),
        "dropped_selected_jobs_shorter_than_30min_telemetry": int(sel.id_job.nunique() - len(jobs)),
        "selection_parameters": {"seed": args.seed, "n_labelled": args.n_labelled, "n_other": args.n_other},
    }, script=__file__, data_kind="public", inputs=[RAW / "slurm-log.csv", RAW / "labelled_jobids.csv"])
    print(f"processed: {len(jobs)} jobs, {len(series)} series -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
