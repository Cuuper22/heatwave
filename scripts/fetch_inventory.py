#!/usr/bin/env python
"""Inventory of every processed public series: rows, start, end, sampling interval, fraction missing.

Writes results/data/public_inventory.json (via write_result, data_kind="public") and refreshes the generated tables
between the AUTO markers of data/SOURCES.md (the prose of that file is written by hand).
Usage: python scripts/fetch_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from heatwave.data import public
from heatwave.data.public._common import PROCESSED_ROOT, REPO_ROOT
from heatwave.data.schema import list_series, load_series
from heatwave.provenance import write_result

SOURCES = REPO_ROOT / "data" / "SOURCES.md"
BEGIN, END = "<!-- AUTO:BEGIN -->", "<!-- AUTO:END -->"


def describe(df: pd.DataFrame, meta: dict) -> dict:
    step = np.diff(df.index.asi8) // 10**9
    miss = df.isna().mean()
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "start": df.index[0].isoformat(), "end": df.index[-1].isoformat(),
        "sampling_interval_s": float(np.median(step)) if len(step) else None,
        "freq": meta["freq"],
        "fraction_missing": float(miss.mean()),
        "fraction_missing_max_column": float(miss.max()),
        "fraction_missing_by_column": {c: float(v) for c, v in miss.items()},
        "data_kind": meta["data_kind"],
    }


def manifest_hash(path: Path) -> str:
    m = json.loads(path.read_text())
    h = hashlib.sha256()
    for k in sorted(m):
        h.update(f"{k} {m[k]['sha256']}\n".encode())
    return h.hexdigest()


def main() -> int:
    series = {}
    for n in list_series(PROCESSED_ROOT):
        if n.startswith("public_"):
            df, meta = load_series(n)
            series[n] = describe(df, meta)

    jobs = public.load_gpu_jobs()
    gpu = {}
    for name in public.list_gpu_series():
        df, meta = load_series(name, public.GPU_DIR)
        d = describe(df, meta)
        gpu[name] = {k: d[k] for k in ("rows", "start", "end", "sampling_interval_s", "fraction_missing")}
        gpu[name]["fraction_missing_node_power"] = float(df["node_power_w"].isna().mean())
        gpu[name]["n_gpus"] = int(sum(1 for c in df.columns if c.endswith("_power_w") and c.startswith("g") and c[1].isdigit()))
    g = pd.DataFrame(gpu).T
    gpu_summary = {
        "series": int(len(g)), "jobs": int(len(jobs)), "labelled_jobs": int(jobs.model.notna().sum()),
        "distinct_nodes": int(jobs.explode("node_ids").node_ids.nunique()),
        "multi_node_jobs": int((jobs.n_nodes > 1).sum()),
        "n_gpus_total": int(jobs.n_gpus.sum()),
        "total_rows_1s": int(g.rows.sum()),
        "total_gpu_hours": float(sum(r.rows * r.n_gpus for r in g.itertuples()) / 3600.0),
        "duration_s": {k: float(v) for k, v in jobs.duration_s.describe().items()},
        "jobs_ge_1800s": int((jobs.duration_s >= 1800).sum()),
        "fraction_missing_node_power_mean": float(g.fraction_missing_node_power.astype(float).mean()),
        "start_utc": jobs.start_utc.min().isoformat(), "end_utc": jobs.end_utc.max().isoformat(),
    }
    tables = {}
    for n in ("alibaba2023_pods", "alibaba2023_nodes", "alibaba2026_gpu_pods_sample"):
        t = public.load_job_trace(n)
        tables[n] = {"rows": int(len(t)), "columns": list(t.columns)}

    manifests = {p.parent.name: {"files": len(json.loads(p.read_text())), "manifest_sha256": manifest_hash(p)}
                 for p in sorted(PROCESSED_ROOT.glob("public_*/raw_manifest.json"))}
    res = {"series": series, "gpu_summary": gpu_summary, "gpu_series": gpu, "job_trace_tables": tables, "raw_manifests": manifests}
    inputs = [PROCESSED_ROOT / f"{n}.parquet" for n in series] + [PROCESSED_ROOT / "public_gpu_mit_jobs.parquet"]
    write_result(REPO_ROOT / "results" / "data" / "public_inventory.json", res, script=__file__, data_kind="public", inputs=inputs)

    # generated tables for SOURCES.md
    rows = ["| series | freq | rows | start (UTC) | end (UTC) | missing (all cells) | worst column missing |", "|---|---|---:|---|---|---:|---:|"]
    for n, d in series.items():
        rows.append(f"| `{n}` | {d['freq']} | {d['rows']:,} | {d['start'][:19]} | {d['end'][:19]} | {d['fraction_missing']:.4f} | {d['fraction_missing_max_column']:.4f} |")
    s = gpu_summary
    gpu_lines = [
        f"- GPU jobs: {s['jobs']} ({s['labelled_jobs']} with a known model label), {s['series']} per-node series, {s['distinct_nodes']} distinct nodes, "
        f"{s['multi_node_jobs']} multi-node jobs, {s['n_gpus_total']} GPUs, {s['total_rows_1s']:,} one-second rows, {s['total_gpu_hours']:.0f} GPU-hours of telemetry.",
        f"- Observed duration per job (seconds): min {s['duration_s']['min']:.0f}, median {s['duration_s']['50%']:.0f}, max {s['duration_s']['max']:.0f}; jobs of at least 1800 s: {s['jobs_ge_1800s']}.",
        f"- Telemetry window {s['start_utc'][:19]} to {s['end_utc'][:19]} UTC; mean fraction of seconds without node power: {s['fraction_missing_node_power_mean']:.4f}.",
    ]
    man_lines = ["| raw set | files | sha256 of the sorted (path, sha256) list |", "|---|---:|---|"]
    for k, v in manifests.items():
        man_lines.append(f"| `data/processed/{k}/raw_manifest.json` | {v['files']} | `{v['manifest_sha256']}` |")
    auto = "\n".join([BEGIN, "", "### Processed series (generated by scripts/fetch_inventory.py from results/data/public_inventory.json)", "", *rows, "",
                      "### MIT Supercloud GPU subset", "", *gpu_lines, "",
                      "### Job trace tables", "", *[f"- `{k}`: {v['rows']:,} rows" for k, v in tables.items()], "",
                      "### Raw file hashes", "", "Each manifest lists the sha256 and byte size of every raw file used, plus the retrieval time and URL.", "", *man_lines, "", END])
    text = SOURCES.read_text() if SOURCES.exists() else f"# Data sources\n\n{BEGIN}\n{END}\n"
    if BEGIN in text:
        text = text[: text.index(BEGIN)] + auto + text[text.index(END) + len(END):]
    else:
        text += "\n" + auto + "\n"
    SOURCES.write_text(text)
    print(f"inventory: {len(series)} series, {len(gpu)} gpu series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
