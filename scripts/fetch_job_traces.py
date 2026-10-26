#!/usr/bin/env python
"""Collect public GPU-cluster job traces: Alibaba cluster-trace-gpu-v2023 (with start/end times) and v2026 (sizes and durations).

Sources (repository README states: use the traces "however you want as long as it is for research or study purpose"):
  v2023 https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023  (raw.githubusercontent.com is reachable)
        paper: Weng et al., "Beware of Fragmentation", USENIX ATC 2023.
  v2026 https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026  (data on tre-clusterdata.oss-cn-hangzhou.aliyuncs.com)
        paper: Li et al., "Heterogeneity at Hyperscale", USENIX OSDI 2026.

Raw   : data/raw/alibaba/ (gitignored); sha256 in data/processed/public_job_traces/raw_manifest.json
Output: data/processed/public_job_traces/alibaba2023_pods.parquet      8152 pods: GPUs, QoS, create/scheduled/deletion time (s, trace-relative)
        data/processed/public_job_traces/alibaba2023_nodes.parquet     1213 GPU nodes: GPU model and count
        data/processed/public_job_traces/alibaba2026_gpu_pods_sample.parquet   seeded sample of GPU pods: type, size, duration
        results/data/public_job_trace_summary.json                     counts and duration quantiles (written with write_result)
v2026 has no start times (relative day/hour partitions only exist in the huge hourly tables, not fetched).
Usage : python scripts/fetch_job_traces.py [--sample 200000] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from heatwave.data.public._common import PROCESSED_ROOT, RAW_ROOT, REPO_ROOT, Manifest, download, utc_now, write_json
from heatwave.provenance import write_result

RAW = RAW_ROOT / "alibaba"
OUT = PROCESSED_ROOT / "public_job_traces"
GH = "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2023/"
OSS = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/"
ZIP = "asi_opensource_job_execution_summary.zip"
COLS = ["job_type_public", "priority_class", "model_type_public", "gpu_spec_public", "gpu_request", "duration_hours", "schedule_delay_sec"]
CATS = ["job_type_public", "priority_class", "model_type_public", "gpu_spec_public"]
SIZE_BINS = [0, 0.999, 1.001, 2.001, 4.001, 8.001, 16.001, 64.001, 1e9]
SIZE_LABELS = ["<1 (fractional)", "1", "2", "3-4", "5-8", "9-16", "17-64", ">64"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = Manifest(OUT / "raw_manifest.json")

    # v2023: pods and nodes
    for f in ("openb_pod_list_default.csv", "openb_node_list_gpu_node.csv"):
        dest = RAW / "v2023" / f
        download(GH + "csv/" + f, dest)
        man.record(f"v2023/{f}", dest, GH + "csv/" + f)
    pods = pd.read_csv(RAW / "v2023" / "openb_pod_list_default.csv")
    pods = pods.rename(columns={"creation_time": "create_s", "scheduled_time": "scheduled_s", "deletion_time": "deletion_s", "pod_phase": "phase"})
    pods["run_duration_s"] = pods.deletion_s - pods.scheduled_s
    pods = pods.drop(columns=["gpu_spec"])
    pods.to_parquet(OUT / "alibaba2023_pods.parquet")
    nodes = pd.read_csv(RAW / "v2023" / "openb_node_list_gpu_node.csv")
    nodes.to_parquet(OUT / "alibaba2023_nodes.parquet")

    # v2026: execution summary
    zdest = RAW / ZIP
    download(OSS + ZIP, zdest)
    man.record(ZIP, zdest, OSS + ZIP)
    man.save()
    ex = RAW / "v2026"
    pq_path = ex / "asi_opensource_job_execution_summary" / "part-000.parquet"
    if not pq_path.exists():
        with zipfile.ZipFile(zdest) as z:
            z.extractall(ex)
    # stream row groups into compact numpy arrays (40 M rows as pandas objects exceed the memory limit of this VM)
    pf = pq.ParquetFile(pq_path)
    vocab: dict[str, dict[str, int]] = {c: {} for c in CATS}
    parts: dict[str, list[np.ndarray]] = {c: [] for c in COLS}
    for i in range(pf.metadata.num_row_groups):
        g = pf.read_row_group(i, columns=COLS).to_pandas()
        g = g[(g.gpu_request > 0) & g.duration_hours.notna()]
        for c in CATS:
            v = vocab[c]
            parts[c].append(np.array([v.setdefault(str(x), len(v)) for x in g[c].fillna("NA")], dtype=np.int8))
        for c in ("gpu_request", "duration_hours", "schedule_delay_sec"):
            parts[c].append(g[c].to_numpy(dtype="float32"))
    arr = {c: np.concatenate(parts[c]) for c in COLS}
    del parts
    inv = {c: {v: k for k, v in vocab[c].items()} for c in CATS}
    n = len(arr["gpu_request"])
    size_code = (np.digitize(arr["gpu_request"], SIZE_BINS[1:-1], right=False)).astype(np.int8)  # 0..7
    q = [0.1, 0.5, 0.9, 0.99]

    def summarise(mask: np.ndarray) -> dict:
        dur = arr["duration_hours"][mask]
        gr = arr["gpu_request"][mask]
        qs = np.quantile(dur, q)
        return {"pods": int(mask.sum()), "gpu_hours": float((gr.astype("float64") * dur).sum()),
                **{f"duration_h_p{int(k * 100)}": float(v) for k, v in zip(q, qs)}, "mean_gpu_request": float(gr.mean())}

    jt = arr["job_type_public"]
    by_type = {inv["job_type_public"][k]: summarise(jt == k) for k in np.unique(jt)}
    train_code = vocab["job_type_public"].get("training", -1)
    by_size = {SIZE_LABELS[k]: summarise((jt == train_code) & (size_code == k)) for k in range(len(SIZE_LABELS)) if ((jt == train_code) & (size_code == k)).any()}
    by_type_size = {}
    for t in np.unique(jt):
        for k in range(len(SIZE_LABELS)):
            m = (jt == t) & (size_code == k)
            if m.sum() >= 1000:
                by_type_size[f"{inv['job_type_public'][t]}|{SIZE_LABELS[k]}"] = summarise(m)

    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(n, size=min(a.sample, n), replace=False))
    samp = pd.DataFrame({
        **{c: pd.Categorical.from_codes(arr[c][idx], categories=[inv[c][j] for j in range(len(inv[c]))]).astype(str) for c in CATS},
        "gpu_request": arr["gpu_request"][idx], "duration_hours": arr["duration_hours"][idx], "schedule_delay_sec": arr["schedule_delay_sec"][idx],
        "size_bin": [SIZE_LABELS[k] for k in size_code[idx]],
    })
    samp.to_parquet(OUT / "alibaba2026_gpu_pods_sample.parquet")
    d_len = n

    p = pods
    gp = p[p.num_gpu > 0]
    res = {
        "alibaba2023": {
            "pods": int(len(p)), "gpu_pods": int(len(gp)),
            "trace_span_s": float(p.deletion_s.max() - p.create_s.min()),
            "num_gpu_counts": {str(k): int(v) for k, v in p.num_gpu.value_counts().sort_index().items()},
            "phase_counts": {k: int(v) for k, v in p.phase.value_counts().items()},
            "run_duration_s_quantiles": {str(k): float(v) for k, v in gp.run_duration_s.quantile(q).items()},
            "gpu_nodes": int(len(nodes)), "gpus_total": int(nodes.gpu.sum()),
            "gpu_models": {k: int(v) for k, v in nodes.model.value_counts().items()},
        },
        "alibaba2026": {
            "gpu_pods_total": int(d_len), "sample_rows": int(len(samp)), "seed": a.seed,
            "by_job_type": by_type, "training_by_gpu_request_bin": by_size, "by_type_and_size_bin_min1000": by_type_size,
        },
    }
    write_result(REPO_ROOT / "results" / "data" / "public_job_trace_summary.json", res, script=__file__, data_kind="public",
                 inputs=[RAW / "v2023" / "openb_pod_list_default.csv", RAW / "v2023" / "openb_node_list_gpu_node.csv", zdest])
    write_json(OUT / "meta.json", {
        "name": "public_job_traces", "retrieved_utc": utc_now(), "data_kind": "public", "producer": "scripts/fetch_job_traces.py",
        "files": {
            "alibaba2023_pods.parquet": "Alibaba cluster-trace-gpu-v2023 openb_pod_list_default.csv; times are seconds on a trace-relative clock (no calendar date given); run_duration_s = deletion_s - scheduled_s; NaN scheduled_s means never scheduled",
            "alibaba2023_nodes.parquet": "Alibaba cluster-trace-gpu-v2023 openb_node_list_gpu_node.csv",
            "alibaba2026_gpu_pods_sample.parquet": f"seeded (seed={a.seed}) random sample of GPU-requesting pods from asi_opensource_job_execution_summary; no start times in this table",
        },
        "sources": {"v2023": GH, "v2026": OSS + ZIP},
        "license_note": "no licence file in the repository; README: traces may be used 'for research or study purpose'",
        "raw_sha256": {k: man.sha(k) for k in man.data},
    })
    print("pods2023", len(p), "pods2026", d_len, "sample", len(samp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
