"""MIT Supercloud Dataset (labelled part, 2022-01 release): per-job GPU telemetry.

Source: https://mit-supercloud-dataset.s3.amazonaws.com/datacenter-challenge/202201/
Paper : Samsi et al., "The MIT Supercloud Dataset", IEEE HPEC 2021, arXiv:2108.02037.
License: CC BY-NC-ND 4.0 (file LICENSE in the bucket); see data/SOURCES.md.

Raw layout: ``gpu/<dddd>/<jobid>-r<rack>-n<node>.csv``, one file per job per node, one row per
GPU per sample (nvidia-smi, about 9.4 samples/s per GPU, timestamps are epoch seconds).
Nodes are Xeon Gold 6248 hosts with two NVIDIA V100 32 GB GPUs (paper, Section III).
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BUCKET = "https://mit-supercloud-dataset.s3.amazonaws.com/"
PREFIX = "datacenter-challenge/202201/"
LICENSE_NAME = "CC BY-NC-ND 4.0"
LICENSE_URL = "http://creativecommons.org/licenses/by-nc-nd/4.0/"
PAPER_URL = "https://arxiv.org/abs/2108.02037"

GPU_COLUMNS = {
    "power_w": ("W", "nvidia-smi power.draw, mean of the samples in the 1 s bin"),
    "power_max_w": ("W", "nvidia-smi power.draw, maximum of the samples in the 1 s bin"),
    "temp_c": ("degC", "nvidia-smi temperature.gpu, mean over the 1 s bin"),
    "temp_mem_c": ("degC", "nvidia-smi temperature.memory (HBM), mean over the 1 s bin"),
    "util_pct": ("%", "nvidia-smi utilization.gpu (SM busy fraction of the sample period), mean over the 1 s bin"),
    "mem_util_pct": ("%", "nvidia-smi utilization.memory (memory-controller busy fraction), mean over the 1 s bin"),
    "mem_used_mib": ("MiB", "nvidia-smi memory.used, mean over the 1 s bin"),
}
_RAW_TO_SRC = {
    "power_w": "power_draw_W",
    "power_max_w": "power_draw_W",
    "temp_c": "temperature_gpu",
    "temp_mem_c": "temperature_memory",
    "util_pct": "utilization_gpu_pct",
    "mem_util_pct": "utilization_memory_pct",
    "mem_used_mib": "memory_used_MiB",
}
NODE_COLUMNS = {
    "node_power_w": ("W", "sum over the node's GPUs of power_w (NaN if any GPU is missing in that second)"),
    "node_temp_c_max": ("degC", "maximum over the node's GPUs of temp_c"),
    "node_util_pct_mean": ("%", "mean over the node's GPUs of util_pct"),
    "node_mem_used_mib": ("MiB", "sum over the node's GPUs of mem_used_mib"),
}
MIN_ROWS = 1801  # 1801 one-second rows span 1800 s: telemetry shorter than 30 min is not kept
USECOLS = [
    "timestamp", "gpu_index", "utilization_gpu_pct", "utilization_memory_pct",
    "memory_used_MiB", "temperature_gpu", "temperature_memory", "power_draw_W",
]


def list_prefix(prefix: str, delimiter: str | None = "/") -> tuple[list[tuple[str, int]], list[str]]:
    """S3 REST listing (ListObjectsV2) of the public bucket via plain HTTPS GET."""
    from ._common import http_get

    keys: list[tuple[str, int]] = []
    prefixes: list[str] = []
    token = None
    while True:
        params: dict[str, Any] = {"list-type": 2, "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        x = http_get(BUCKET, params=params).text
        keys += [(html.unescape(k), int(s)) for k, s in re.findall(r"<Key>(.*?)</Key><LastModified>.*?</LastModified><ETag>.*?</ETag><Size>(\d+)</Size>", x)]
        prefixes += re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix>", x)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m:
            return keys, prefixes
        token = html.unescape(m.group(1))


def parse_key(key: str) -> tuple[int, str]:
    stem = key.rsplit("/", 1)[-1][: -len(".csv")]
    job, node = stem.split("-", 1)
    return int(job), node


def select_files(
    files: pd.DataFrame,
    slurm: pd.DataFrame,
    labelled: pd.DataFrame,
    *,
    n_labelled: int = 200,
    n_other: int = 260,
    min_duration_s: int = 1800,
    max_duration_s: int = 8 * 3600,
    max_file_bytes: int = 25_000_000,
    max_total_bytes: int = 2_000_000_000,
    max_nodes_per_job: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Seeded selection of GPU jobs: >= 30 min, spread over nodes and (for labelled jobs) over model types.

    ``files`` has columns key,size,id_job,node. Returns the rows of ``files`` for the chosen jobs
    (all node-files of each chosen job).
    """
    rng = np.random.default_rng(seed)
    s = slurm[["id_job", "time_start", "time_end", "nodes_alloc"]].copy()
    s["dur"] = s.time_end - s.time_start
    per_job = files.groupby("id_job").agg(n_files=("key", "size"), max_size=("size", "max"), tot=("size", "sum")).reset_index()
    j = per_job.merge(s, on="id_job").merge(labelled, on="id_job", how="left")
    ok = j[
        (j.dur >= min_duration_s) & (j.dur <= max_duration_s) & (j.nodes_alloc <= max_nodes_per_job)
        & (j.n_files == j.nodes_alloc) & (j.max_size <= max_file_bytes)
    ].copy()
    first_node = files.sort_values("key").groupby("id_job").node.first().rename("first_node")
    ok = ok.merge(first_node, on="id_job")
    ok = ok.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)

    def round_robin(pool: pd.DataFrame, group: str, n: int, budget: int) -> list[int]:
        queues = {g: list(x.id_job) for g, x in pool.groupby(group)}
        order = list(queues)
        rng.shuffle(order)
        chosen: list[int] = []
        used = 0
        sizes = dict(zip(pool.id_job, pool.tot))
        while len(chosen) < n and any(queues.values()):
            for g in order:
                if queues[g] and len(chosen) < n:
                    jid = queues[g].pop(0)
                    if used + sizes[jid] <= budget:
                        chosen.append(jid)
                        used += sizes[jid]
        return chosen

    lab = ok[ok.model.notna()]
    oth = ok[ok.model.isna()]
    c_lab = round_robin(lab, "model", n_labelled, max_total_bytes // 2)
    used = int(ok[ok.id_job.isin(c_lab)].tot.sum())
    c_oth = round_robin(oth, "first_node", n_other, max_total_bytes - used)
    chosen = set(c_lab) | set(c_oth)
    return files[files.id_job.isin(chosen)].sort_values("key").reset_index(drop=True)


def process_csv(path: Path) -> pd.DataFrame | None:
    """Raw nvidia-smi CSV -> 1 s UTC grid with per-GPU columns ``g<k>_<metric>`` plus node aggregates."""
    d = pd.read_csv(path, usecols=USECOLS, engine="pyarrow")
    for c in USECOLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["timestamp", "gpu_index"])
    if d.empty:
        return None
    d["sec"] = np.floor(d.timestamp).astype("int64")
    d["gpu_index"] = d.gpu_index.astype("int64")
    g = d.groupby(["gpu_index", "sec"])
    mean = g[["power_draw_W", "temperature_gpu", "temperature_memory", "utilization_gpu_pct", "utilization_memory_pct", "memory_used_MiB"]].mean()
    pmax = g["power_draw_W"].max().rename("power_max_w")
    agg = mean.join(pmax)
    t0, t1 = int(d.sec.min()), int(d.sec.max())
    idx = pd.date_range(pd.to_datetime(t0, unit="s", utc=True), pd.to_datetime(t1, unit="s", utc=True), freq="1s", name="time")
    out: dict[str, np.ndarray] = {}
    gpus = sorted(int(x) for x in d.gpu_index.unique())
    for k in gpus:
        a = agg.xs(k, level="gpu_index").reindex(np.arange(t0, t1 + 1))
        for m, src in _RAW_TO_SRC.items():
            col = "power_max_w" if m == "power_max_w" else src
            out[f"g{k}_{m}"] = a[col].to_numpy(dtype="float64")
    df = pd.DataFrame(out, index=idx)
    if len(df) < MIN_ROWS:
        return None
    p = df[[f"g{k}_power_w" for k in gpus]].to_numpy()
    df["node_power_w"] = p.sum(axis=1)  # NaN if any GPU missing
    df["node_temp_c_max"] = np.fmax.reduce([df[f"g{k}_temp_c"].to_numpy() for k in gpus])
    df["node_util_pct_mean"] = np.mean([df[f"g{k}_util_pct"].to_numpy() for k in gpus], axis=0)
    df["node_mem_used_mib"] = np.sum([df[f"g{k}_mem_used_mib"].to_numpy() for k in gpus], axis=0)
    return df.astype("float32")


def column_meta(gpus: list[int]) -> dict[str, dict[str, str]]:
    cols: dict[str, dict[str, str]] = {}
    for k in gpus:
        for m, (u, desc) in GPU_COLUMNS.items():
            cols[f"g{k}_{m}"] = {"unit": u, "description": f"GPU {k} (nvidia-smi gpu_index {k}): {desc}"}
    for m, (u, desc) in NODE_COLUMNS.items():
        cols[m] = {"unit": u, "description": desc}
    return cols


def series_name(job_id: int, node: str) -> str:
    return f"gpu_{job_id}_{node}"
