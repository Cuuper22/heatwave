"""Shared helpers for the public-data fetchers: polite HTTP, hashing, manifests.

Raw downloads live in ``data/raw/<source>/`` (gitignored). The sha256 of every
raw file is written to a manifest that is committed next to the processed
series, so a re-fetch can be checked byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
USER_AGENT = "heatwave-research/0.1 (public-data fetcher; contact: repo owner)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Blocked(RuntimeError):
    """Raised when a host answers 401/403/407 (blocked by the allowlist proxy or the site)."""

    def __init__(self, url: str, status: int):
        super().__init__(f"{status} for {url}")
        self.url = url
        self.status = status


def http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 6,
    backoff: float = 10.0,
    timeout: float = 120.0,
    stream: bool = False,
    session: requests.Session | None = None,
) -> requests.Response:
    """GET with exponential backoff on 429/5xx. 401/403/407 raise ``Blocked`` (no retry)."""
    s = session or requests
    h = {"User-Agent": USER_AGENT, **(headers or {})}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, headers=h, timeout=timeout, stream=stream)
        except requests.RequestException as exc:  # network hiccup
            last = exc
            time.sleep(backoff * (2**attempt) / 4)
            continue
        if r.status_code in (401, 403, 407):
            raise Blocked(r.url, r.status_code)
        if r.status_code == 429 or r.status_code >= 500:
            wait = backoff * (2**attempt)
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                wait = max(wait, float(ra))
            last = RuntimeError(f"{r.status_code} for {r.url}")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"giving up on {url}: {last}")


def download(url: str, dest: Path, *, expected_size: int | None = None, **kw: Any) -> str:
    """Download ``url`` to ``dest`` (skip if present with the expected size). Returns sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return sha256_file(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with http_get(url, stream=True, **kw) as r:
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return sha256_file(dest)


def load_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.JSONEncoder(indent=1, sort_keys=True).encode(obj) + "\n")


class Manifest:
    """sha256 manifest for raw files of one source: {relative_path: {sha256, bytes, retrieved_utc, url}}."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict[str, Any]] = load_json(path, {})

    def record(self, rel: str, file: Path, url: str | None = None) -> str:
        old = self.data.get(rel)
        sha = sha256_file(file)
        size = file.stat().st_size
        if old and old["sha256"] == sha and old["bytes"] == size:
            return sha
        self.data[rel] = {"sha256": sha, "bytes": size, "retrieved_utc": utc_now(), "url": url}
        return sha

    def save(self) -> None:
        write_json(self.path, self.data)

    def sha(self, rel: str) -> str | None:
        e = self.data.get(rel)
        return e["sha256"] if e else None


def run_pool(fn: Callable, items: list, workers: int) -> list:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
