"""Result provenance: every committed number traces to a committed script.

``write_result`` wraps a result dict with:
  - the producing script path and its sha256
  - the git commit of this repo (and "dirty" if uncommitted changes exist)
  - sha256 of every input file
  - a ``data_kind`` label: "measured", "public", "simulated" or "mixed"
  - a UTC timestamp

Rules for all Heatwave code: never hand-type a number into a result or doc;
produce it with a committed script and write it through this function.
Simulated results must say so (data_kind="simulated").
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

DATA_KINDS = ("measured", "public", "simulated", "mixed")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def git_state(cwd: Path = REPO_ROOT) -> dict[str, str]:
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    dirty = bool(_git("status", "--porcelain", cwd=cwd))
    return {"commit": commit, "dirty": str(dirty).lower()}


def _clean(value: Any) -> Any:
    """Make numpy scalars/arrays JSON-safe; NaN/inf become None."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _clean(value.tolist())
        if isinstance(value, np.generic):
            return _clean(value.item())
    except ImportError:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_result(
    out_path: str | Path,
    result: Mapping[str, Any],
    *,
    script: str | Path,
    data_kind: str,
    inputs: Iterable[str | Path] = (),
    notes: str = "",
) -> dict[str, Any]:
    if data_kind not in DATA_KINDS:
        raise ValueError(f"data_kind must be one of {DATA_KINDS}")
    script = Path(script).resolve()
    record = {
        "provenance": {
            "script": str(script.relative_to(REPO_ROOT)) if script.is_relative_to(REPO_ROOT) else str(script),
            "script_sha256": sha256_file(script),
            "git": git_state(),
            "inputs": {
                str(Path(p)): sha256_file(p) for p in inputs if Path(p).exists()
            },
            "data_kind": data_kind,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": notes,
        },
        "result": _clean(dict(result)),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=False, allow_nan=False) + "\n")
    return record
