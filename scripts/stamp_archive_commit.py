"""Record the git commit of a result produced from a `git archive` tarball (e.g. on Colab).

Usage: python scripts/stamp_archive_commit.py <result.json> <commit>

Checks that the result's script_sha256 equals the sha256 of that script at
<commit> (so the stamp cannot point at the wrong code), then sets
provenance.git = {"commit": <commit>, "dirty": "false", "source": "archive",
"stamped_by": "scripts/stamp_archive_commit.py"}. No result number is touched.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main(path: str, commit: str) -> None:
    p = Path(path)
    d = json.loads(p.read_text())
    prov = d["provenance"]
    blob = subprocess.run(["git", "show", f"{commit}:{prov['script']}"], capture_output=True, check=True).stdout
    sha = hashlib.sha256(blob).hexdigest()
    if sha != prov["script_sha256"]:
        raise SystemExit(f"script sha mismatch: result {prov['script_sha256']} vs {commit} {sha}")
    full = subprocess.run(["git", "rev-parse", commit], capture_output=True, text=True, check=True).stdout.strip()
    prov["git"] = {"commit": full, "dirty": "false", "source": "archive", "stamped_by": "scripts/stamp_archive_commit.py"}
    p.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    print("stamped", p, full)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
