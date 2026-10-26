#!/usr/bin/env python
"""Re-write every data/processed/public_*/**/*.parquet with zstd (level 19) and byte-stream-split floats.

Keeps committed data small (about 45 % of the default snappy size for these float series). Content is unchanged: the same
DataFrame is read back by pandas / ``heatwave.data.schema.load_series``. Safe to re-run (files already compressed are skipped).
Usage: python scripts/fetch_compress.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from heatwave.data.public._common import PROCESSED_ROOT


def main() -> int:
    before = after = 0
    n = 0
    for f in sorted(PROCESSED_ROOT.rglob("*.parquet")):
        rel = f.relative_to(PROCESSED_ROOT)
        if not rel.parts[0].startswith("public_"):
            continue
        md = pq.ParquetFile(f).metadata
        if md.num_row_groups and md.row_group(0).column(0).compression == "ZSTD":
            continue
        size0 = f.stat().st_size
        t = pq.read_table(f)
        tmp = Path(tempfile.mkstemp(suffix=".parquet", dir=f.parent)[1])
        floats = [fl.name for fl in t.schema if pa.types.is_floating(fl.type)]
        pq.write_table(t, tmp, compression="zstd", compression_level=19, use_byte_stream_split=floats or False,
                       use_dictionary=[c for c in t.column_names if c not in floats])
        tmp.chmod(0o644)
        tmp.replace(f)
        before += size0
        after += f.stat().st_size
        n += 1
    print(f"recompressed {n} files: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
