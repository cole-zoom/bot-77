"""Scalar indexes on the columns the data browser and the build scripts filter by (2026-10-22): without them every
`WHERE match_id = ...` on detections scans 32M rows. Re-run after rebuilding a table.

  uv run python scripts/build_indexes.py
"""
import time

import lancedb

INDEXES = {  # table: [(column, index type)]
    "detections": [("match_id", "BTREE"), ("run_id", "BITMAP"), ("t", "BTREE")],
    "moments": [("moment_id", "BTREE"), ("game_id", "BTREE"), ("kind", "BITMAP"), ("source", "BITMAP")],
    "labels": [("moment_id", "BTREE"), ("source", "BITMAP")],
    "plays": [("game_id", "BTREE"), ("source", "BITMAP")],
    "matches": [("match_id", "BTREE")],
    "hud_states": [("match_id", "BTREE")],
    "events": [("match_id", "BTREE")],
    "videos": [("video_id", "BTREE")],
}
db = lancedb.connect("data/lancedb")
for table, cols in INDEXES.items():
    t = db.open_table(table)
    for col, kind in cols:
        t0 = time.time()
        t.create_scalar_index(col, index_type=kind, replace=True)
        print(f"{table}.{col} ({kind}) {time.time() - t0:.1f} s", flush=True)
