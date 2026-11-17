"""Match results and top/bottom pairs for train_bc.py --match-meta, so training runs on a box without the tables (2026-11-17).
  uv run --no-sync python scripts/pod/train_meta.py OUT.json
"""
import json
import sys

from bot77.sim import store

results = {m["match_id"]: m["result"] for m in store.read("matches", "source IN ('replay_tv', 'replay_bl')")}
pairs = {m["match_id"]: m["pair_match_id"] for m in store.read("matches", "pov_side = 'bottom'")}
json.dump({"results": results, "pairs": pairs}, open(sys.argv[1], "w"))
print(len(results), "results,", len(pairs), "pairs")
