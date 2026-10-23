"""Phase 2 check (docs/training-architecture.md, 2026-10-22): rebuild A's training rows from the new tables (`moments`
play moments + `plays` history) and compare them field by field with `move_samples` (built by build_moves.py from
events + hud_states). Writes them to `move_samples_m` so A v1 can also be retrained from them:

  uv run python scripts/check_moments.py
  uv run python scripts/train_moves.py --table move_samples_m --out /tmp/... --eval heldback   (expect 46.6%)
"""
import math
from collections import defaultdict

import lancedb
import pyarrow as pa

from bot77.features import CAND_FEATURES, CTX_FEATURES
from bot77.features.datasets import card_rows

db = lancedb.connect("data/lancedb")
cards = {c["id"]: c for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()}
rows = card_rows(db, cards)  # bot77.features.datasets
db.create_table("move_samples_m", pa.Table.from_pylist(rows), mode="overwrite")

old = {(r["game"], r["decision"], r["cand_slot"]): r for r in db.open_table("move_samples").to_arrow().to_pylist()}
new = {(r["game"], r["decision"], r["cand_slot"]): r for r in rows}
feats = CAND_FEATURES + CTX_FEATURES + ["chosen", "split"]
diff = defaultdict(int)
for key in old.keys() & new.keys():
    for f in feats:
        a, b = old[key][f], new[key][f]
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and (math.isclose(a, b, abs_tol=1e-4) or (math.isnan(a) and math.isnan(b))))):
            diff[f] += 1
print(f"move_samples {len(old):,} rows, rebuilt {len(new):,}; only old {len(old.keys() - new.keys()):,}, "
      f"only new {len(new.keys() - old.keys()):,}; rows compared {len(old.keys() & new.keys()):,}")
print("fields that differ (rows):", dict(diff) or "none")
