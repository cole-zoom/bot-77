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

from bot77.moves import CAND_FEATURES, CTX_FEATURES, candidates

db = lancedb.connect("data/lancedb")
cards = {c["id"]: c for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()}
names = {c["name"]: i for i, c in cards.items()}
plays = defaultdict(list)
for p in db.open_table("plays").search().where("source = 'hand'").select(["game_id", "side", "t", "card", "kind"]).limit(10**7) \
        .to_arrow().to_pylist():
    plays[(p["game_id"], p["side"])].append({"t": p["t"], "card": "__ability" if p["kind"] == "ability" else p["card"]})
for v in plays.values():
    v.sort(key=lambda p: p["t"])
# human replays only: replay_bot's bottom player is the bot, never imitation data
moms = db.open_table("moments").search().where("kind = 'play' AND source IN ('replay_tv', 'replay_bl')").select(
    ["game_id", "side", "split", "league", "t", "t_match", "decision", "hand", "next", "elixir", "multiplier"]).limit(10**7).to_arrow().to_pylist()
lab = {}
rows = []
for m in moms:
    own = plays[(m["game_id"], m["side"])]
    opp = plays[(m["game_id"], "top" if m["side"] == "bottom" else "bottom")]
    k = m["decision"]
    last = {}
    for p in own[:k]:
        if p["card"] and names.get(p["card"]) is not None:
            last[names[p["card"]]] = p["t"]
    h = {"hand": m["hand"], "next": m["next"], "elixir": m["elixir"], "t_match": m["t_match"], "multiplier": m["multiplier"]}
    cid = names[own[k]["card"]]
    for r in candidates(h, m["t"], own[:k], opp, last, cards, names):
        rows.append({**r, "decision": k, "t": m["t"], "chosen": int(r["cand_card"] == cid), "match_id": m["game_id"],
                     "side": m["side"], "split": m["split"], "league": m["league"] if m["league"] is not None else -1,
                     "game": f"{m['game_id']}:{m['side']}"})
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
