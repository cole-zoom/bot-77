"""Phase 2 of docs/training-architecture.md (2026-10-22): the bot-play reviews into LanceDB.

reviews: one row per exported review (file, reviewer, approved_by, exported_at, plays, the raw export).
labels (source review_cole / review_claude) on the bot's play moments:
  OK                 action = the card the bot played (its spot is fine)
  Mistake + a fix    action = the card it should have played or "wait"; x, y = Cole's clicked spot, if any (a tap)
  Mistake, no fix    action = null, verdict bad: the bot's choice was wrong
Claude's reviews count only once Cole has approved them (bot_review_1_claude: approved 2026-10-22, "they all look good").
Re-running replaces the review labels (the other labels stay).

  uv run python scripts/import_reviews.py
"""
import glob, json
from collections import Counter
from pathlib import Path

import lancedb
import pyarrow as pa

from bot77.layout import Layout
from bot77.features.datasets import BOT_VIDEO_LAG_S

APPROVED = {"bot_review_1_claude.json": "cole"}  # Claude exports Cole approved
db = lancedb.connect("data/lancedb")
bot_start = {g["game_id"]: g["t_start"] for g in db.open_table("bot_games").to_arrow().to_pylist()}
acts = {(a["game_id"], a["k"]): {**a, "t": a["t"] - bot_start.get(a["game_id"], 0.0) - BOT_VIDEO_LAG_S}  # video time, as in moments
        for a in db.open_table("bot_actions").to_arrow().to_pylist()}
lay = Layout.load("bluestacks_live")
ax0, ay0, ax1, _ = lay.katacr_arena

reviews, labels = [], []
for f in sorted(glob.glob("data/ground_truth/reviews/bot_review_*.json")):
    ex = json.loads(Path(f).read_text())
    name = Path(f).name
    reviewer = ex.get("reviewer", "cole")
    if reviewer == "claude" and name not in APPROVED:
        print(f"{name}: Claude's review not approved by Cole yet: skipped")
        continue
    rid = name.removesuffix(".json")
    reviews.append({"review_id": rid, "file": f, "reviewer": reviewer, "approved_by": APPROVED.get(name),
                    "exported_at": ex.get("exported_at"), "plays": len(ex["plays"]), "raw": json.dumps(ex)})
    for p in ex["plays"]:
        a = acts.get((p["game"], p["k"]))
        if a is None:
            print("no bot action for", p["id"]); continue
        mid = f"{a['game_id']}:bottom:{a['t']:.2f}:play"
        bad = p["verdict"] == "bad"
        fix = p.get("card") if bad and p.get("card") and p.get("card") != a["card"] else None
        if bad and p.get("card") == a["card"]:
            fix = a["card"]  # right card, wrong spot (e.g. Giant at the bridge): the card stands, the spot doesn't
        x = y = None
        if p.get("spot_px"):
            s = lay.scale(a["frame_w"]); tile = (ax1 - ax0) * s / 18
            x, y = (p["spot_px"][0] - ax0 * s) / tile, (p["spot_px"][1] - ay0 * s) / tile
        labels.append({"label_id": f"review:{rid}:{p['id']}", "moment_id": mid, "source": f"review_{reviewer}",
                       "action": a["card"] if not bad else ("wait" if fix == "wait (no play)" else fix),
                       "card_id": None, "x": x, "y": y, "xy_kind": "tap" if x is not None else None,
                       "verdict": p["verdict"], "note": p.get("note"), "weight": 1.0,
                       "reviewer": reviewer if not APPROVED.get(name) else f"{reviewer} (approved by {APPROVED[name]})",
                       "review_id": rid})
names = {c["name"]: c["id"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["id", "name"]).to_arrow().to_pylist()}
for l in labels:
    l["card_id"] = names.get(l["action"])
db.create_table("reviews", pa.Table.from_pylist(reviews), mode="overwrite")
lt = db.open_table("labels")
lt.delete("source LIKE 'review_%'")
lt.add(pa.Table.from_pylist(labels, schema=lt.schema))
moments = {m["moment_id"] for m in db.open_table("moments").search().where("source = 'bot'").select(["moment_id"])
           .limit(10**6).to_arrow().to_pylist()}
print(f"reviews: {len(reviews)}; review labels: {len(labels)}", Counter((l["source"], l["verdict"]) for l in labels))
print("labels whose moment is missing:", sum(l["moment_id"] not in moments for l in labels),
      " with a clicked spot:", sum(l["x"] is not None for l in labels),
      " actions:", Counter(l["action"] for l in labels if l["verdict"] == "bad").most_common(6))
