"""Phase 1c of docs/training-architecture.md (2026-10-22): one `plays` table for every play we know of, whoever saw it.

source:
  hand      read from the player's visible hand (`events`, kind play / ability): exact card and time. In a replay both
            hands are visible, so both players' plays are exact (the replay answer keys).
  detected  C's opponent pipeline from the arena (`opponent_events`); run_id says which pipeline output. (The sweep
            tables were binned 2026-10-22; the current pipeline's output gets regenerated into here when needed.)
  bot       what the bot played (`bot_actions`): card, the tile it tapped, and why.
Columns: play_id, source, run_id, match_id (as in `matches`), game_id (the game: a replay's bottom match id), side
(bottom / top player of the game), t (video time), t_match, card, card_id, form, kind (play / ability), slot, elixir
before / after, x, y (arena tiles, bottom player's frame: the unit's first detection for hand plays when matched by
build_placements.py, the tap for the bot, C's estimate for detected plays), reason (bot).

  uv run python scripts/build_plays.py
"""
import json
from collections import Counter

import lancedb
import pyarrow as pa

from bot77.features import to_player_frame
from bot77.features.datasets import BOT_VIDEO_LAG_S

db = lancedb.connect("data/lancedb")
cards = {c["name"]: c for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["id", "name", "type"]).to_arrow().to_pylist()}
game = lambda m: m.replace("_top_m", "_m", 1)
side = lambda m: "top" if "_top_m" in m else "bottom"

# hand plays' positions: placements stores the player's frame; back to the bottom frame (the flip is its own inverse)
pos = {}
for r in db.open_table("placements").to_arrow().to_pylist():
    x, y = (r["x"], r["y"]) if r["side"] == "bottom" else to_player_frame(r["x"], r["y"], "top", spell=r["type"] == "spell")
    pos[(r["match_id"], r["side"], round(r["t"], 2), r["card"])] = (x, y)

rows = []
for e in db.open_table("events").search().where("kind IN ('play', 'ability')").limit(10**8).select(
        ["event_id", "match_id", "t_video", "t_match", "kind", "card_id", "card", "form", "slot", "elixir_before",
         "elixir_after"]).to_arrow().to_pylist():
    g, s = game(e["match_id"]), side(e["match_id"])
    x, y = pos.get((g, s, round(e["t_video"], 2), e["card"]), (None, None))
    rows.append({"play_id": f"hand:{e['event_id']}", "source": "hand", "run_id": None, "match_id": e["match_id"], "game_id": g,
                 "side": s, "t": e["t_video"], "t_match": e["t_match"], "card": e["card"], "card_id": e["card_id"],
                 "form": e["form"], "kind": e["kind"], "slot": e["slot"], "elixir_before": e["elixir_before"],
                 "elixir_after": e["elixir_after"], "x": x, "y": y, "reason": None})
# C's output with the current defaults on each evaluation set (restored from the binned sweeps, 2026-10-22: V5 dets,
# fake filter v4, POV spells v2+v5, replay spells v2+v5+v6), plus the old default table
C_RUNS = {"legacy": "opponent_events", "c_dev": "opponent_events_dev_dev_s6_m25", "c_live": "opponent_events_dev_lt_s6_m25",
          "c_heldback": "opponent_events_dev_va6_m256", "c_test": "opponent_events_dev_rt6_v5l_e15"}
for run, tn in C_RUNS.items():
    for e in db.open_table(tn).to_arrow().to_pylist():
        rows.append({"play_id": f"detected:{run}:{e['event_id']}", "source": "detected", "run_id": run,
                     "match_id": e["match_id"], "game_id": game(e["match_id"]), "side": "top" if side(e["match_id"]) == "bottom" else "bottom",
                     "t": e["t_video"], "t_match": e["t_match"], "card": e["card"], "card_id": cards.get(e["card"] or "", {}).get("id"),
                     "form": "evo" if e.get("evo") else "normal", "kind": "play", "slot": None, "elixir_before": None,
                     "elixir_after": None, "x": e["x_tiles"], "y": e["y_tiles"], "reason": None})
# the bot logs session time; its recording starts with the game: video time = t - the game's start (bot_games)
bot_start = {g["game_id"]: g["t_start"] for g in db.open_table("bot_games").to_arrow().to_pylist()}
for a in db.open_table("bot_actions").to_arrow().to_pylist():
    rows.append({"play_id": f"bot:{a['game_id']}:{a['k']}", "source": "bot", "run_id": None, "match_id": a["game_id"],
                 "game_id": a["game_id"], "side": "bottom", "t": a["t"] - bot_start.get(a["game_id"], 0.0) - BOT_VIDEO_LAG_S,
                 "t_match": None if a["time_left_s"] is None else 180 - a["time_left_s"],
                 "card": a["card"], "card_id": cards.get(a["card"], {}).get("id"), "form": None, "kind": "play",
                 "slot": f"slot{a['slot'] + 1}", "elixir_before": a["elixir"], "elixir_after": None, "x": a["x_tiles"],
                 "y": a["y_tiles"], "reason": a["reason"]})
db.create_table("plays", pa.Table.from_pylist(rows), mode="overwrite")

# checks
ev = db.open_table("events")
n_ev = ev.count_rows("kind IN ('play', 'ability')")
c = Counter(r["source"] for r in rows)
print(f"plays: {len(rows):,}  {dict(c)}")
print(f"hand plays {c['hand']:,} = events plays + abilities {n_ev:,}: {'OK' if c['hand'] == n_ev else 'MISMATCH'}")
print(f"bot plays {c['bot']:,} = bot_actions {db.open_table('bot_actions').count_rows():,}")
placed = sum(1 for r in rows if r["source"] == "hand" and r["x"] is not None)
print(f"hand plays with a position: {placed:,} (placements has {len(pos):,})")
ms = {m["match_id"] for m in db.open_table("matches").search().select(["match_id"]).limit(10**6).to_arrow().to_pylist()}
print("plays whose match isn't in `matches`:", Counter(r["source"] for r in rows if r["match_id"] not in ms))
