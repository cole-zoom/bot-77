"""Phase 2 of docs/training-architecture.md (2026-10-22): `moments` (one player, one point in time, everything they saw)
and the replay / bot `labels` on them (reviews: scripts/import_reviews.py).

Moments:
  play   every play of either player in a keyed replay, with the HUD read just before it (the same filter as A's
         decision rows: the hand must be read cleanly and show the played card), and every bot play;
  idle   a replay player who could afford a card in hand but didn't play in the next second: every 1 s between plays
         (not within 0.5 s after a play, while the hand animates). These teach "wait".
Columns: moment_id, game_id, match_id (the match whose hand is visible), side, source, split, league, video_id, t,
t_match, kind, decision (the play's index among the player's plays, A's numbering), hand (4 ids), next, elixir,
multiplier, board_frame (a detector frame was found within 1.5 s), units (list of {card, cls, team own/opp, x, y, conf} in the player's frame: bot77.board), towers
({own_l, own_r, opp_l, opp_r}), board_run (detector run), board_partial (bot games before full-board logging).
Labels (source replay / bot): action (card name or "wait"), card_id, x, y (player frame), xy_kind (box: where the unit
was first detected, its box centre, for replay plays; tap: where the screen was tapped, ~FEET_DY = 1.27 tiles below a
unit's box centre, for the bot and review clicks), weight 1.

  uv run python scripts/build_moments.py
"""
import bisect, json
from collections import Counter, defaultdict
from datetime import timedelta

import lancedb
import pyarrow as pa

from bot77.features import TOWER_CLASSES, to_player_frame
from bot77.features import hud_before
from bot77.features.datasets import BOT_VIDEO_LAG_S

RUN, LAG, IDLE_STEP, AFTER_PLAY = "v5l_e15", 0.25, 1.0, 0.5
db = lancedb.connect("data/lancedb")
cards = {c["id"]: c for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
         .select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()}
names = {c["name"]: i for i, c in cards.items()}
matches = {m["match_id"]: m for m in db.open_table("matches").to_arrow().to_pylist()}
hud_t, det_t, plays_t = db.open_table("hud_states"), db.open_table("detections"), db.open_table("plays")
UNIT = pa.struct([("card", pa.string()), ("cls", pa.string()), ("team", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
                  ("conf", pa.float32())])
TOWERS = pa.struct([("own_l", pa.bool_()), ("own_r", pa.bool_()), ("opp_l", pa.bool_()), ("opp_r", pa.bool_())])


def board(frames, ts, t, side):
    """Units in the last frame >= LAG before t (within 1.5 s) and the princess towers seen in the second before it."""
    i = bisect.bisect_right(ts, t - LAG) - 1
    if i < 0 or t - ts[i] > 1.5:
        return None, {"own_l": False, "own_r": False, "opp_l": False, "opp_r": False}  # no detector frame
    towers = {"own_l": False, "own_r": False, "opp_l": False, "opp_r": False}
    for tf in ts[bisect.bisect_left(ts, t - LAG - 1.0):i + 1]:
        for d in frames[tf]:
            if d["cls"] in TOWER_CLASSES:
                x, _ = to_player_frame(d["x"], 0.0, side)
                team = "own" if (d["team"] == "friendly") == (side == "bottom") else "opp"
                towers[f"{team}_{'l' if x < 9 else 'r'}"] = True
    units = []
    for d in frames[ts[i]]:
        if d["card"] is None:
            continue
        x, y = to_player_frame(d["x"], d["y"], side)
        units.append({"card": d["card"], "cls": d["cls"], "team": "own" if (d["team"] == "friendly") == (side == "bottom") else "opp",
                      "x": round(x, 2), "y": round(y, 2), "conf": round(d["conf"], 2)})
    return units, towers


def hand_ok(h):
    return h is not None and h["hand"] and len(h["hand"]) == 4 and not any(c is None or c < 0 for c in h["hand"])


moments, labels = [], []
games = sorted(m["match_id"] for m in matches.values()
               if m["truth_set"] == "replay" and m["pov_side"] == "bottom" and m["source"].startswith("replay"))
for n, g in enumerate(games):
    frames = defaultdict(list)
    for d in det_t.search().where(f"run_id = '{RUN}' AND match_id = '{g}'").select(["t", "cls", "card", "team", "conf", "x", "y"]) \
            .limit(10**7).to_arrow().to_pylist():
        frames[d["t"]].append(d)
    ts = sorted(frames)
    gp = plays_t.search().where(f"source = 'hand' AND game_id = '{g}'").select(["side", "t", "card", "kind", "x", "y"]) \
        .limit(10**5).to_arrow().to_pylist()
    for side in ("bottom", "top"):
        mid = g if side == "bottom" else matches[g]["pair_match_id"]
        if mid is None:
            continue
        hud = sorted(hud_t.search().where(f"match_id = '{mid}'").limit(10**6).select(
            ["t_video", "t_match", "hand", "next", "elixir", "multiplier"]).to_arrow().to_pylist(), key=lambda r: r["t_video"])
        own = sorted(({"t": p["t"], "card": "__ability" if p["kind"] == "ability" else p["card"], "x": p["x"], "y": p["y"]}
                      for p in gp if p["side"] == side), key=lambda p: p["t"])
        base = {"game_id": g, "match_id": mid, "side": side, "source": matches[g]["source"], "split": matches[g]["split"],
                "league": matches[g]["league"], "video_id": matches[g]["video_id"], "board_run": RUN, "board_partial": False}

        def add(t, h, kind, decision, action, card_id, x, y):
            units, towers = board(frames, ts, t, side)
            frame_ok = units is not None
            mid_ = f"{g}:{side}:{t:.2f}:{kind}"
            moments.append({**base, "moment_id": mid_, "t": t, "t_match": h["t_match"], "kind": kind, "decision": decision,
                            "hand": h["hand"], "next": h["next"], "elixir": h["elixir"], "multiplier": h["multiplier"],
                            "units": units or [], "towers": towers, "board_frame": frame_ok})
            px, py = (to_player_frame(x, y, side, spell=cards.get(card_id, {}).get("type") == "spell")
                      if x is not None else (None, None))
            labels.append({"label_id": f"replay:{mid_}", "moment_id": mid_, "source": "replay", "action": action,
                           "card_id": card_id, "x": px, "y": py, "xy_kind": "box" if px is not None else None,
                           "verdict": None, "note": None, "weight": 1.0,
                           "reviewer": None, "review_id": None})

        # play moments: A's decision filter (bot77.moves.decision_rows)
        for k, p in enumerate(own):
            if p["card"] in (None, "__ability", "Mirror") or names.get(p["card"]) is None:
                continue
            h = hud_before(hud, p["t"])
            cid = names[p["card"]]
            if not hand_ok(h) or cid not in h["hand"]:
                continue
            add(p["t"], h, "play", k, p["card"], cid, p["x"], p["y"])
        # idle moments: could afford a card, didn't play in the next second
        pts = [p["t"] for p in own]
        if not hud:
            continue
        t = hud[0]["t_video"] + 1.0
        while t < hud[-1]["t_video"]:
            j = bisect.bisect_left(pts, t)
            next_play = pts[j] if j < len(pts) else 1e9
            prev_play = pts[j - 1] if j > 0 else -1e9
            h = hud_before(hud, t)
            if next_play - t >= IDLE_STEP and t - prev_play >= AFTER_PLAY and hand_ok(h) and h["elixir"] is not None and \
                    h["elixir"] >= min(cards.get(c, {}).get("elixir") or 99 for c in h["hand"]):
                add(round(t, 2), h, "idle", None, "wait", None, None, None)
            t += IDLE_STEP
    if n % 50 == 0:
        print(f"{n + 1}/{len(games)} games, {len(moments):,} moments", flush=True)

# the bot's plays (no idle moments: the bot only logs plays). Times: video time (session time - the game's start)
bot_start = {g["game_id"]: g["t_start"] for g in db.open_table("bot_games").to_arrow().to_pylist()}
for a in db.open_table("bot_actions").to_arrow().to_pylist():
    a = {**a, "t": a["t"] - bot_start.get(a["game_id"], 0.0) - BOT_VIDEO_LAG_S}
    m = matches.get(a["game_id"], {})
    full = a.get("units") is not None
    if full:
        units = [{"card": u[0], "cls": u[1], "team": "own" if u[2] == "friendly" else "opp", "x": u[3], "y": u[4], "conf": u[5]}
                 for u in json.loads(a["units"]) if u[0]]
    else:  # before 2026-10-22 only the enemy units on our half were logged
        units = [{"card": c, "cls": None, "team": "opp", "x": x, "y": y, "conf": None} for c, x, y in json.loads(a["threats"] or "[]")]
    mid_ = f"{a['game_id']}:bottom:{a['t']:.2f}:play"
    moments.append({"moment_id": mid_, "game_id": a["game_id"], "match_id": a["game_id"], "side": "bottom", "source": "bot",
                    "split": m.get("split"), "league": None, "video_id": a["game_id"], "board_run": "live_v6l", "board_partial": not full,
                    "t": a["t"], "t_match": None if a["time_left_s"] is None else 180 - a["time_left_s"], "kind": "play",
                    "decision": a["k"], "hand": json.loads(a["hand"]), "next": None, "elixir": a["elixir"], "multiplier": None,
                    "units": units, "towers": {"own_l": False, "own_r": False, "opp_l": False, "opp_r": False},
                    "board_frame": True})
    labels.append({"label_id": f"bot:{mid_}", "moment_id": mid_, "source": "bot", "action": a["card"],
                   "card_id": names.get(a["card"]), "x": a["x_tiles"], "y": a["y_tiles"], "xy_kind": "tap", "verdict": None,
                   "note": a["reason"],
                   "weight": 1.0, "reviewer": None, "review_id": None})

schema = pa.schema([("moment_id", pa.string()), ("game_id", pa.string()), ("match_id", pa.string()), ("side", pa.string()),
                    ("source", pa.string()), ("split", pa.string()), ("league", pa.int64()), ("video_id", pa.string()),
                    ("t", pa.float64()), ("t_match", pa.float64()), ("kind", pa.string()), ("decision", pa.int64()),
                    ("hand", pa.list_(pa.int64())), ("next", pa.int64()), ("elixir", pa.float64()), ("multiplier", pa.int64()),
                    ("units", pa.list_(UNIT)), ("towers", TOWERS), ("board_run", pa.string()), ("board_partial", pa.bool_()),
                    ("board_frame", pa.bool_())])
lab_schema = pa.schema([("label_id", pa.string()), ("moment_id", pa.string()), ("source", pa.string()), ("action", pa.string()),
                        ("card_id", pa.int64()), ("x", pa.float64()), ("y", pa.float64()), ("xy_kind", pa.string()),
                        ("verdict", pa.string()),
                        ("note", pa.string()), ("weight", pa.float64()), ("reviewer", pa.string()), ("review_id", pa.string())])
db.create_table("moments", pa.Table.from_pylist(moments, schema=schema), mode="overwrite")
lt = db.create_table("labels", pa.Table.from_pylist(labels, schema=lab_schema), mode="overwrite")
db.open_table("moments").optimize(cleanup_older_than=timedelta(0))
print(f"moments: {len(moments):,}", Counter((m["source"][:6], m["kind"]) for m in moments))
print(f"labels: {len(labels):,}", Counter(l["source"] for l in labels))
print("replay play moments with a position:", sum(1 for l in labels if l["source"] == "replay" and l["x"] is not None))
print("moments without a detector frame:", sum(1 for m in moments if not m["board_frame"]),
      " with an empty board (frame, no units):", sum(1 for m in moments if m["board_frame"] and not m["units"]))
