"""Board at every decision of A's training samples (2026-10-22): for each (game, decision) in `move_samples`, the
units C detected (unit_detections_v5l_e15, every 0.25 s, all 524 keyed replays) in the last frame at least 0.25 s
before the play, in the deciding player's frame (bot77.board), and the princess towers seen in the second before.
-> LanceDB table `board_states` (game, decision, t, units JSON, towers JSON).

  uv run python scripts/build_board_states.py
"""
import bisect, json
from collections import defaultdict

import lancedb
import pyarrow as pa

from bot77.board import TOWER_CLASSES, to_player_frame
from bot77.detect.names import class_to_card
from bot77.layout import Layout

LAG, TOWER_WINDOW = 0.25, 1.0
db = lancedb.connect("data/lancedb")
names = [c["name"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name"])
         .to_arrow().to_pylist()]
dets_t, vids = db.open_table("unit_detections_v5l_e15"), db.open_table("videos")
dec = defaultdict(list)  # match -> [(side, decision, t)]
for r in db.open_table("move_samples").search().where("chosen = 1").select(["match_id", "side", "decision", "t"]) \
        .limit(10**7).to_arrow().to_pylist():
    dec[r["match_id"]].append((r["side"], r["decision"], r["t"]))
card_of = {}
rows, empty = [], 0
for n, (m, ds) in enumerate(sorted(dec.items())):
    v = vids.search().where(f"video_id = '{m.rsplit('_m', 1)[0]}'").to_arrow().to_pylist()[0]
    ax0, ay0, ax1, _ = Layout.load(v["layout_id"]).katacr_arena
    tile = (ax1 - ax0) / 18
    frames = defaultdict(list)
    for d in dets_t.search().where(f"match_id = '{m}'").select(["t_video", "cls", "team", "conf", "x0", "y0", "x1", "y1"]) \
            .limit(10**7).to_arrow().to_pylist():
        frames[d["t_video"]].append(d)
    ts = sorted(frames)
    for side, k, t in ds:
        i = bisect.bisect_right(ts, t - LAG) - 1
        units, towers = [], {}
        if i >= 0 and t - ts[i] <= 1.5:
            for tf in ts[bisect.bisect_left(ts, t - LAG - TOWER_WINDOW):i + 1]:
                for d in frames[tf]:
                    if d["cls"] not in TOWER_CLASSES:
                        continue
                    x, _ = to_player_frame(((d["x0"] + d["x1"]) / 2 - ax0) / tile, 0.0, side)
                    team = "own" if (d["team"] == "friendly") == (side == "bottom") else "opp"
                    towers[f"{team}_{'L' if x < 9 else 'R'}"] = 1
            for d in frames[ts[i]]:
                if d["cls"] not in card_of:
                    card_of[d["cls"]] = class_to_card(d["cls"], names)[0]
                card = card_of[d["cls"]]
                if card is None:
                    continue
                x, y = to_player_frame(((d["x0"] + d["x1"]) / 2 - ax0) / tile, ((d["y0"] + d["y1"]) / 2 - ay0) / tile, side)
                units.append([card, "own" if (d["team"] == "friendly") == (side == "bottom") else "opp", round(x, 2),
                              round(y, 2), round(d["conf"], 2)])
        else:
            empty += 1
        rows.append({"game": f"{m}:{side}", "match_id": m, "side": side, "decision": k, "t": t,
                     "units": json.dumps(units), "towers": json.dumps(towers)})
    if n % 50 == 0:
        print(f"{n + 1}/{len(dec)} games, {len(rows)} decisions ({empty} without a frame)", flush=True)
db.create_table("board_states", pa.Table.from_pylist(rows), mode="overwrite")
print(f"board_states: {len(rows)} decisions, {empty} without a frame in the 1.5 s before")
