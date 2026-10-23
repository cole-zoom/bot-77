"""Where each card was placed (placement model data, 2026-10-22): every play of both players in the keyed replays with
its spot (the first detection of the unit / spell, bot77.playbook.placed over bot77.spawns.deploys, box centre in
arena tiles) and the board right before it (`board_states`), all in the player's frame (bot77.board).
-> LanceDB table `placements`.

  uv run python scripts/build_placements.py   (after build_board_states.py)
"""
import json
from pathlib import Path

import lancedb
import pyarrow as pa

from bot77.features import to_player_frame
from bot77.holdout import held_out
from bot77.layout import Layout
from bot77.playbook import placed
from bot77.spawns import deploys

db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
names = [c["name"] for c in cards]; ctype = {c["name"]: c["type"] for c in cards}
test = set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
heldback = set(json.loads(Path("config/heldback_replays.json").read_text())["matches"])
hold = held_out()
ev, dets, vids = db.open_table("events"), db.open_table("unit_detections_v5l_e15"), db.open_table("videos")
boards = {}
for b in db.open_table("board_states").to_arrow().to_pylist():
    boards[(b["game"], round(b["t"], 2))] = b


def plays(mid):
    return sorted(({"t": e["t_video"], "card": e["card"]} for e in ev.search().where(f"match_id = '{mid}' AND kind = 'play'")
                   .limit(10**4).select(["t_video", "card"]).to_arrow().to_pylist() if e["card"]), key=lambda p: p["t"])


matches = sorted({b["match_id"] for b in boards.values()})
rows, n_plays = [], 0
for n, m in enumerate(matches):
    d = dets.search().where(f"match_id = '{m}'").limit(10**7).to_arrow().to_pylist()
    v = vids.search().where(f"video_id = '{m.rsplit('_m', 1)[0]}'").to_arrow().to_pylist()[0]
    ax0, ay0, ax1, _ = Layout.load(v["layout_id"]).katacr_arena
    dep = deploys(d, names, (ax1 - ax0) / 18, lambda c: ctype.get(c) == "building", lambda c: ctype.get(c) == "spell",
                  (ax0, ay0), spell_team=None)
    split = "test" if m in test else "heldout" if m in hold else "heldback" if m in heldback else "train"
    for side, mid, team in (("bottom", m, "friendly"), ("top", m.replace("_m", "_top_m", 1), "enemy")):
        for p in placed(plays(mid), dep, team):
            n_plays += 1
            b = boards.get((f"{m}:{side}", round(p["t"], 2)))
            if p["x"] is None or b is None:
                continue
            x, y = to_player_frame(p["x"], p["y"], side, spell=ctype.get(p["card"]) == "spell")
            rows.append({"game": f"{m}:{side}", "match_id": m, "side": side, "split": split, "t": p["t"], "card": p["card"],
                         "type": ctype.get(p["card"]), "x": x, "y": y, "units": b["units"], "towers": b["towers"]})
    if n % 50 == 0:
        print(f"{n + 1}/{len(matches)} games, {len(rows)} placed plays of {n_plays}", flush=True)
db.create_table("placements", pa.Table.from_pylist(rows), mode="overwrite")
print(f"placements: {len(rows)} plays with a spot and a board, of {n_plays}")
