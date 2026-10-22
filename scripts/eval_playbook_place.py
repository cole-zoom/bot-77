"""Playbook step 2 check (2026-10-21): where to put an answer — the bot's rules vs the playbook — on held-back replays
(never in the playbook). For each real answer with known positions: distance from the true spot and same-lane rate.

  uv run python scripts/eval_playbook_place.py
"""
import json, math, statistics
from pathlib import Path

import lancedb

from bot77.layout import Layout
from bot77.live.policy import Policy
from bot77.live.state import LiveState, Unit
from bot77.playbook import interactions, placed
from bot77.spawns import deploys

db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
names = [c["name"] for c in cards]; ctype = {c["name"]: c["type"] for c in cards}
book = {}
for b in db.open_table("playbook").to_arrow().to_pylist():
    book[(b["opp_card"], b["opp_zone"], b["answer"])] = b
pol = Policy(cards)
ev, dets, vids = db.open_table("events"), db.open_table("unit_detections_v5l_e15"), db.open_table("videos")
plays = lambda mid: sorted(({"t": e["t_video"], "card": e["card"]} for e in ev.search().where(f"match_id = '{mid}' AND kind = 'play'")
                            .limit(10**4).select(["t_video", "card"]).to_arrow().to_pylist() if e["card"]), key=lambda p: p["t"])
res = {"rules": [], "playbook offset": [], "playbook spot": []}
for m in json.loads(Path("config/heldback_replays.json").read_text())["matches"]:
    d = dets.search().where(f"match_id = '{m}'").limit(10**7).to_arrow().to_pylist()
    v = vids.search().where(f"video_id = '{m.rsplit('_m', 1)[0]}'").to_arrow().to_pylist()[0]
    ax0, ay0, ax1, _ = Layout.load(v["layout_id"]).katacr_arena
    dep = deploys(d, names, (ax1 - ax0) / 18, lambda c: ctype.get(c) == "building", lambda c: ctype.get(c) == "spell", (ax0, ay0), spell_team=None)
    for r in interactions(placed(plays(m), dep, "friendly"), placed(plays(m.replace("_m", "_top_m", 1)), dep, "enemy")):
        if r["ans_x"] is None or r["opp_x"] is None or ctype.get(r["answer"]) is None:
            continue
        true = (r["ans_x"], r["ans_y"])
        opp = Unit(r["opp_card"], "", "enemy", r["opp_x"], r["opp_y"], 0.9)
        st = LiveState(0.0, [None] * 4, None, 7.0, 100, 80, 1, "regulation", [opp])
        guesses = {"rules": pol.place(r["answer"], ctype[r["answer"]], st.threats(), st)}
        b = book.get((r["opp_card"], r["opp_zone"], r["answer"])) or book.get((r["opp_card"], "any", r["answer"]))
        if b and b["dx"] is not None:
            guesses["playbook offset"] = (min(max(r["opp_x"] + b["dx"], 0.5), 17.5), min(max(r["opp_y"] + b["dy"], 0.5), 29.0))
            guesses["playbook spot"] = (b["ans_x"], b["ans_y"]) if (r["opp_x"] < 9) == (b["ans_x"] < 9) else (18 - b["ans_x"], b["ans_y"])
        for k, g in guesses.items():
            if g is not None:
                res[k].append((math.dist(g, true), (g[0] < 9) == (true[0] < 9), k))
for k, v in res.items():
    if v:
        print(f"{k:16s} n={len(v):5d}  median distance {statistics.median(x[0] for x in v):4.1f} tiles  "
              f"within 2 tiles {sum(x[0] <= 2 for x in v) / len(v):.0%}  same lane {sum(x[1] for x in v) / len(v):.0%}")
