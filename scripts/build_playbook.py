"""Build the LanceDB tables `interactions` and `playbook` (bot77.playbook) from the keyed replays' train split (test,
held-out and held-back games are left out, so the playbook can be measured on them later).

  uv run python scripts/build_playbook.py
"""
import glob, json, statistics
from collections import defaultdict
from pathlib import Path

import lancedb
import pyarrow as pa

from bot77.holdout import held_out
from bot77.layout import Layout
from bot77.playbook import interactions, placed
from bot77.spawns import deploys

db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
names = [c["name"] for c in cards]; ctype = {c["name"]: c["type"] for c in cards}
skip = held_out() | set(json.loads(Path("config/heldback_replays.json").read_text())["matches"])
league = {}
for f in glob.glob("data/raw/tv_royale/*/runner_log.jsonl"):
    for line in open(f):
        r = json.loads(line); league[f"tv_{Path(f).parent.name}_r{int(r['i']):02d}"] = r.get("tab", 0)
ev, dets, vids = db.open_table("events"), db.open_table("unit_detections_v5l_e15"), db.open_table("videos")


def plays(mid):
    return sorted(({"t": e["t_video"], "card": e["card"]} for e in ev.search().where(f"match_id = '{mid}' AND kind = 'play'")
                   .limit(10**4).select(["t_video", "card"]).to_arrow().to_pylist() if e["card"]), key=lambda p: p["t"])


rows, n_games, n_placed, n_plays = [], 0, 0, 0
files = sorted(glob.glob("data/ground_truth/opponent_truth_replay_*.json"))
for k, f in enumerate(files):
    m = json.loads(Path(f).read_text())["matches"][0]
    if m in skip:
        continue
    vid = m.rsplit("_m", 1)[0]
    d = dets.search().where(f"match_id = '{m}'").limit(10**7).to_arrow().to_pylist()
    if not d:
        continue
    v = vids.search().where(f"video_id = '{vid}'").to_arrow().to_pylist()[0]
    lay = Layout.load(v["layout_id"])
    ax0, ay0, ax1, _ = lay.katacr_arena
    dep = deploys(d, names, (ax1 - ax0) / 18, lambda c: ctype.get(c) == "building", lambda c: ctype.get(c) == "spell",
                  (ax0, ay0), spell_team=None)
    b = placed(plays(m), dep, "friendly")
    t = placed(plays(m.replace("_m", "_top_m", 1)), dep, "enemy")
    n_plays += len(b) + len(t); n_placed += sum(p["x"] is not None for p in b + t)
    for r in interactions(b, t):
        rows.append({**r, "match_id": m, "league": league.get(vid, -1)})
    n_games += 1
    if k % 50 == 0:
        print(f"{k + 1}/{len(files)} files, {n_games} games, {len(rows)} interactions", flush=True)
db.create_table("interactions", pa.Table.from_pylist(rows), mode="overwrite")
print(f"interactions: {len(rows)} from {n_games} games; plays placed {n_placed}/{n_plays} ({n_placed / max(n_plays, 1):.0%})")

# playbook: per opponent card and zone, the answers
groups = defaultdict(list)
for r in rows:
    groups[(r["opp_card"], r["opp_zone"])].append(r)
    groups[(r["opp_card"], "any")].append(r)
book = []
for (card, z), rs in groups.items():
    by = defaultdict(list)
    for r in rs:
        by[r["answer"]].append(r)
    for ans, a in by.items():
        if len(a) < 3:
            continue
        pos = [x for x in a if x["dx"] is not None]
        book.append({"opp_card": card, "opp_zone": z, "answer": ans, "n": len(a), "share": len(a) / len(rs),
                     "n_opp": len(rs), "delay_s": statistics.median(x["delay_s"] for x in a),
                     "dx": statistics.median(x["dx"] for x in pos) if pos else None,
                     "dy": statistics.median(x["dy"] for x in pos) if pos else None,
                     "same_lane": sum(x["same_lane"] for x in pos) / len(pos) if pos else None,
                     "ans_x": statistics.median(x["ans_x"] for x in pos) if pos else None,
                     "ans_y": statistics.median(x["ans_y"] for x in pos) if pos else None})
db.create_table("playbook", pa.Table.from_pylist(book), mode="overwrite")
print(f"playbook: {len(book)} rows for {len({b['opp_card'] for b in book})} opponent cards")
for card in ("Giant", "Hog Rider", "Goblin Barrel", "Mini P.E.K.K.A"):
    top = sorted((b for b in book if b["opp_card"] == card and b["opp_zone"] == "any"), key=lambda b: -b["share"])[:4]
    print(card, [(b["answer"], f"{b['share']:.0%}", round(b["delay_s"], 1), b["dy"] and round(b["dy"], 1)) for b in top])
