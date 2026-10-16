"""Per-card delay between a play (card leaves the hand) and the first detection of its unit, from the development set
only (paired matches minus config/benchmark_holdout.json). Uses raw enemy tracks (more samples than finished plays).
Writes config/play_timing.json, used by bot77.playrules.
  uv run python scripts/play_timing.py <det tag>
"""
import collections, glob, json, statistics, sys
from pathlib import Path

import lancedb

from bot77.layout import Layout
from bot77.spawns import track_units

tag = sys.argv[1]
db = lancedb.connect("data/lancedb")
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
names = [c["name"] for c in cards]; ctype = {c["name"]: c["type"] for c in cards}
d = collections.defaultdict(list)
for f in sorted(glob.glob("data/ground_truth/opponent_truth_auto_*.json")):
    t = json.loads(Path(f).read_text()); m = t["matches"][0]
    if m in hold:
        continue
    vid = m.rsplit("_m", 1)[0]
    video = db.open_table("videos").search().where(f"video_id = '{vid}'").to_arrow().to_pylist()[0]
    lay = Layout.load(video["layout_id"]); ax0, ay0, ax1, ay1 = lay.katacr_arena; tile = (ax1 - ax0) / 18
    dets = db.open_table(f"unit_detections_{tag}").search().where(f"match_id = '{m}'").limit(10**7).to_arrow().to_pylist()
    dets = [{**x, "x0": x["x0"] - ax0, "x1": x["x1"] - ax0, "y0": x["y0"] - ay0, "y1": x["y1"] - ay0} for x in dets]
    tracks = [tr for tr in track_units(dets, names, tile, lambda c: ctype.get(c) == "building") if tr.team == "enemy"]
    for x in t["items"]:
        c = [tr.first_t - x["t_video"] for tr in tracks if tr.card == x["label"] and -1.0 <= tr.first_t - x["t_video"] <= 5.0]
        if c:
            d[x["label"]].append(min(c, key=lambda v: abs(v - 1.0)))
out = {c: round(statistics.median(v), 2) for c, v in d.items() if len(v) >= 4}
Path("config/play_timing.json").write_text(json.dumps(dict(sorted(out.items())), indent=1))
print(len(out), "cards:", out)
