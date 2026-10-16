"""Reference crops of each card as it looks in replays, for the replay label check (decisions.md, 2026-10-14).

Round 2 of the replay label review was 35% right: the badge nearest the clock often belonged to another unit, and the
DINOv2 check that should catch that had no approved slices for most cards (50 of 60 frames went unchecked). Here the
references come from the plays V4 already got right: the truth names the card, the clock (replay window) fixes the
spot, and a confident V4 detection of exactly that entity at the spot is the unit. Benchmark, V5 check and replay test
games are never used. Writes data/external/replay_refs/refs.npz (emb, entity, match_id) and the crops, for a look.

  uv run python scripts/replay_refs.py unit_detections_v5l_e15 [--per 40]
"""
import argparse, collections, json, random
from pathlib import Path

import cv2
import lancedb
import numpy as np

from bot77.autolabel import FORM_CLASS, MULTI, UNITS, slug
from bot77.calibrate import read_frame
from bot77.clocks import clock_starts, side_team
from bot77.layout import Layout
from bot77.mine import Embedder

LAG = (-0.3, 0.4)  # replay clock start vs the truth's play time (see autolabel_clock_spots.py)
SPOT_TILES = 2.5  # the detection's centre is this close to the clock
MIN_CONF = 0.7

ap = argparse.ArgumentParser()
ap.add_argument("table"); ap.add_argument("--per", type=int, default=40, help="max references per entity")
ap.add_argument("--out", default="data/external/replay_refs")
a = ap.parse_args()

hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
hold |= set(json.loads(Path("config/v5_check_games.json").read_text())["matches"])
hold |= set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
db = lancedb.connect("data/lancedb")
dets_t = db.open_table(a.table)
videos = {r["video_id"]: r for r in db.open_table("videos").search().limit(10**6).to_arrow().to_pylist()}
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
ctype = {c["name"]: c["type"] for c in cards}
out = Path(a.out); (out / "crops").mkdir(parents=True, exist_ok=True)

files = sorted(Path("data/ground_truth").glob("opponent_truth_replay_*.json"))
random.Random(0).shuffle(files)  # spread each entity's references over many matches
per, stats, crops, meta = collections.Counter(), collections.Counter(), [], []
for f in files:
    tr = json.loads(f.read_text())
    m = tr["matches"][0]
    v = videos.get(tr["video_id"])
    if m in hold or not v:
        continue
    lay = Layout.load(v["layout_id"])
    ax0, ay0, ax1, _ = lay.katacr_arena
    tile = (ax1 - ax0) / 18
    dets = dets_t.search().where(f"match_id = '{m}'").limit(10**7).to_arrow().to_pylist()
    clocks = [c for c in clock_starts(dets, tile, (ax0, ay0), 0.3) if c.n >= 2 and c.y >= 1.5 and side_team(c, 1.0) == "enemy"]
    for i in tr["items"]:
        card = i["label"]
        if card in ("Mirror", "__ability") or ctype.get(card) == "spell" or card in MULTI or UNITS.get(card, 1) > 1:
            continue
        ent = FORM_CLASS.get(slug(card, i.get("form")), slug(card, i.get("form")))
        if per[ent] >= a.per:
            continue
        cl = [c for c in clocks if LAG[0] <= c.t - i["t_video"] <= LAG[1]]
        if len(cl) != 1:
            continue
        c = cl[0]
        hits = [d for d in dets if d["cls"] == ent and d["conf"] >= MIN_CONF and c.t <= d["t_video"] <= c.t + 3
                and abs(((d["x0"] + d["x1"]) / 2 - ax0) / tile - c.x) + abs(((d["y0"] + d["y1"]) / 2 - ay0) / tile - c.y) <= SPOT_TILES]
        if not hits:
            stats["not_named"] += 1
            continue
        d = max(hits, key=lambda d: d["conf"])
        fr = read_frame(Path(v["path"]), d["t_video"])
        s = lay.scale(fr.shape[1])
        crop = fr[int(d["y0"] * s):int(d["y1"] * s), int(d["x0"] * s):int(d["x1"] * s)]
        if crop.size == 0:
            continue
        cv2.imwrite(str(out / "crops" / f"{ent}_{m}_{d['t_video']:07.2f}.jpg"), crop)
        crops.append(crop); meta.append((ent, m)); per[ent] += 1; stats["refs"] += 1
emb = Embedder()
E = np.concatenate([emb(crops[k:k + 64]) for k in range(0, len(crops), 64)])
np.savez(out / "refs.npz", emb=E, entity=np.array([e for e, _ in meta]), match_id=np.array([m for _, m in meta]))
print(dict(stats), "entities", len(per), "refs per entity:", sorted(per.items(), key=lambda kv: -kv[1]))
