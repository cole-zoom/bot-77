"""Mine detector training candidates at plays the detector never saw (failure mining, 2026-10-16).

For each missed play in `play_outcomes` with no detection of the true card (train-split replays only: held-back,
test, benchmark and dev games stay out of detector training), re-run the detector at a low threshold on a few frames
just after the play and keep the best box that is either called the true card's class, or looks like it: DINOv2
cosine similarity to confident crops of that class in `unit_crops` (LanceDB vector search, bot77.mine). Boxes that are
already a confident detection of another card are skipped. Candidates go to the LanceDB table `mined_candidates` for
Cole's review; approved ones become detector labels.

  uv run python scripts/mine_failures.py [--cards "Miner,Goblin Drill"] [--max-per-card 40]
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import lancedb
import numpy as np
import pyarrow as pa

from bot77.calibrate import read_frame
from bot77.detect.names import NON_UNITS, class_to_card
from bot77.detect.yolo import Bot77Detector
from bot77.layout import Layout
from bot77.mine import Embedder

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="data/models/bot77_detector/v6l/best.pt")
ap.add_argument("--cards", default="", help="comma list (default: every card with a never-seen miss)")
ap.add_argument("--max-per-card", type=int, default=40)
ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--sim", type=float, default=0.6, help="min similarity for a box of another class")
ap.add_argument("--margin", type=float, default=0.05,
                help="a box of another class must look this much more like the card than like its own class")
ap.add_argument("--offsets", default="0.4,0.8,1.2,1.6,2.0,2.4", help="seconds after the play to look")
ap.add_argument("--table", default="mined_candidates")
ap.add_argument("--skip-mined", action="store_true", help="skip plays already mined into --table (next round)")
a = ap.parse_args()

db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
names = [c["name"] for c in cards]
ctype = {c["name"]: c["type"] for c in cards}
classes = json.loads(Path("config/classes_v2.json").read_text())
card_classes = defaultdict(set)
for c in classes:
    k = class_to_card(c, names)[0]
    if k:
        card_classes[k].add(c)

rows = db.open_table("play_outcomes").search().where("kind = 'MISSED' AND seen_n = 0 AND split = 'train'") \
    .limit(10**6).to_arrow().to_pylist()
rows = [r for r in rows if ctype.get(r["true_card"]) not in ("spell", None) and card_classes.get(r["true_card"])]
if a.cards:
    want = set(a.cards.split(","))
    rows = [r for r in rows if r["true_card"] in want]
if a.skip_mined and a.table in db.table_names(limit=10_000):
    done = {(c["match_id"], round(c["t_play"], 1), c["card"]) for c in db.open_table(a.table).search()
            .limit(10**6).select(["match_id", "t_play", "card"]).to_arrow().to_pylist()}
    rows = [r for r in rows if (r["match_id"], round(r["t_video"], 1), r["true_card"]) not in done]
per = defaultdict(list)
for r in sorted(rows, key=lambda r: (r["match_id"], r["t_video"])):
    if len(per[r["true_card"]]) < a.max_per_card:
        per[r["true_card"]].append(r)
rows = [r for v in per.values() for r in v]
print(len(rows), "plays to mine:", Counter(r["true_card"] for r in rows).most_common(), flush=True)

# reference look of each card: confident enemy crops of its classes already in unit_crops
crops = db.open_table("unit_crops")
refs = {}
for card in per:
    cl = " OR ".join(f"old_cls = '{c}'" for c in card_classes[card])
    v = crops.search().where(f"({cl}) AND conf >= 0.8").limit(200).select(["vector"]).to_arrow().column("vector").to_pylist()
    refs[card] = np.array(v, np.float32) if v else None
print("reference crops:", {c: (0 if v is None else len(v)) for c, v in refs.items()}, flush=True)

cls_refs = {}


def class_ref(c):
    """Reference vectors of one detector class (confident crops), cached."""
    if c not in cls_refs:
        v = crops.search().where(f"old_cls = '{c}' AND conf >= 0.8").limit(100).select(["vector"]).to_arrow() \
            .column("vector").to_pylist()
        cls_refs[c] = np.array(v, np.float32) if v else None
    return cls_refs[c]


det = Bot77Detector(Path(a.weights), conf=a.conf)
emb = Embedder()
videos = {v["video_id"]: v for v in db.open_table("videos").search().limit(10**5).to_arrow().to_pylist()}
known = db.open_table("unit_detections_v5l_e15")
offsets = [float(x) for x in a.offsets.split(",")]


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


out = []
for n, r in enumerate(rows):
    v = videos[r["video_id"]]
    path = Path(v["path"])
    if not path.exists():
        print("missing video (LaCie unplugged?)", path); continue
    layout = Layout.load(v["layout_id"])
    card, best = r["true_card"], None
    kn = known.search().where(f"match_id = '{r['match_id']}' AND t_video >= {r['t_video'] - 0.5} AND "
                              f"t_video <= {r['t_video'] + 3.0} AND conf >= 0.5").limit(10**4).to_arrow().to_pylist()
    for off in offsets:
        t = r["t_video"] + off
        f = read_frame(path, t)
        if f is None:
            continue
        cand = []
        for d in det.detect(f, layout):
            if d.name in NON_UNITS or d.team == "friendly":
                continue
            mine = d.name in card_classes[card]
            if not mine and any(abs(k["t_video"] - t) <= 0.2 and iou(d.box, (k["x0"], k["y0"], k["x1"], k["y1"])) > 0.5
                                and class_to_card(k["cls"], names)[0] not in (card, None) for k in kn):
                continue  # a confident detection of another card already
            x0, y0, x1, y1 = (int(round(z)) for z in d.box)
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            cand.append((d, f[max(y0, 0):y1, max(x0, 0):x1]))
        if not cand:
            continue
        vecs = emb([c for _, c in cand])
        for (d, crop), vec in zip(cand, vecs):
            sim = float(np.sort(refs[card] @ vec)[-5:].mean()) if refs[card] is not None else 0.0  # top-5 mean
            mine = d.name in card_classes[card]
            own = class_ref(d.name)
            sim_own = float(np.sort(own @ vec)[-5:].mean()) if (own is not None and not mine) else 0.0
            score = (1.0 if mine else 0.0) + sim - sim_own
            # a class with no reference crops (new v2 classes) can't be compared: then demand a closer look-alike
            ok = mine or (sim >= a.sim and sim - sim_own >= a.margin and (own is not None or sim >= a.sim + 0.08))
            if ok and (best is None or score > best["score"]):
                x0, y0, x1, y1 = d.box
                pw, ph = (x1 - x0) * 0.6, (y1 - y0) * 0.6
                ctx = f[max(int(y0 - ph), 0):int(y1 + ph), max(int(x0 - pw), 0):int(x1 + pw)].copy()
                cv2.rectangle(ctx, (int(x0 - max(x0 - pw, 0)), int(y0 - max(y0 - ph, 0))),
                              (int(x1 - max(x0 - pw, 0)), int(y1 - max(y0 - ph, 0))), (0, 255, 255), 2)
                best = {"cand_id": f"{r['match_id']}_{r['t_video']:.1f}_{card}", "match_id": r["match_id"],
                        "video_id": r["video_id"], "t_play": float(r["t_video"]), "t_video": float(t), "card": card,
                        "det_cls": d.name, "det_conf": float(d.conf), "similarity": sim, "sim_own_class": sim_own, "score": score,
                        "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
                        "image": cv2.imencode(".jpg", ctx, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(),
                        "vector": vec.astype(np.float32).tolist(), "review": None}
    if best:
        out.append(best)
    if n % 20 == 0:
        print(f"{n + 1}/{len(rows)} plays, {len(out)} candidates", flush=True)

if out:
    t = pa.Table.from_pylist(out)
    if a.table in db.table_names(limit=10_000):
        tb = db.open_table(a.table)
        tb.delete(" OR ".join(f"cand_id = '{o['cand_id']}'" for o in out))
        tb.add(t)
    else:
        db.create_table(a.table, t)
print(len(out), "candidates of", len(rows), "plays:", Counter(o["card"] for o in out).most_common())
