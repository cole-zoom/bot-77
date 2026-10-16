"""Mining pass over every embedded unit crop (Cole, 2026-10-03: "take a mining pass on every single card").

Scores all `unit_crops` vectors with the crop classifier (bot77/cropclf.py; held-out ≥0.9 confidence was 92.5% right
on 2026-10-03), and for every tracked entity keeps the most confident crops: opponent side preferred (our labels are
almost all the POV player's own units), at most one per match-second and PER_VIDEO per video, nothing within 3 s of a
frame already labelled for that entity. Writes a ✓/✗ crop page; approved crops become training frames via
scripts/mine_import.py.

  uv run python scripts/mine_all.py [per_entity=40] [min_p=0.9]
"""
import collections, heapq, json, sys
from pathlib import Path

import cv2
import lancedb
import numpy as np
import torch
import yaml

from bot77.cropclf import CropClassifier
from bot77.spotcheck import write_spotcheck

PER_ENTITY = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MIN_P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
PER_VIDEO = 6
OUT = Path("data/external/mined_v1")

clf = CropClassifier()
base = set(yaml.safe_load(open("data/external/synth_v1/data.yaml"))["names"].values())
# every class beyond KataCR's, plus every entity still under the bar in the cleaned count (some, like evo Archers or
# evo Zap, carry KataCR names)
NEED = {l.split()[0]: int(l.split()[-1]) for l in open("data/logs/coverage_clean6.log") if "need" in l}
targets = [c for c in clf.classes if c not in base or c in NEED]
tidx = np.array([clf.cidx[c] for c in targets])
print(len(targets), "entities to mine", flush=True)

# frames already labelled per entity (any source), so mining adds new moments only
labelled = collections.defaultdict(list)
for f in Path("data/external/real_v1").glob("manifest_*.json"):
    for x in json.loads(f.read_text()):
        labelled[(x["video_id"], x["entity"])].append(x["t_video"])

db = lancedb.connect("data/lancedb")
tab = db.open_table("unit_crops")
heaps = {c: [] for c in targets}  # (score, seq, crop_id, meta); seq breaks ties
seq = 0
in_heap = {c: set() for c in targets}
cols = ["crop_id", "video_id", "match_id", "t_video", "team", "old_cls", "conf", "x0", "y0", "x1", "y1", "vector"]
n = 0
total = tab.count_rows()
for off in range(0, total, 65536):  # paged queries (no pylance here); the image column is never loaded
    rows = tab.search().select(cols).offset(off).limit(65536).to_arrow().to_pydict()
    if not rows["crop_id"]:
        break
    V = np.asarray(rows["vector"], np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    with torch.no_grad():
        P = torch.softmax(clf.head(torch.tensor(V) * clf.scale), 1).numpy()
    top = P.argmax(1)
    for j in np.where(np.isin(top, tidx) & (P.max(1) >= MIN_P))[0]:
        c = clf.classes[top[j]]
        p = float(P[j, top[j]])
        score = p + (0.05 if rows["team"][j] == "enemy" else 0.0)  # prefer the opponent's side
        cid = rows["crop_id"][j]
        if cid in in_heap[c]:
            continue  # duplicate crop rows (a video re-embedded after an error) would be counted twice
        seq += 1
        item = (score, seq, cid, {k: rows[k][j] for k in cols if k != "vector"} | {"p": p})
        h = heaps[c]
        if len(h) < PER_ENTITY * 20:
            heapq.heappush(h, item); in_heap[c].add(cid)
        elif score > h[0][0]:
            in_heap[c].discard(heapq.heapreplace(h, item)[2]); in_heap[c].add(cid)
    n += len(V)
    print(f"scored {n} crops", flush=True)

picked = {}
for c, h in heaps.items():
    per_sec, per_vid, out = set(), collections.Counter(), []
    for score, _, cid, m in sorted(h, key=lambda it: (it[0], it[1]), reverse=True):
        key = (m["match_id"], int(m["t_video"]))
        if key in per_sec or per_vid[m["video_id"]] >= PER_VIDEO:
            continue
        if any(abs(m["t_video"] - t) <= 3 for t in labelled.get((m["video_id"], c), [])):
            continue
        per_sec.add(key); per_vid[m["video_id"]] += 1; out.append(m)
        if len(out) >= PER_ENTITY:
            break
    picked[c] = out
print({c: len(v) for c, v in picked.items() if v}, flush=True)

# crop images for the review page: one "frame" per crop, the yellow box around the unit in some context
(OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(parents=True, exist_ok=True)
ids = [m["crop_id"] for v in picked.values() for m in v]
imgs = {}
for i in range(0, len(ids), 500):
    q = ", ".join(f"'{x}'" for x in ids[i:i + 500])
    for r in tab.search().where(f"crop_id IN ({q})").select(["crop_id", "image"]).limit(1000).to_arrow().to_pylist():
        imgs[r["crop_id"]] = r["image"]
items = []
for c, v in picked.items():
    for m in v:
        if m["crop_id"] not in imgs:
            continue
        im = cv2.imdecode(np.frombuffer(imgs[m["crop_id"]], np.uint8), 1)
        h, w = im.shape[:2]; pad = max(h, w) // 2
        canvas = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        canvas = cv2.resize(canvas, None, fx=2, fy=2)
        H, W = canvas.shape[:2]
        name = f"mined_{m['crop_id']}"
        cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), canvas)
        (OUT / "labels" / f"{name}.txt").write_text(f"{clf.classes.index(c) if False else json.load(open('config/classes_v2.json')).index(c)} 0.5 0.5 {w / (w + 2 * pad):.5f} {h / (h + 2 * pad):.5f}\n")
        items.append({**m, "name": name, "entity": c, "source": f"{m['team']} · p {m['p']:.2f}", "how": "mined"})
(OUT / "manifest_0.json").write_text(json.dumps(items, indent=1, default=float))
items.sort(key=lambda x: (-NEED.get(x["entity"], 0), x["entity"]))  # shortest entities first
print(write_spotcheck(OUT, Path("data/reviews/mining.html"), len(items), 0, None, "Mining · every card"), "crops on the page", flush=True)
