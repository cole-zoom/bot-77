"""Classifier on DINOv2 crop embeddings, trained on our verified labels plus KataCR's real crops (Cole,
2026-10-02: "isn't using our own labels the point?").

Until now, our labels were only *references* for raw cosine similarity (approved slices as anchors, a kNN vote
over KataCR's 5,051 crops for grey boxes). Raw similarity weighs the background as much as the unit, which sank
the click-anchored candidates, and the kNN index had none of the new cards. A trained head learns which parts of
the embedding separate classes (e.g. evo vs base) and knows every class we have verified examples of.

  uv run python scripts/crop_classifier.py            # embed, split by video, train, compare with kNN, save

Our crops are split by *video* so held-out accuracy isn't inflated by near-duplicate frames from the same match.
Only verified labels are used: Cole's clicks and approved candidates (auto-labels join after spot-check 3A).
"""
import collections, json, random, sys
from pathlib import Path

import cv2
import lancedb
import numpy as np
import torch

from bot77.mine import Embedder

ROOT = Path("data/external/real_v1")
OUT = Path("data/models/crop_clf.pt")
rng = random.Random(0)

# --- data: KataCR crops (already embedded) + our verified crops -------------------------------------------------
db = lancedb.connect("data/lancedb")
kat = db.open_table("real_crop_index").search().limit(10_000).to_arrow().to_pylist()
X, y, grp = [np.array(r["vector"], np.float32) for r in kat], [r["cls"] for r in kat], ["katacr"] * len(kat)

classes_all = json.loads(Path("config/classes_v2.json").read_text())
ours = [x for f in ("manifest_clicks.json", "manifest_reviewed.json") if (ROOT / f).exists()
        for x in json.loads((ROOT / f).read_text())]
import sys
if "--exclude-candidates" in sys.argv:  # for testing against Cole's candidate verdicts without leakage
    cand = json.loads(Path("data/external/real_v1_candidates/manifest_0.json").read_text())
    near = {(c["video_id"], round(c["t_video"])) for c in cand}
    ours = [x for x in ours if x.get("how") != "candidate-approved" and (x["video_id"], round(x["t_video"])) not in near]
    OUT = Path("data/models/crop_clf_nocand.pt")
crops, meta = [], []
for x in ours:
    img, lbl = ROOT / "images" / f"{x['name']}.jpg", ROOT / "labels" / f"{x['name']}.txt"
    if not (img.exists() and lbl.exists()):
        continue
    im = cv2.imread(str(img)); H, W = im.shape[:2]
    for line in lbl.read_text().splitlines()[:x.get("n_main", 1)]:
        c, cx, cy, w, h = line.split(); cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        crop = im[max(int((cy - h / 2) * H), 0):int((cy + h / 2) * H), max(int((cx - w / 2) * W), 0):int((cx + w / 2) * W)]
        if crop.shape[0] >= 8 and crop.shape[1] >= 8:
            crops.append(crop); meta.append((classes_all[int(c)], x["video_id"]))
# spot-check 3 ✓ crops (Cole, 2026-10-02): auto-label main boxes and grey boxes he confirmed
for kind, root, sc in (("auto", ROOT, Path("data/ground_truth/reviews/spotcheck3_auto_2026-10-02.json")),
                       ("grey", Path("data/external/spotcheck3_grey"), Path("data/ground_truth/reviews/spotcheck3_grey_2026-10-02.json")),
                       ("form", ROOT, Path("data/ground_truth/reviews/formreview_2026-10-02.json"))):
    if "--exclude-candidates" in sys.argv or not sc.exists():
        continue
    for fr in json.loads(sc.read_text())["frames"]:
        if fr.get("verdict") != "right":
            continue
        img, lbl = root / "images" / f"{fr['id']}.jpg", root / "labels" / f"{fr['id']}.txt"
        if not (img.exists() and lbl.exists()):
            continue
        im = cv2.imread(str(img)); H, W = im.shape[:2]
        c, cx, cy, w, h = lbl.read_text().splitlines()[0].split(); cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        crop = im[max(int((cy - h / 2) * H), 0):int((cy + h / 2) * H), max(int((cx - w / 2) * W), 0):int((cx + w / 2) * W)]
        if crop.shape[0] >= 8 and crop.shape[1] >= 8:
            crops.append(crop); meta.append((classes_all[int(c)], fr.get("source", kind) + "_" + fr["id"].split("_m")[0]))
emb = Embedder()
aug = []  # mirrored copies of our crops: training-only (sprites are mirrored by lane in-game)
for i in range(0, len(crops), 64):
    for v, (c, vid) in zip(emb(crops[i:i + 64]), meta[i:i + 64]):
        X.append(np.asarray(v, np.float32)); y.append(c); grp.append(vid)
    for v, (c, vid) in zip(emb([cv2.flip(cr, 1) for cr in crops[i:i + 64]]), meta[i:i + 64]):
        aug.append((np.asarray(v, np.float32), c, vid))
print(f"{len(kat)} KataCR crops + {len(crops)} of ours from {len({m[1] for m in meta})} videos")

# --- split: our crops by video, KataCR crops at random ----------------------------------------------------------
vids = sorted({g for g in grp if g != "katacr"}); rng.shuffle(vids)
test_vids = set(vids[: max(1, len(vids) // 5)])
test = [i for i, g in enumerate(grp) if (g in test_vids) or (g == "katacr" and rng.random() < 0.2)]
tset = set(test); train = [i for i in range(len(y)) if i not in tset]
for v, c, vid in aug:  # mirrored copies join training only, and never from held-out videos
    if vid not in test_vids:
        X.append(v); y.append(c); grp.append(vid + "#flip"); train.append(len(y) - 1)
classes = sorted(set(y[i] for i in train)); cidx = {c: k for k, c in enumerate(classes)}
test = [i for i in test if y[i] in cidx]  # a class seen only in held-out videos can't be scored
Xn = np.stack(X); Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
yt = np.array([cidx.get(c, -1) for c in y])

# --- baseline: the kNN vote we use today (k=7, majority), over the training crops -------------------------------
def knn(i, k=7):
    s = Xn[train] @ Xn[i]
    top = np.argsort(-s)[:k]
    return collections.Counter(yt[train][top]).most_common(1)[0][0]

# --- trained head: linear softmax, class-balanced -----------------------------------------------------------------
Xt, Yt = torch.tensor(Xn[train]), torch.tensor(yt[train])
counts = torch.bincount(Yt, minlength=len(classes)).float().clamp(min=1)
head = torch.nn.Sequential(torch.nn.Linear(Xn.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.3),
                           torch.nn.Linear(512, len(classes)))
opt = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-4)
lossf = torch.nn.CrossEntropyLoss(weight=(1 / counts) / (1 / counts).mean())
for ep in range(400):
    head.train(); opt.zero_grad(); loss = lossf(head(Xt * 20), Yt); loss.backward(); opt.step()  # *20: cosine scale
head.eval()
with torch.no_grad():
    prob = torch.softmax(head(torch.tensor(Xn[test]) * 20), 1)
    conf_p, pred = prob.max(1); conf_p, pred = conf_p.numpy(), pred.numpy()

# --- report -----------------------------------------------------------------------------------------------------
gold = yt[test]
k_pred = np.array([knn(i) for i in test])
ours_mask = np.array([grp[i] != "katacr" for i in test])
for name, m in (("all", np.ones_like(ours_mask)), ("ours (held-out videos)", ours_mask), ("KataCR", ~ours_mask)):
    if m.sum():
        print(f"{name:24} n={m.sum():5}  kNN {np.mean(k_pred[m] == gold[m]):.1%}   head {np.mean(pred[m] == gold[m]):.1%}")
print("confidence threshold -> accuracy / share of our held-out crops kept:")
for th in (0.5, 0.7, 0.8, 0.9, 0.95):
    k = ours_mask & (conf_p >= th)
    if k.sum():
        print(f"  >= {th:.2f}: {np.mean(pred[k] == gold[k]):.1%} right, {k.sum() / ours_mask.sum():.0%} kept")
conf = collections.Counter((classes[g], classes[p]) for g, p in zip(gold[ours_mask], pred[ours_mask]) if g != p)
print("top confusions on our held-out crops (true -> predicted):")
for (g, p), n in conf.most_common(12):
    print(f"  {g:28} -> {p:28} {n}")
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"state": head.state_dict(), "classes": classes, "scale": 20.0, "test_videos": sorted(test_vids)}, OUT)
print("saved", OUT)
