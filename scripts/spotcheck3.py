"""Spot-check 3 (Cole, 2026-10-01: "I want to make this good"): measure every label source before the paid run.
  A  auto-labels since spot-check 2 (yellow = the play's unit)        -> data/reviews/spotcheck3_auto.html
  B  Cole's clicks turned into boxes by SAM                           -> data/reviews/spotcheck3_clicks.html
  C  grey boxes (old detector + kNN relabel), ~96% of all real boxes -> data/reviews/spotcheck3_grey.html
     each item is one grey box, cropped with context: judge its label and fit."""
import json, random
from pathlib import Path
import cv2
from bot77.spotcheck import write_spotcheck

ROOT = Path("data/external/real_v1")
auto = [f for f in sorted(ROOT.glob("manifest_*.json")) if f.stem not in ("manifest_clicks", "manifest_reviewed")
        and f.stem not in ("manifest_0", "manifest_1")]  # 0/1 = batches up to 3, measured by spot-check 2
print("A", write_spotcheck(ROOT, Path("data/reviews/spotcheck3_auto.html"), 100, 3, auto, "Spot check 3A · auto-labels"))
print("B", write_spotcheck(ROOT, Path("data/reviews/spotcheck3_clicks.html"), 50, 3, [ROOT / "manifest_clicks.json"],
                           "Spot check 3B · click labels"))
# C: one grey box per item, stratified over classes, on a crop with context
G = Path("data/external/spotcheck3_grey")
(G / "images").mkdir(parents=True, exist_ok=True); (G / "labels").mkdir(parents=True, exist_ok=True)
classes = json.loads(Path("config/classes_v2.json").read_text())
rng = random.Random(3)
frames = [x for f in sorted(ROOT.glob("manifest_*.json")) for x in json.loads(f.read_text())]
rng.shuffle(frames)
by_cls, items = {}, []
for x in frames[:1500]:
    lines = (ROOT / "labels" / f"{x['name']}.txt").read_text().splitlines()[x.get("n_main", 1):]
    for k, l in enumerate(lines):
        by_cls.setdefault(l.split()[0], []).append((x, k, l))
pool = [rng.choice(v) for v in by_cls.values()]
rest = [t for v in by_cls.values() for t in v]
rng.shuffle(rest)
pool += rest[:max(0, 150 - len(pool))]
for x, k, l in pool[:150]:
    im = cv2.imread(str(ROOT / "images" / f"{x['name']}.jpg")); H, W = im.shape[:2]
    c, cx, cy, w, h = l.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
    m = max(w, h) * 1.5 + 40
    x0, y0, x1, y1 = int(max(cx - m, 0)), int(max(cy - m, 0)), int(min(cx + m, W)), int(min(cy + m, H))
    crop = im[y0:y1, x0:x1]; ch, cw = crop.shape[:2]
    name = f"{x['name']}_g{k}"
    cv2.imwrite(str(G / "images" / f"{name}.jpg"), cv2.resize(crop, None, fx=2, fy=2))
    (G / "labels" / f"{name}.txt").write_text(f"{c} {(cx - x0) / cw:.5f} {(cy - y0) / ch:.5f} {w / cw:.5f} {h / ch:.5f}\n")
    items.append({**x, "name": name, "entity": classes[int(c)], "how": "grey box"})
(G / "manifest_0.json").write_text(json.dumps(items, indent=1))
print("C", write_spotcheck(G, Path("data/reviews/spotcheck3_grey.html"), len(items), 3, None, "Spot check 3C · grey boxes"))
