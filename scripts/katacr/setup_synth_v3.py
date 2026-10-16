"""Sprite folder for synth_v3 (decisions.md, 2026-10-10): the game's own art plus the real-footage cut-outs.

  uv run python scripts/katacr/setup_synth_v3.py <KataCR repo> <dataset dir> [sprite_review_v2.json] [--accept-3d a,b,c]

Per class, <dataset dir>/images/segment/<class>/ gets:
  - game sprites (extract_game_sprites.py, ~/tools/bluestacks/sprites): every animation, direction and Star Level
    skin, red (team 1) and blue (team 0);
  - 3D renders (scripts/blender/render_3d_units.py, ~/tools/bluestacks/sprites3d) for the classes passed in --accept-3d
    (the ones that passed the visual check);
  - the real cut-outs as before (KataCR segments, approved, auto_v2 minus Cole's rejects), so the generator still sees
    video-looking units (~10% of a class's files where game art exists, all of it where none does: spells, some heroes).

Scale: game art is drawn at its own resolution (evo sprites ~2x), so each class is resized to the median height its
unit has in real arena crops: the real cut-outs in the folder (KataCR segments + ours, all cut at arena-crop scale),
else the median ratio over all classes (~0.91).
Video look: 70% of game sprites are degraded once (down/up-scale blur, slight JPEG) to match 1080p-video crops.
The generator samples files uniformly within a class and balances classes itself, so file counts only add variety.
Also patches label_list / constant.py like setup_synth_v2.py (idempotent).
"""
import argparse, collections, json, random, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("ds"); ap.add_argument("review", nargs="?")
ap.add_argument("--accept-3d", default="")
a = ap.parse_args()
BOT = Path(__file__).resolve().parents[2]
HOME = Path.home()
GS, R3 = HOME / "tools/bluestacks/sprites", HOME / "tools/bluestacks/sprites3d"
classes = json.loads((BOT / "config/classes_v2.json").read_text())
accept3d = set(x for x in a.accept_3d.split(",") if x)

# 1-2. label_list / constant.py (same patches as v2)
subprocess.run([sys.executable, str(BOT / "scripts/katacr/setup_synth_v2.py"), a.repo, a.ds] + ([a.review] if a.review else []), check=True)
seg = Path(a.ds) / "images/segment"  # now holds KataCR segments + approved + auto (setup_synth_v2)


def alpha_h(p):
    i = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if i is None or i.ndim < 3 or i.shape[2] < 4:
        return None
    ys = np.nonzero(i[:, :, 3] > 40)[0]
    return int(ys.max() - ys.min() + 1) if len(ys) else None


# real heights per class
real_h = {}
for c in classes:
    d = seg / c
    hs = [h for f in sorted(d.glob("*.png"))[:80] if (h := alpha_h(f))] if d.exists() else []
    if len(hs) >= 4:
        real_h[c] = float(np.median(hs))
gs_man = json.loads((GS / "manifest.json").read_text())
# 3D renders are read by file name (<class>_<team>_r3d-...png): several Blender processes render in parallel
r3_man = [{"file": str(f), "class": f.parent.name, "team": int(f.name[len(f.parent.name) + 1])}
          for d in sorted(R3.iterdir()) if d.is_dir() for f in sorted(d.glob("*.png"))] if R3.exists() else []
by_cls = collections.defaultdict(list)
for r in gs_man:
    by_cls[r["class"]].append(("gs", r))
for r in r3_man:
    if r["class"] in accept3d:
        by_cls[r["class"]].append(("r3", r))

ratios = {}
for c, items in by_cls.items():
    hs = [alpha_h(r["file"]) for k, r in items[:200] if k == "gs" and r.get("skin", "base") == "base"] or \
         [alpha_h(r["file"]) for k, r in items[:200]]
    hs = [h for h in hs if h]
    if c in real_h and real_h[c] and hs:
        ratios[c] = real_h[c] / float(np.median(hs))
fallback = float(np.median(list(ratios.values()))) if ratios else 0.91
print(f"scale from real data for {len(ratios)} classes; fallback {fallback:.2f}")

random.seed(0)
stats = collections.Counter()
for c, items in sorted(by_cls.items()):
    d = seg / c
    d.mkdir(parents=True, exist_ok=True)
    s = ratios.get(c, fallback)
    for k, r in items:
        im = cv2.imread(r["file"], cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if k == "r3":  # 3D renders: crop to alpha (fixed canvas)
            ys, xs = np.nonzero(im[:, :, 3] > 8)
            if len(xs) == 0 or xs.min() == 0 or ys.min() == 0 or xs.max() == im.shape[1] - 1 or ys.max() == im.shape[0] - 1:
                stats["r3_empty_or_clipped"] += 1; continue
            im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            # 3D renders have their own pixel scale: fit to the class's real height directly
            sc = (real_h.get(c) or 0) / im.shape[0] if real_h.get(c) else s
        else:
            sc = s
        h, w = im.shape[:2]
        nh, nw = max(4, round(h * sc)), max(4, round(w * sc))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        if random.random() < 0.7:  # video look: blur by down/up-scaling, then light JPEG on the colour
            f = random.uniform(0.55, 0.85)
            small = cv2.resize(im, (max(2, round(nw * f)), max(2, round(nh * f))), interpolation=cv2.INTER_AREA)
            im = cv2.resize(small, (nw, nh), interpolation=cv2.INTER_LINEAR)
            ok, enc = cv2.imencode(".jpg", im[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 85)])
            im[:, :, :3] = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        tag = Path(r["file"]).stem.split("_", 2)[-1]
        cv2.imwrite(str(d / f"{c}_{r['team']}_{k}{tag}.png"), im)
        stats[k] += 1
    stats["classes"] += 1
print(dict(stats))
counts = {c: len(list((seg / c).glob("*.png"))) for c in classes if (seg / c).exists()}
print("files per class (min 15):", sorted(counts.items(), key=lambda t: t[1])[:15])
(Path(a.ds) / "synth_v3_scales.json").write_text(json.dumps({"ratios": ratios, "fallback": fallback, "real_h": real_h}, indent=1))
