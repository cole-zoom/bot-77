"""Cut synthetic-generator sprites from verified real boxes (decisions.md, 2026-10-10), for classes with few or no
approved sprites. synth_v2 needs every new card/form as a sprite: the generator [Wu et al. 2025, §3.1] only pastes
sprites, and 50+ of our classes had none (all hero forms, most evos).

Source: real_v2 frames whose main box is verified (Cole's clicks, approved candidates, mined, form review). These
frames are already the generator's 576x896 arena crop, so sprites need no rescaling.

Per box: SAM cut (box prompt) with the unit's level badge / HP bar erased, largest component, leftover badge
stripped, enclosed holes filled. Kept if the mask fills a plausible share of the box, the box isn't clipped by the
frame edge, and the crop classifier gives the card family p >= 0.3. Ranked by that p, max 3 per video, up to --per.

  uv run python scripts/auto_sprites.py [--per 10] [--min-have 8]
Writes data/slices/auto_v2/<class>/*.png + manifest.json, and data/reviews/sprite_review_v2.html (one row per class,
everything kept by default; Cole clicks the bad ones).
"""
import argparse, base64, collections, glob, json, re
from pathlib import Path

import cv2
import numpy as np

from bot77.cropclf import REVIEW_MIN_P, CropClassifier
from bot77.slices import Cutter, _largest_component, fill_holes, strip_leftover_badge

ap = argparse.ArgumentParser()
ap.add_argument("--per", type=int, default=10)
ap.add_argument("--min-have", type=int, default=8, help="only classes with fewer approved sprites than this")
a = ap.parse_args()

REAL, OUT, PAGE = Path("data/external/real_v2"), Path("data/slices/auto_v2"), Path("data/reviews/sprite_review_v2.html")
classes = json.loads(Path("config/classes_v2.json").read_text())
MERGED = {"bats-evolution", "skeletons-evolution"}  # unused indices (merged into bat-/skeleton-evolution)
BADGES = {classes.index(c) for c in ("bar", "bar-level")}
have = collections.Counter(m["class"] for m in json.loads(Path("data/slices/approved/manifest.json").read_text()))
targets = [c for c in classes[158:] if have[c] < a.min_have and c not in MERGED]
approved = {fr["id"] for f in glob.glob("data/ground_truth/reviews/formreview_*.json")
            for fr in json.loads(Path(f).read_text())["frames"] if fr.get("verdict") == "right"}
man = json.loads((REAL / "manifest_clean.json").read_text())
verified = lambda x: x.get("how") in ("click", "candidate-approved", "mined") or x.get("source") == "click" or x["name"] in approved

clf = CropClassifier()
family = lambda c: re.sub(r"-(evolution|hero)(-dismounted)?$", "", c)
FAM = collections.defaultdict(list)
for i, c in enumerate(clf.classes):
    FAM[family(c)].append(i)
cutter = Cutter()


def boxes(path, W, H):
    out = []
    for l in path.read_text().splitlines():
        if l.strip():
            c, cx, cy, w, h = l.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            out.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


OUT.mkdir(parents=True, exist_ok=True)
manifest, stats = [], collections.Counter()
for cls in targets:
    ci = classes.index(cls)
    cands = []
    for x in (x for x in man if x["entity"] == cls and verified(x)):
        im = cv2.imread(str(REAL / "images" / f"{x['name']}.jpg")); H, W = im.shape[:2]
        bx = boxes(REAL / "labels" / f"{x['name']}.txt", W, H)
        for b in bx[:x.get("n_main", 1)]:
            if b[0] != ci:
                continue
            _, x0, y0, x1, y1 = b
            if x1 - x0 < 12 or y1 - y0 < 12:
                stats["too_small"] += 1; continue
            if x0 < 2 or y0 < 2 or x1 > W - 2 or y1 > H - 2:
                stats["clipped_by_edge"] += 1; continue
            crop = im[int(y0):int(y1), int(x0):int(x1)]
            fam = FAM.get(family(cls))
            p = float(clf.probs([crop])[0][fam].sum()) if fam else 1.0
            cands.append((p, x, im, b, bx))
    # the classifier never saw some forms (hero Magic Archer, evo Electro Giant: every box rejected); for classes with
    # too few passing boxes, take the best-ranked rest anyway and let Cole's review catch bad ones
    passing = [t for t in cands if t[0] >= REVIEW_MIN_P]
    stats["classifier_disagrees"] += len(cands) - len(passing)
    if len(passing) >= 5:
        cands = passing
    picked, per_video = [], collections.Counter()
    for p, x, im, b, bx in sorted(cands, key=lambda t: -t[0]):
        if len(picked) >= a.per:
            break
        if per_video[x["video_id"]] >= 3:
            continue
        _, x0, y0, x1, y1 = b
        w, h = x1 - x0, y1 - y0
        erase = [(e[1], e[2], e[3], e[4]) for e in bx if e[0] in BADGES
                 and x0 - 0.3 * w <= (e[1] + e[3]) / 2 <= x1 + 0.3 * w and y0 - 0.6 * h <= (e[2] + e[4]) / 2 <= y0 + 0.5 * h]
        rgba = cutter.cut(im, (x0, y0, x1, y1), erase)
        if rgba is None:
            stats["sam_empty"] += 1; continue
        alpha = _largest_component(rgba[:, :, 3] > 0)
        fill = alpha.sum() / max(w * h, 1)
        if not 0.2 <= fill <= 0.97:
            stats["mask_implausible"] += 1; continue
        rgba[:, :, 3] = alpha.astype(np.uint8) * 255
        rgba, _ = strip_leftover_badge(rgba)
        rgba = fill_holes(rgba)
        per_video[x["video_id"]] += 1
        f = OUT / cls / f"{cls}_auto_{x['name']}.png"
        f.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(f), rgba)
        picked.append({"file": str(f), "class": cls, "source_frame": x["name"], "video_id": x["video_id"],
                       "how": x.get("how"), "p_family": round(p, 3), "size": [int(rgba.shape[1]), int(rgba.shape[0])]})
    stats["kept"] += len(picked)
    manifest += picked
    print(f"{cls:32s} candidates {len(cands):3d}  kept {len(picked):2d}", flush=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("done", dict(stats))

# review page: one row per class, everything kept unless Cole clicks it
rows = collections.defaultdict(list)
for m in manifest:
    rows[m["class"]].append({"id": Path(m["file"]).name, "img": "data:image/png;base64," + base64.b64encode(Path(m["file"]).read_bytes()).decode()})
data = [{"cls": c, "items": rows[c]} for c in targets if rows[c]]
PAGE.parent.mkdir(parents=True, exist_ok=True)
PAGE.write_text(r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sprite review v2</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--muted:#5d6673;--line:#e2e5ea;--bad:#c43d3d;--accent:#6b3fd4}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1a1e24;--ink:#e8eaed;--muted:#9aa3ad;--line:#2c323a;--bad:#f07575;--accent:#a98bff}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}
button{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding:16px;display:flex;flex-direction:column;gap:14px}
.cls{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}
.cls h2{font-size:14px;margin:0 0 8px;display:flex;gap:8px;align-items:center}
.strip{display:flex;gap:8px;flex-wrap:wrap}
.sp{width:96px;height:110px;display:flex;align-items:center;justify-content:center;border:2px solid var(--line);border-radius:8px;cursor:pointer;
background:repeating-conic-gradient(#8883 0% 25%,transparent 0% 50%) 50%/12px 12px}
.sp img{max-width:92px;max-height:106px}
.sp.bad{border-color:var(--bad);opacity:.35}
.cls.allbad{opacity:.5}
</style></head><body>
<header><h1>Sprite review v2</h1><span class="muted" id="progress"></span>
<span class="muted">Each row is one class. Everything is kept unless you click it. Click a sprite that is NOT a clean cut-out of that unit (wrong unit, two units, big chunk of background or missing half). "Whole row bad" if the class is wrong throughout.</span>
<button class="primary" id="export">Export</button></header><main id="m"></main>
<script>
const DATA = __DATA__;
const KEY = "sprite_review_v2";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} render(); };
function render() {
  const m = document.getElementById("m"); m.innerHTML = "";
  let bad = 0, n = 0;
  for (const r of DATA) {
    const box = document.createElement("div"); box.className = "cls" + (st["row:" + r.cls] ? " allbad" : "");
    const h = document.createElement("h2"); h.textContent = r.cls + " (" + r.items.length + ")";
    const b = document.createElement("button"); b.textContent = st["row:" + r.cls] ? "Undo whole row" : "Whole row bad";
    b.onclick = () => { st["row:" + r.cls] = !st["row:" + r.cls]; save(); }; h.append(b); box.append(h);
    const s = document.createElement("div"); s.className = "strip";
    for (const it of r.items) {
      n++; const isBad = st[it.id] || st["row:" + r.cls]; if (isBad) bad++;
      const d = document.createElement("div"); d.className = "sp" + (isBad ? " bad" : ""); d.title = it.id;
      const im = document.createElement("img"); im.src = it.img; d.append(im);
      d.onclick = () => { st[it.id] = !st[it.id]; save(); }; s.append(d);
    }
    box.append(s); m.append(box);
  }
  document.getElementById("progress").textContent = `${DATA.length} classes, ${n} sprites, ${bad} marked bad`;
}
document.getElementById("export").onclick = () => {
  const out = { page: "sprite_review_v2", rows: DATA.map(r => ({ cls: r.cls, row_bad: !!st["row:" + r.cls],
    bad: r.items.filter(i => st[i.id] || st["row:" + r.cls]).map(i => i.id), all: r.items.map(i => i.id) })) };
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }));
  a.download = "sprite_review_v2.json"; a.click();
};
render();
</script></body></html>""".replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
print("page", PAGE, len(data), "classes")
