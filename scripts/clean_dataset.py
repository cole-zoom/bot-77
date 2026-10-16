"""Build the cleaned real-frame set data/external/real_v2 from real_v1, using spot-check 3's decisions
(decisions.md, 2026-10-02). real_v1 is left untouched.

  - Cole's clicks and approved candidates: kept as they are (verified)
  - auto-labelled main boxes: the frame is kept only if the crop classifier gives every main box p(label) >= 0.3
    (on Cole's 3A verdicts: 48% -> 86% right, wrong 27% -> 0%); classes it can't score are kept
  - grey boxes: kept only where the classifier's top-1 is the grey label (3C: 59% -> 88% right). The others are
    masked (filled with the colour around them) so the detector isn't taught that a unit is background, unless
    the box overlaps a kept box, in which case it is just dropped.
"""
import collections, json, shutil
from pathlib import Path

import cv2
import numpy as np

import re
from bot77.autolabel import KEEP_OLD, ROLLING
from bot77.cropclf import REVIEW_MIN_P, CropClassifier

import sys
SRC, DST = (Path(sys.argv[1]), Path(sys.argv[2])) if len(sys.argv) > 2 else (Path("data/external/real_v1"), Path("data/external/real_v2"))  # V4: real_v4_raw real_v4_clean
(DST / "images").mkdir(parents=True, exist_ok=True); (DST / "labels").mkdir(parents=True, exist_ok=True)
classes = json.loads(Path("config/classes_v2.json").read_text())
clf = CropClassifier()
family = lambda c: re.sub(r"-(evolution|hero)$", "", c)
FAMILY = collections.defaultdict(list)
for i, c in enumerate(clf.classes):
    FAMILY[family(c)].append(i)
CONFUSED = {"goblin-giant-evolution": "phoenix", "vines": "suspicious-bush"}  # card -> companion class prefix
# Forms the detector can't tell apart on the board in a single frame: trained as the base card; the form comes from
# the hand (POV) or cycle counting (opponent). Cole, 2026-10-08: "u can always tell the evo by the card in the hand".
# evo Zap was merged briefly, then kept separate (Cole: truer blue, zaps twice). bats-evolution / skeletons-evolution
# were duplicate names for KataCR's single-unit classes bat-evolution / skeleton-evolution (2026-10-08).
MERGE_FORMS: dict[str, str] = {"bats-evolution": "bat-evolution", "skeletons-evolution": "skeleton-evolution"}
# Group cards: a main box bigger than one unit is a group box (Cole, 2026-10-08: evo Bats / hero Goblins boxed as the
# whole swarm). Limits in arena tiles, a bit above KataCR's 90th-percentile single-unit size.
MAX_UNIT_TILES = {"bats-evolution": 2.6, "bat-evolution": 2.6, "goblins-hero": 2.2, "skeletons-evolution": 1.6,
                  "skeleton-evolution": 1.6, "royal-hogs-evolution": 2.4}
MAX_MASK_TILES = 3.0  # wrong grey boxes larger than this (in arena tiles) are dropped, not masked
VERIFIED = ("click", "candidate-approved", "mined")  # mined = Cole approved the crop on the mining page


def boxes(lines, W, H):
    out = []
    for l in lines:
        c, cx, cy, w, h = l.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        out.append((int(c), int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)))
    return out


def iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1])); iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    u = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def crop(im, b):
    return im[max(b[2], 0):b[4], max(b[1], 0):b[3]]


def fill(im, b):
    x0, y0, x1, y1 = max(b[1], 0), max(b[2], 0), min(b[3], im.shape[1]), min(b[4], im.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    m = 6
    ring = np.concatenate([im[max(y0 - m, 0):y0, x0:x1].reshape(-1, 3), im[y1:y1 + m, x0:x1].reshape(-1, 3),
                           im[y0:y1, max(x0 - m, 0):x0].reshape(-1, 3), im[y0:y1, x1:x1 + m].reshape(-1, 3)])
    im[y0:y1, x0:x1] = np.median(ring, axis=0) if len(ring) else 0


# frames Cole approved on review pages (form review: forms the classifier never saw) count as verified
APPROVED = {fr["id"] for f in Path("data/ground_truth/reviews").glob("formreview_*.json")
            for fr in json.loads(f.read_text())["frames"] if fr.get("verdict") == "right"}
HOLDOUT = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])  # never trained on
seen, out, st = set(), [], collections.Counter()
frames = [x for f in sorted(SRC.glob("manifest_*.json")) for x in json.loads(f.read_text())]
for i, x in enumerate(frames):
    if x["name"] in seen:
        continue
    if x.get("match_id") in HOLDOUT:
        st["frames_held_out_benchmark"] += 1
        continue
    seen.add(x["name"])
    img, lbl = SRC / "images" / f"{x['name']}.jpg", SRC / "labels" / f"{x['name']}.txt"
    if not (img.exists() and lbl.exists()):
        continue
    im = cv2.imread(str(img)); H, W = im.shape[:2]
    lines = lbl.read_text().splitlines(); n_main = x.get("n_main", 1)
    bx = boxes(lines, W, H); mains, greys = bx[:n_main], bx[n_main:]
    crops = [crop(im, b) for b in bx]
    ok = [k for k, c in enumerate(crops) if c.size and c.shape[0] >= 4 and c.shape[1] >= 4]
    P = dict(zip(ok, clf.probs([crops[k] for k in ok]))) if ok else {}
    # co-occurrence contamination (Cole, 2026-10-05): a main box overlapping a box of its usual companion is dropped,
    # whatever its source, so the detector can't learn the companion as the card
    if any(classes[m[0]] in CONFUSED and any(classes[g[0]].startswith(CONFUSED[classes[m[0]]]) and iou(m, g) > 0.2
                                             for g in greys) for m in mains):
        st["frames_dropped_cooccurrence"] += 1
        continue
    if any(classes[m[0]] in MAX_UNIT_TILES and max(m[3] - m[1], m[4] - m[2]) / (W / 18) > MAX_UNIT_TILES[classes[m[0]]]
           for m in mains):
        st["frames_dropped_group_box"] += 1
        continue
    if x["entity"] in ROLLING and x.get("how") == "spell-radius":
        st["frames_dropped_snowball_area"] += 1  # area boxes for the evo Snowball were all wrong (3A, candidates)
        continue
    if x["name"] in APPROVED:
        st["frames_approved_by_cole"] += 1
    elif x.get("how") not in VERIFIED and x.get("source") != "click" and x.get("how") != "spell-radius":
        # spell-radius boxes are effect areas, which a unit classifier can't judge: kept, as spot-checked before.
        # The card family (all forms) is scored, not the exact form: the form comes from the play log, and the
        # classifier confuses rare forms with their base card (hero Knight 180 -> 12 frames with p(label)).
        bad = False
        for k, b in enumerate(mains):
            fam_idx = FAMILY.get(family(classes[b[0]]))
            if fam_idx and k in P and P[k][fam_idx].sum() < REVIEW_MIN_P:
                bad = True
        if bad:
            st["frames_dropped_auto_main"] += 1
            continue
    keep_lines, masked = lines[:n_main], []
    for k, (l, b) in enumerate(zip(lines[n_main:], greys), start=n_main):
        name = classes[b[0]]
        if name in KEEP_OLD or name not in clf.cidx:
            # towers, HP bars, clock, elixir... and any class the classifier never learned: it can't judge them, so
            # they are kept as they are (first run masked every tower, 2026-10-02)
            keep_lines.append(l); st["grey_kept_unjudged"] += 1
        elif k in P and int(P[k].argmax()) == clf.cidx[name]:
            keep_lines.append(l); st["grey_kept"] += 1
        elif any(iou(b, m) > 0.3 for m in mains):
            st["grey_dropped_overlap"] += 1
        elif (b[3] - b[1]) > MAX_MASK_TILES * W / 18 or (b[4] - b[2]) > MAX_MASK_TILES * W / 18:
            st["grey_dropped_big"] += 1  # a huge wrong box is a misdetection, not a unit: masking it leaves a big patch
        else:
            masked.append(b); st["grey_masked"] += 1
    if x.get("how") == "spell-radius":
        # a unit inside a spell area must keep its own correct label, otherwise the detector learns the unit as part
        # of the spell (Cole, 2026-10-05: Vines "looks like" Suspicious Bush; vines wrap the unit they hit, ~0.1 s
        # after the drop, so every Vines frame contains its target)
        kept_set = set(keep_lines[n_main:])
        inside = [g for g, l in zip(greys, lines[n_main:]) if classes[g[0]] not in KEEP_OLD and l not in kept_set
                  and any(m[1] <= (g[1] + g[3]) / 2 <= m[3] and m[2] <= (g[2] + g[4]) / 2 <= m[4] for m in mains)]
        if inside:
            st["frames_dropped_spell_unlabelled_unit"] += 1
            continue
    kept_boxes = boxes(keep_lines, W, H)
    for b in masked:
        if not any(iou(b, kb) > 0.3 for kb in kept_boxes):
            fill(im, b)
    remap = {classes.index(a): classes.index(b) for a, b in MERGE_FORMS.items()}
    keep_lines = [" ".join([str(remap.get(int(l.split()[0]), int(l.split()[0])))] + l.split()[1:]) for l in keep_lines]
    if x["entity"] in MERGE_FORMS:
        x = {**x, "entity": MERGE_FORMS[x["entity"]]}
    cv2.imwrite(str(DST / "images" / f"{x['name']}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (DST / "labels" / f"{x['name']}.txt").write_text("\n".join(keep_lines) + "\n")
    out.append({**x, "n_main": n_main, "grey_kept": len(keep_lines) - n_main, "grey_masked": len(masked)})
    st["frames_kept"] += 1
    if i % 500 == 0:
        print(f"{i}/{len(frames)}", dict(st), flush=True)
(DST / "manifest_clean.json").write_text(json.dumps(out, indent=1))
print("done", dict(st), flush=True)
