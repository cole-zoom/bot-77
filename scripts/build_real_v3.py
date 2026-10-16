"""Rebuild the real training frames without unlabelled troops (decisions.md, 2026-10-10): data/external/real_v3.

v2m showed the problem: a real_v2 frame has ~2.6 unit boxes (the verified main box + what v1 caught) while ~10-25
units are on screen, so the detector learned "troops in real footage are background".

Per real_v2 frame:
  1. window = the verified main box(es) + WINDOW_PAD tiles, grown to take in any accepted box it cuts through
  2. the window is pasted onto an EMPTY arena of the same match: the per-pixel median of BG_FRAMES frames across that
     match (units move, the arena doesn't), with the same crop geometry as the frame, so it looks native
  3. inside the window: other units get pseudo-labels from v2m-best (epoch 3) when the crop classifier agrees
     (same card family as top-1), HUD classes (badge, HP bar, clock...) at conf >= HUD_CONF; anything detected
     but not accepted is painted over with the empty arena
  4. towers: kept from the frame (pixels + labels); a tower that's standing in the empty arena but destroyed in the
     frame gets the frame's pixels (the ruin) so no unlabelled tower is left
Matches whose empty arena still shows units (v2m finds a non-tower unit at conf >= 0.4) are skipped.

  uv run python scripts/build_real_v3.py [--bg-only] [--jobs 4]
  V4: --src data/external/real_v4_raw --dst data/external/real_v4 --pseudo <V3 weights> --manifest manifest_all.json
  (backgrounds are shared: --bg-dir data/external/real_v3/bg reuses the cached empty arenas)
"""
import argparse, collections, json, re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SRC, DST = Path("data/external/real_v2"), Path("data/external/real_v3")
BG_DIR = DST / "bg"
BG_FRAMES = 30
BG_KEEP = 15  # least-red frames used for the median: holding a card tints the half you can't place on red
WINDOW_PAD = 1.5  # tiles
TILE = 576 / 18
HUD_CONF, UNIT_CONF, ANY_CONF = 0.4, 0.3, 0.2
PSEUDO = Path("data/models/bot77_detector/v2m/best.pt")
classes = json.loads(Path("config/classes_v2.json").read_text())
TOWERS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"}
HUD = {"bar", "bar-level", "tower-bar", "king-tower-bar", "dagger-duchess-tower-bar", "skeleton-king-bar", "clock",
       "elixir", "emote", "text", "evolution-symbol", "ice-spirit-evolution-symbol"}
family = lambda c: re.sub(r"-(evolution|hero)(-dismounted)?$", "", c)


def match_bg(args):
    """Median empty arena for one match (cached)."""
    match_id, path, layout_id, t0, t1 = args
    out = BG_DIR / f"{match_id}.png"
    if out.exists():
        return match_id, True
    from bot77.detect.units import INPUT_WH, UnitDetector
    from bot77.layout import Layout
    lay = Layout.load(layout_id)
    cap = cv2.VideoCapture(str(path))
    ims = []
    for t in np.linspace(t0 + 5, t1 - 3, BG_FRAMES):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        crop, *_ = UnitDetector.arena(None, f, lay)
        ims.append(cv2.resize(crop, INPUT_WH, interpolation=cv2.INTER_AREA))
    if len(ims) < 12:
        return match_id, False
    red = [float((im[..., 2].astype(np.int16) - (im[..., 0].astype(np.int16) + im[..., 1]) // 2).mean()) for im in ims]
    ims = [ims[i] for i in np.argsort(red)[:BG_KEEP]]
    cv2.imwrite(str(out), np.median(np.stack(ims), axis=0).astype(np.uint8))
    return match_id, True


def yolo_boxes(r):
    return [(r.names[int(c)], float(p), tuple(map(float, b))) for b, c, p in
            zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def read_labels(p, W, H):
    out = []
    for l in p.read_text().splitlines():
        if l.strip():
            c, cx, cy, w, h = l.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            out.append((classes[int(c)], (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def ioa(a, win):  # share of box a inside win
    ix = max(0, min(a[2], win[2]) - max(a[0], win[0])); iy = max(0, min(a[3], win[3]) - max(a[1], win[1]))
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src"); ap.add_argument("--dst"); ap.add_argument("--pseudo"); ap.add_argument("--bg-dir")
    ap.add_argument("--manifest", default="manifest_clean.json")
    ap.add_argument("--bg-only", action="store_true"); ap.add_argument("--test", type=int, default=0, help="composite N frames from matches whose empty arena exists, into real_v3_test"); ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    global SRC, DST, BG_DIR, PSEUDO
    SRC, DST = Path(a.src or SRC), Path(a.dst or DST)
    BG_DIR = Path(a.bg_dir) if a.bg_dir else DST / "bg"
    PSEUDO = Path(a.pseudo or PSEUDO)
    import lancedb
    db = lancedb.connect("data/lancedb")
    man = json.loads((SRC / a.manifest).read_text())
    vids = {v["video_id"]: v for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
    ms = {m["match_id"]: m for m in db.open_table("matches").search().limit(100_000).to_arrow().to_pylist()}
    BG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(mid, vids[ms[mid]["video_id"]]["path"], vids[ms[mid]["video_id"]]["layout_id"], ms[mid]["t_start"], ms[mid]["t_end"])
            for mid in sorted({x["match_id"] for x in man}) if mid in ms]
    ok = {}
    if a.test:
        ok = {p.stem: True for p in BG_DIR.glob("*.png")}
        man = [x for x in man if x["match_id"] in ok][:a.test]
        DST = Path("data/external/real_v3_test"); jobs = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, (mid, good) in enumerate(ex.map(match_bg, jobs)):
            ok[mid] = good
            if k % 50 == 0:
                print(f"bg {k}/{len(jobs)}", flush=True)
    print("backgrounds:", sum(ok.values()), "of", len(jobs), flush=True)
    if a.bg_only:
        return

    from ultralytics import YOLO
    from bot77.cropclf import CropClassifier
    det = YOLO(str(PSEUDO)); clf = CropClassifier()
    FAM = collections.defaultdict(set)
    for i, c in enumerate(clf.classes):
        FAM[family(c)].add(i)
    # empty arenas that still show units are unusable; towers standing in each are remembered
    bg_towers, dirty = {}, set()
    for mid in [m for m, g in ok.items() if g]:
        bg = cv2.imread(str(BG_DIR / f"{mid}.png"))
        bx = yolo_boxes(det.predict(bg, imgsz=896, conf=0.4, device="mps", verbose=False)[0])
        if any(n not in TOWERS | HUD for n, _, _ in bx):
            dirty.add(mid)
        bg_towers[mid] = [b for n, _, b in bx if n in TOWERS]
    print("dirty empty arenas (skipped):", len(dirty), flush=True)

    (DST / "images").mkdir(parents=True, exist_ok=True); (DST / "labels").mkdir(exist_ok=True)
    out, st = [], collections.Counter()
    for k, x in enumerate(man):
        mid = x["match_id"]
        if not ok.get(mid) or mid in dirty:
            st["no_clean_bg"] += 1; continue
        im = cv2.imread(str(SRC / "images" / f"{x['name']}.jpg")); H, W = im.shape[:2]
        bg = cv2.imread(str(BG_DIR / f"{mid}.png"))
        labels = read_labels(SRC / "labels" / f"{x['name']}.txt", W, H)
        n_main = x.get("n_main", 1)
        mains, rest = labels[:n_main], labels[n_main:]
        # pseudo-labels
        dets = yolo_boxes(det.predict(im, imgsz=896, conf=ANY_CONF, device="mps", verbose=False)[0])
        accepted, unconfirmed = [], []
        for n, p, b in dets:
            if n in TOWERS:
                continue
            if n in HUD:
                (accepted if p >= HUD_CONF else unconfirmed).append((n, b)); continue
            crop = im[max(int(b[1]), 0):int(b[3]), max(int(b[0]), 0):int(b[2])]
            fam = FAM.get(family(n))
            if crop.size and fam and p >= UNIT_CONF and int(clf.probs([crop])[0].argmax()) in fam:
                accepted.append((n, b))
            elif not fam and p >= 0.5:
                accepted.append((n, b))  # classes the classifier never learned: confident detections only
            else:
                unconfirmed.append((n, b))
        # existing labels win over pseudo-labels for the same object
        have = [b for _, b in labels]
        accepted = [(n, b) for n, b in accepted if not any(iou(b, h) > 0.5 for h in have)]
        keep = [(n, b) for n, b in rest if n not in TOWERS] + accepted
        # window around the verified boxes, grown over any kept box it cuts
        win = [min(b[0] for _, b in mains) - WINDOW_PAD * TILE, min(b[1] for _, b in mains) - WINDOW_PAD * TILE,
               max(b[2] for _, b in mains) + WINDOW_PAD * TILE, max(b[3] for _, b in mains) + WINDOW_PAD * TILE]
        for _ in range(3):
            grow = [b for _, b in keep if 0.05 < ioa(b, win) < 0.95]
            if not grow:
                break
            win = [min([win[0]] + [b[0] for b in grow]), min([win[1]] + [b[1] for b in grow]),
                   max([win[2]] + [b[2] for b in grow]), max([win[3]] + [b[3] for b in grow])]
        win = [max(0, win[0]), max(0, win[1]), min(W, win[2]), min(H, win[3])]
        x0, y0, x1, y1 = map(int, win)
        if (x1 - x0) * (y1 - y0) > 0.6 * W * H:
            st["window_too_big"] += 1; continue  # the frame is mostly window: little gain, keep it out
        # composite: empty arena + feathered window
        mask = np.zeros((H, W), np.float32); mask[y0:y1, x0:x1] = 1
        mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
        comp = (im * mask + bg * (1 - mask)).astype(np.uint8)
        inside = lambda b: ioa(b, win) >= 0.6
        final = [(n, b) for n, b in mains] + [(n, b) for n, b in keep if inside(b)]
        # paint unconfirmed detections inside the window with the empty arena
        for n, b in unconfirmed:
            if inside(b) and not any(iou(b, fb) > 0.3 for _, fb in final):
                bx0, by0, bx1, by1 = max(int(b[0]), 0), max(int(b[1]), 0), min(int(b[2]), W), min(int(b[3]), H)
                comp[by0:by1, bx0:bx1] = bg[by0:by1, bx0:bx1]; st["painted_over"] += 1
        # towers + their bars from the frame
        frame_towers = [(n, b) for n, b in labels if n in TOWERS or n in ("tower-bar", "king-tower-bar", "dagger-duchess-tower-bar")]
        for n, b in frame_towers:
            bx0, by0, bx1, by1 = max(int(b[0]), 0), max(int(b[1]), 0), min(int(b[2]), W), min(int(b[3]), H)
            comp[by0:by1, bx0:bx1] = im[by0:by1, bx0:bx1]
            if not any(n2 == n and iou(b, fb) > 0.5 for n2, fb in final):
                final.append((n, b))
        for b in bg_towers.get(mid, []):  # standing in the empty arena, destroyed in this frame -> show the ruin
            if not any(n in TOWERS and iou(b, fb) > 0.3 for n, fb in frame_towers):
                bx0, by0, bx1, by1 = max(int(b[0]), 0), max(int(b[1]), 0), min(int(b[2]), W), min(int(b[3]), H)
                comp[by0:by1, bx0:bx1] = im[by0:by1, bx0:bx1]; st["tower_ruin_restored"] += 1
        lines = []
        for n, b in final:
            cx0, cy0, cx1, cy1 = max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)
            if cx1 - cx0 < 3 or cy1 - cy0 < 3:
                continue
            lines.append(f"{classes.index(n)} {(cx0 + cx1) / 2 / W:.6f} {(cy0 + cy1) / 2 / H:.6f} {(cx1 - cx0) / W:.6f} {(cy1 - cy0) / H:.6f}")
        cv2.imwrite(str(DST / "images" / f"{x['name']}.jpg"), comp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (DST / "labels" / f"{x['name']}.txt").write_text("\n".join(lines) + "\n")
        out.append({**x, "window": [x0, y0, x1, y1], "pseudo": len([1 for n, b in accepted if inside(b)]), "n_labels": len(lines)})
        st["frames_kept"] += 1; st["pseudo_labels"] += out[-1]["pseudo"]
        if k % 300 == 0:
            print(f"{k}/{len(man)}", dict(st), flush=True)
    (DST / "manifest.json").write_text(json.dumps(out, indent=1))
    print("done", dict(st), flush=True)


if __name__ == "__main__":
    main()
