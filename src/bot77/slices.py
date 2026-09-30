"""Cut sprite slices for cards the 2024 detector doesn't know, from our own footage.

Following KataCR's generative-dataset method [Wu et al. 2025, §3]: a box around a unit is
given to SAM, and the masked cut-out becomes an RGBA "slice" that the generator later pastes
onto empty arenas. Here the boxes come from reviews: a corrected opponent play says which card
appeared when and where, and the unit is followed through its track to get several poses.
Slices land in data/slices/<class>/ with a manifest for approval in the gallery page.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import lancedb
import numpy as np

from bot77.calibrate import read_frame
from bot77.layout import Layout
from bot77.spawns import track_units

SLICE_DIR = Path("data/slices")
SAM_WEIGHTS = Path("data/models/sam/sam2.1_t.pt")
POSE_EVERY_S = 1.0  # one slice per second of a unit's life
MAX_POSES = 10
BORN_WITH_S, BORN_WITH_TILES = 0.8, 3.0  # units born this close to the play belong to it
BADGE_CLASSES = {"bar-level", "bar", "skeleton-king-bar"}
OVERLAY_CLASSES = BADGE_CLASSES | {"clock", "text", "elixir", "emote"}  # drawn over units; never part of a sprite


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 2:
        return mask
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
BOX_PAD = 0.0  # detector boxes already hug the unit; padding pulls in the level badge (its own class)


class Cutter:
    def __init__(self):
        from ultralytics import SAM

        self.sam = SAM(str(SAM_WEIGHTS))

    def cut(self, frame: np.ndarray, box: tuple[float, float, float, float],
            erase: list[tuple[float, float, float, float]] = ()) -> np.ndarray | None:
        """RGBA slice of the unit inside `box` (frame pixels), tight to SAM's mask. `erase`
        boxes (the unit's level badge and HP bar, which are their own classes) are cut out of
        the mask so they aren't baked into the sprite [Wu et al. 2025, §3.1]."""
        x0, y0, x1, y1 = box
        pw, ph = (x1 - x0) * BOX_PAD, (y1 - y0) * BOX_PAD
        b = [max(x0 - pw, 0), max(y0 - ph, 0), min(x1 + pw, frame.shape[1]), min(y1 + ph, frame.shape[0])]
        res = self.sam(frame, bboxes=[b], verbose=False)[0]
        if res.masks is None or not len(res.masks.data):
            return None
        mask = res.masks.data[0].cpu().numpy().astype(bool)
        for ex0, ey0, ex1, ey1 in erase:
            mask[max(int(ey0) - 2, 0):int(ey1) + 2, max(int(ex0) - 2, 0):int(ex1) + 2] = False
        ys, xs = np.where(mask)
        if len(xs) < 50:
            return None
        cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgba = cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask[cy0:cy1, cx0:cx1].astype(np.uint8) * 255
        return rgba


def slices_from_truth(truth_path: Path, labels: set[str], db_dir: Path = Path("data/lancedb"),
                      out: Path = SLICE_DIR, log=print) -> list[dict]:
    truth = json.loads(truth_path.read_text())
    db = lancedb.connect(db_dir)
    video = db.open_table("videos").search().where(f"video_id = '{truth['video_id']}'").to_arrow().to_pylist()[0]
    layout = Layout.load(video["layout_id"])
    ax0, ay0, ax1, _ = layout.katacr_arena
    tile = (ax1 - ax0) / 18
    names = [c["name"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
             .select(["name"]).to_arrow().to_pylist()]
    events = {(e["match_id"], round(e["t_video"], 2)): e for e in db.open_table("opponent_events").search()
              .where(f"video_id = '{truth['video_id']}'").limit(100_000).to_arrow().to_pylist()}
    cutter = Cutter()
    manifest = []
    for item in truth["items"]:
        if item["label"] not in labels:
            continue
        ev = events.get((item["match_id"], item["t_video"]))
        if ev is None:  # a missed play: no known location yet
            continue
        dets = db.open_table("unit_detections").search().where(
            f"match_id = '{item['match_id']}' AND t_video >= {item['t_video'] - 2} AND t_video <= {item['t_video'] + 15}"
        ).limit(10**6).to_arrow().to_pylist()
        # positions in tiles of the arena crop, as in opponent_events
        dets = [{**d, "x0": d["x0"] - ax0, "x1": d["x1"] - ax0, "y0": d["y0"] - ay0, "y1": d["y1"] - ay0} for d in dets]
        tracks = [tr for tr in track_units(dets, names, tile, lambda c: False) if tr.team == "enemy"
                  and abs(tr.first_t - item["t_video"]) <= BORN_WITH_S
                  and (tr.dets[0][1] - ev["x_tiles"]) ** 2
                  + (tr.dets[0][2] - ev["y_tiles"]) ** 2 <= BORN_WITH_TILES ** 2]
        for ti, tr in enumerate(sorted(tracks, key=lambda tr: -np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in tr.boxes]))):
            picked, last = [], -1e9
            for (t, _, _), box in zip(tr.dets, tr.boxes):
                if t - last >= POSE_EVERY_S and len(picked) < MAX_POSES:
                    picked.append((t, box))
                    last = t
            cls = item["label"].lower().replace(" ", "-") + "-candidate"
            for t, (bx0, by0, bx1, by1) in picked:
                frame = read_frame(Path(video["path"]), t)
                s = layout.scale(frame.shape[1])
                # badges / HP bars detected in this frame that touch the unit's box
                bars = [((d["x0"] + ax0) * s, (d["y0"] + ay0) * s, (d["x1"] + ax0) * s, (d["y1"] + ay0) * s)
                        for d in dets if d["cls"] in BADGE_CLASSES and abs(d["t_video"] - t) < 0.01
                        and d["x1"] >= bx0 and d["x0"] <= bx1 and d["y1"] >= by0 - (by1 - by0) * 0.3 and d["y0"] <= by1]
                rgba = cutter.cut(frame, ((bx0 + ax0) * s, (by0 + ay0) * s, (bx1 + ax0) * s, (by1 + ay0) * s), bars)
                if rgba is None:
                    continue
                d = out / cls
                d.mkdir(parents=True, exist_ok=True)
                name = f"{cls}_1_{item['match_id']}_{t:07.2f}_{ti}.png"
                cv2.imwrite(str(d / name), rgba)
                manifest.append({"file": str(d / name), "card": item["label"], "team": "enemy", "video_id": truth["video_id"],
                                 "match_id": item["match_id"], "t_video": round(t, 2), "track": ti,
                                 "detector_class": tr.cls, "size": list(rgba.shape[:2])})
        log(f"{item['label']} @ {item['t_video']}: {len(tracks)} unit track(s)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _crop_size(c: dict, video: dict, frame: np.ndarray) -> tuple[int, int]:
    """The shown crop's size in frame pixels. Exports from the first click page lack it; it's
    recoverable from the layout's detector crop, exactly as bot77.clicks builds the page."""
    if "crop_w" in c:
        return c["crop_w"], c["crop_h"]
    layout = Layout.load(video["layout_id"])
    s = layout.scale(frame.shape[1])
    x0, y0, x1, y1 = layout.katacr_arena
    cx0, cy0 = max(int(x0 * s), 0), max(int(y0 * s), 0)
    return min(int(x1 * s), frame.shape[1]) - cx0, min(int(y1 * s), frame.shape[0]) - cy0


def slices_from_clicks(clicks_path: Path, db_dir: Path = Path("data/lancedb"), out: Path = SLICE_DIR / "clicked") -> list[dict]:
    """Cut a slice at each reviewer click (SAM point prompt), erasing detected badges/bars."""
    from ultralytics import SAM

    data = json.loads(clicks_path.read_text())
    db = lancedb.connect(db_dir)
    sam = SAM(str(SAM_WEIGHTS))
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in data["clicks"]:
        card = c.get("card") or data["card"]
        video = db.open_table("videos").search().where(f"video_id = '{c['video_id']}'").to_arrow().to_pylist()[0]
        frame = read_frame(Path(video["path"]), c["t"])
        crop_w, crop_h = _crop_size(c, video, frame)
        x = c["offset_x"] + c["u"] * crop_w  # u, v: fractions of the shown crop
        y = c["offset_y"] + c["v"] * crop_h
        res = sam(frame, points=[[x, y]], labels=[1], verbose=False)[0]
        if res.masks is None or not len(res.masks.data):
            continue
        mask = res.masks.data[0].cpu().numpy().astype(bool)
        # erase overlays the detector saw in the nearest detection frame (badge, HP bar, deploy clock)
        near = db.open_table("unit_detections").search().where(
            f"match_id = '{c['match_id']}' AND t_video >= {c['t'] - 0.3} AND t_video <= {c['t'] + 0.3}").limit(10**4).to_arrow().to_pylist()
        if near:
            t_near = min({d["t_video"] for d in near}, key=lambda t: abs(t - c["t"]))
            for d in near:
                if d["t_video"] == t_near and d["cls"] in OVERLAY_CLASSES:
                    mask[max(int(d["y0"]) - 2, 0):int(d["y1"]) + 2, max(int(d["x0"]) - 2, 0):int(d["x1"]) + 2] = False
        mask = _largest_component(mask)
        ys, xs = np.where(mask)
        if len(xs) < 50:
            continue
        cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgba = cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask[cy0:cy1, cx0:cx1].astype(np.uint8) * 255
        d = out / c["entity"]
        d.mkdir(parents=True, exist_ok=True)
        name = f"{c['entity']}_1_{c['frame']}.png"
        cv2.imwrite(str(d / name), rgba)
        manifest.append({"file": str(d / name), "card": card, "entity": c["entity"], "team": "enemy",
                         "video_id": c["video_id"], "match_id": c["match_id"], "t_video": c["t"], "click": [x, y],
                         "size": list(rgba.shape[:2])})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def strip_leftover_badge(rgba: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove a level badge the detector didn't box: a roughly square red/blue blob with white
    digits in the top third of the slice (plus its crown above), and thin badge-edge strips left
    over from an erased box. Returns (slice, changed)."""
    hsv = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    visible = rgba[:, :, 3] > 0
    # red (enemy), blue (friendly) or gold (some captures draw badges gold)
    badge = (((h < 8) | (h > 168)) | ((h >= 95) & (h <= 118)) | ((h >= 15) & (h <= 32))) & (s > 150) & (v > 120) & visible
    white = (s < 70) & (v > 200)
    n, _, st, _ = cv2.connectedComponentsWithStats(badge.astype(np.uint8))
    H = rgba.shape[0]
    out, changed = rgba.copy(), False
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        top = y + hh / 2 < H * 0.35
        square = 20 <= w <= 60 and 20 <= hh <= 60 and 0.6 <= w / hh <= 1.6 and white[y:y + hh, x:x + w].mean() > 0.08
        strip = hh <= 6 and w >= 20 and y < H * 0.2
        if top and (square or strip):
            out[max(y - (hh // 2 if square else 2), 0):y + hh + 2, max(x - 2, 0):x + w + 2, 3] = 0
            changed = True
    return out, changed


def locate_label(video: Path, layout: Layout, card: str, t: float) -> tuple[float, float] | None:
    """Where our own play landed: the "<Card> lvl.N" text the game draws at the drop spot for the POV
    player's plays (decisions.md, 2026-09-24). Returns frame-pixel (x, y) of the label centre."""
    import difflib
    import re as _re

    from ocrmac import ocrmac
    from PIL import Image

    x0, y0, x1, y1 = layout.katacr_arena
    want = _re.sub(r"[^a-z]", "", card.lower())
    for dt in (0.2, 0.5, 0.9, 1.3):
        f = read_frame(video, t + dt)
        s = layout.scale(f.shape[1])
        cx0, cy0 = max(int(x0 * s), 0), max(int(y0 * s), 0)
        crop = f[cy0:int(y1 * s), cx0:int(x1 * s)]
        H, W = crop.shape[:2]
        for text, _conf, (bx, by, bw, bh) in ocrmac.OCR(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                                                       recognition_level="accurate").recognize():
            got = _re.sub(r"[^a-z]", "", text.lower())
            if len(got) >= 4 and difflib.SequenceMatcher(None, want, got).ratio() >= 0.7:
                return cx0 + (bx + bw / 2) * W, cy0 + (1 - (by + bh / 2)) * H  # Vision boxes: origin bottom-left
    return None


def slices_from_labels(items: list[dict], out: Path, db_dir: Path = Path("data/lancedb"), poses: int = 4,
                       log=print) -> list[dict]:
    """Auto-cut candidates for labelled plays. `items`: {video_id, match_id, t_video, label, team, form}.
    The unit born with the play (same team, within BORN_WITH_S of t, on the owner's half) is taken
    from the old detector's tracks — its class name is ignored, only its boxes are used — and SAM cuts
    up to `poses` poses, one per POSE_EVERY_S. Candidates go to the gallery for approval."""
    db = lancedb.connect(db_dir)
    names = [c["name"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
             .select(["name"]).to_arrow().to_pylist()]
    videos = {v["video_id"]: v for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
    cutter = Cutter()
    manifest = []
    for k, it in enumerate(items):
        v = videos[it["video_id"]]
        layout = Layout.load(v["layout_id"])
        ax0, ay0, ax1, _ = layout.katacr_arena
        tile = (ax1 - ax0) / 18
        dets = db.open_table("unit_detections").search().where(
            f"match_id = '{it['match_id']}' AND t_video >= {it['t_video'] - 1} AND t_video <= {it['t_video'] + 12}"
        ).limit(10**6).to_arrow().to_pylist()
        rel = [{**d, "x0": d["x0"] - ax0, "x1": d["x1"] - ax0, "y0": d["y0"] - ay0, "y1": d["y1"] - ay0} for d in dets]
        tracks = [tr for tr in track_units(rel, names, tile, lambda c: False)
                  if tr.team == it["team"] and -0.5 <= tr.first_t - it["t_video"] <= 3.0 and len(tr.dets) >= 2]
        if not tracks:
            continue
        # POV plays: the on-screen name label marks the drop spot; only a unit born right there counts
        spot = locate_label(Path(v["path"]), layout, it["label"], it["t_video"]) if it["team"] == "friendly" else None
        if it["team"] == "friendly":
            if spot is None:
                continue
            s0 = layout.scale(1080)  # tracks are in layout pixels; labels in frame pixels (same for 1080-wide videos)
            lx, ly = (spot[0] / s0 - ax0) / tile, (spot[1] / s0 - ay0) / tile
            tracks = [tr for tr in tracks if ((tr.dets[0][1] - lx) ** 2 + (tr.dets[0][2] - ly - 1.0) ** 2) ** 0.5 <= 3.0]
            if not tracks:
                continue
        # the unit born closest in time to the play (the drop animation lasts ~0.3-1 s), on the owner's half
        river = 14.9
        own = [tr for tr in tracks if (tr.dets[0][2] >= river - 0.8) == (it["team"] == "friendly")]
        tr = min(own or tracks, key=lambda tr: (abs(tr.first_t - it["t_video"] - 0.5), -len(tr.dets)))
        picked, last = [], -1e9
        for (t, _, _), box in zip(tr.dets, tr.boxes):
            if t - last >= POSE_EVERY_S and len(picked) < poses:
                picked.append((t, box)); last = t
        slug = it["label"].lower().replace(" ", "-").replace(".", "") + ("" if it.get("form") in (None, "normal") else f"-{it['form']}")
        for t, (bx0, by0, bx1, by1) in picked:
            frame = read_frame(Path(v["path"]), t)
            s = layout.scale(frame.shape[1])
            bars = [((d["x0"]) * s, d["y0"] * s, d["x1"] * s, d["y1"] * s) for d in dets
                    if d["cls"] in OVERLAY_CLASSES and abs(d["t_video"] - t) < 0.01]
            rgba = cutter.cut(frame, ((bx0 + ax0) * s, (by0 + ay0) * s, (bx1 + ax0) * s, (by1 + ay0) * s), bars)
            if rgba is None:
                continue
            rgba, _ = strip_leftover_badge(rgba)
            a = _largest_component(rgba[:, :, 3] > 0)
            if a.sum() < 200:
                continue
            rgba[:, :, 3] = a.astype(np.uint8) * 255
            ys, xs = np.where(a)
            rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            d = out / slug
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{slug}_{it['match_id']}_{t:07.2f}.png"
            cv2.imwrite(str(f), rgba)
            manifest.append({"file": str(f), "card": it["label"], "form": it.get("form"), "entity": slug, "team": it["team"],
                             "video_id": it["video_id"], "match_id": it["match_id"], "t_video": round(t, 2),
                             "play_t": it["t_video"], "source": it.get("source", ""),
                             "caption": f"{it['video_id'][:14]} {int(t // 60)}:{int(t % 60):02d} · {it.get('source', '')}"})
        if k % 25 == 0:
            log(f"{k}/{len(items)} plays, {len(manifest)} candidates")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
