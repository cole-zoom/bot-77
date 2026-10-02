"""Auto-label real frames for detector training (decisions.md, 2026-09-30).

For each trusted label (a POV play, or a paired opponent play from the dual-POV recordings):
1. **Where**: our own plays show "<Card> lvl.N" at the drop spot (OCR). For a paired opponent play, the
   partner's label in *their* video gives the spot, point-mirrored across the arena into ours.
2. **The unit**: the old detector's track born next to that spot right after the play (its class name is
   ignored — only its boxes are used). Tracks are checked against Cole's approved slices with DINOv2
   (multi-entity cards pick the best-matching entity); spells get an effect box of their known radius.
3. **The frame**: the KataCR arena crop (576×896) at a detection timestamp, labelled with the new box plus the
   old detector's boxes for every other unit (provisional; covered by Cole's spot-check). Old boxes overlapping
   the new one are dropped (they're the old detector's wrong name for the same unit).
Output: YOLO-format images/labels under data/external/real_v1 and a manifest with provenance.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import lancedb
import numpy as np

from bot77.calibrate import read_frame
from bot77.detect.names import NON_UNITS
from bot77.detect.units import INPUT_WH, UnitDetector
from bot77.layout import Layout
from bot77.slices import locate_label
from bot77.spawns import RIVER_Y_TILES, track_units

OUT = Path("data/external/real_v1")
CLASSES = Path("config/classes_v2.json")
FRAMES_PER_PLAY = 6  # 2026-09-30: 3 -> 6 (units move and animate; later frames add variety)
FRAME_GAP_S = 0.8
SPOT_RADIUS_TILES = 2.5
PAIRED_RADIUS_TILES = 3.5  # mirrored spots carry the error of two videos' calibrations
LABEL_TO_UNIT_TILES = 1.0  # the name label is drawn about a tile above the unit
MIN_SIM = 0.5  # spot-check round 1: below 0.5 the wrong-unit rate was ~23%, above it ~10%  # DINOv2 similarity to the entity's approved slices, when it has any
SPELL_RADIUS = {"Vines": 2.0, "Void": 2.5, "Goblin Curse": 3.0, "Earthquake": 3.5}
OLD_MIN_CONF = 0.5
# evo Giant Snowball rolls after landing: box the ball (Cole, 2026-10-01: area boxes were all "box off")
ROLLING = {"giant-snowball-evolution"}
MIN_AFTER_PLAY_S = 0.3  # frames before this are the card's drag preview, not the unit
CLICK_REFS = 40  # clicked real-frame crops per entity used as anchors
KNN_K, KNN_DECISIVE = 7, 5  # relabel a grey box only when >=5 of its 7 nearest labelled real crops agree
SAM_WEIGHTS = Path("data/models/sam/sam2.1_t.pt")
# Cards whose deploy is several identical units: box every unit born together (Cole, spot-check round 2:
# only one of four evo Royal Hogs was boxed).
UNITS = {"Royal Hogs": 4, "Bats": 5, "Skeletons": 3, "Goblins": 4, "Barbarians": 5, "Spear Goblins": 3, "Minions": 3,
         "Guards": 3, "Royal Recruits": 6, "Goblin Barrel": 3, "Archers": 2, "Wall Breakers": 2, "Goblin Gang": 6,
         "Skeleton Army": 15, "Minion Horde": 6, "Elite Barbarians": 2, "Rascals": 3, "Zappies": 3, "Fire Spirit": 1}
# Spells that drop units (API type "spell"): label the units, not an effect radius. Round 2 boxed evo Goblin
# Barrel and Spirit Empress as radii.
SPAWNING_SPELLS = {"Goblin Barrel", "Spirit Empress", "Royal Delivery", "Barbarian Barrel", "Graveyard", "Goblin Drill"}
MULTI = {"Goblinstein": ["goblinstein-doctor", "goblinstein-monster"],
         "Spirit Empress": ["spirit-empress-ground", "spirit-empress-air"]}
# Spirit Empress costs 3 on the ground and 6 in the air (Cole, 2026-10-01: "placing with < 6 elixir" gives the
# ground form). The play detector expects 6, so ground plays come out at medium confidence; accept those.
SE_GROUND_MAX_COST = 4.5


def spirit_empress_form(e: dict) -> str | None:
    """Entity for a Spirit Empress play event from its measured cost, or None if the play isn't trusted."""
    if e["measured_cost"] is not None and e["measured_cost"] < SE_GROUND_MAX_COST:
        return "spirit-empress-ground" if e["confidence"] in ("high", "medium") else None
    return "spirit-empress-air" if e["confidence"] == "high" else None


KEEP_OLD = {"bar", "bar-level", "clock", "elixir", "emote", "king-tower", "queen-tower", "tower-bar", "king-tower-bar",
            "cannoneer-tower", "dagger-duchess-tower", "dagger-duchess-tower-bar"}  # HUD-like classes the model still needs


def slug(card: str, form: str | None) -> str:
    base = card.lower().replace(" ", "-").replace(".", "")
    if form == "evo":
        return f"{base}-evolution"
    if form == "hero":
        return f"{base}-hero"
    return base


def class_list() -> list[str]:
    """KataCR's classes (as used by v1) + every entity we have labels or slices for, + every evo and hero form
    in the card table (Cole, 2026-10-01: all forms get their own class).

    Append-only: label files store class *indices*, so the order already in config/classes_v2.json is kept and
    new classes go at the end. (It used to be re-sorted, which would have shifted indices when a class was added.)"""
    import yaml

    names = list(yaml.safe_load(open("data/external/synth_v1/data.yaml"))["names"].values())
    extra = sorted({m["class"] for m in json.loads(Path("data/slices/approved/manifest.json").read_text())} |
                   {"ronin", "rune-giant", "berserker", "boss-bandit", "goblin-curse", "goblin-demolisher", "goblin-machine",
                    "suspicious-bush", "vines", "void", "tombstone-hero", "tomb-queen", "spirit-empress-air",
                    "spirit-empress-ground", "dark-prince-hero", "dark-prince-hero-dismounted"})
    cards = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000) \
        .select(["name", "has_evo", "has_hero"]).to_arrow().to_pylist()
    forms = sorted({FORM_CLASS.get(slug(c["name"], f), slug(c["name"], f)) for c in cards for f, has in
                    (("evo", c["has_evo"]), ("hero", c["has_hero"])) if has})
    out = json.loads(CLASSES.read_text()) if CLASSES.exists() else names[:]
    for c in names + extra + forms:
        if c not in out:
            out.append(c)
    return out


# card-level slug -> class name where KataCR's naming differs (singular unit names; existing evo classes)
FORM_CLASS = {"archers-evolution": "archer-evolution", "barbarians-evolution": "barbarian-evolution",
              "bats-evolution": "bats-evolution", "skeletons-evolution": "skeletons-evolution",
              "royal-recruits-evolution": "royal-recruit-evolution", "wall-breakers-evolution": "wall-breaker-evolution",
              "royal-hogs-evolution": "royal-hogs-evolution", "goblins-hero": "goblins-hero",
              "minion-horde-evolution": "minion-horde-evolution", "elite-barbarians-evolution": "elite-barbarian-evolution",
              "skeleton-army-evolution": "skeleton-army-evolution"}


class Anchor:
    """DINOv2 similarity of a crop to Cole's approved slices of each entity, plus up to CLICK_REFS of the
    real-frame crops he clicked (2026-10-01: with one approved slice, hero Valkyrie / evo Goblin Drill plays in
    his own recordings were mostly rejected as low-similarity). With clicks the yield went 12 -> 66 frames but
    about half the boxes were wrong in a sample, so click anchors are opt-in and their output is reviewed."""

    def __init__(self, use_clicks: bool = False):
        from bot77.mine import Embedder, slice_to_bgr

        self.emb = Embedder()
        by = defaultdict(list)
        for m in json.loads(Path("data/slices/approved/manifest.json").read_text()):
            by[m["class"]].append(slice_to_bgr(cv2.imread(m["file"], cv2.IMREAD_UNCHANGED)))
        clicks = OUT / "manifest_clicks.json"
        if not use_clicks:  # opt-in: same-arena crops lift every crop's similarity (backgrounds match), so
            clicks = Path("/nonexistent")  # click-anchored labels go to Cole for review, not straight in
        per = defaultdict(int)
        for m in (json.loads(clicks.read_text()) if clicks.exists() else []):
            if per[m["entity"]] >= CLICK_REFS:
                continue
            img, lbl = OUT / "images" / f"{m['name']}.jpg", OUT / "labels" / f"{m['name']}.txt"
            if not (img.exists() and lbl.exists()):
                continue
            im = cv2.imread(str(img))
            H, W = im.shape[:2]
            _, cx, cy, w, h = (float(x) for x in lbl.read_text().splitlines()[0].split())
            crop = im[max(int((cy - h / 2) * H), 0):int((cy + h / 2) * H), max(int((cx - w / 2) * W), 0):int((cx + w / 2) * W)]
            if crop.size:
                by[m["entity"]].append(crop)
                per[m["entity"]] += 1
        self.refs = {c: self.emb(ims) for c, ims in by.items() if ims}

    def score(self, crop: np.ndarray, entity: str) -> float | None:
        if entity not in self.refs or crop.size == 0:
            return None
        return float((self.refs[entity] @ self.emb([crop])[0]).max())


def river_tiles(layout: Layout) -> float:
    """The arena's mirror line in detector-crop tiles, from the two princess towers' HP bars (drawn at the
    same height above each tower, so their midpoint sits a fixed distance from the true centre; that
    distance is calibrated on Alina's layout, where the river is at RIVER_Y_TILES)."""
    def mid(lay):
        a = _tiles(lay, 0, (lay.regions["opp_left_hp"][1] + lay.regions["opp_left_hp"][3]) / 2)[1]
        b = _tiles(lay, 0, (lay.regions["my_left_hp"][1] + lay.regions["my_left_hp"][3]) / 2)[1]
        return (a + b) / 2
    return mid(layout) + (RIVER_Y_TILES - mid(Layout.load("alina_portrait")))


def _tiles(layout: Layout, x: float, y: float) -> tuple[float, float]:
    ax0, ay0, ax1, _ = layout.katacr_arena
    t = (ax1 - ax0) / 18
    return (x - ax0) / t, (y - ay0) / t


def _px(layout: Layout, tx: float, ty: float) -> tuple[float, float]:
    ax0, ay0, ax1, _ = layout.katacr_arena
    t = (ax1 - ax0) / 18
    return ax0 + tx * t, ay0 + ty * t


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


class AutoLabeler:
    def __init__(self, db_dir: Path = Path("data/lancedb"), out: Path = OUT, click_anchors: bool = False):
        self.db = lancedb.connect(db_dir)
        self.out = out
        self.classes = class_list()
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        self.videos = {v["video_id"]: v for v in self.db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
        cards = self.db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
        self.card_names = [c["name"] for c in cards]
        self.ctype = {c["name"]: c["type"] for c in cards}
        self.anchor = Anchor(use_clicks=click_anchors)
        from ultralytics import SAM

        self.sam = SAM(str(SAM_WEIGHTS))
        self.knn = self.db.open_table("real_crop_index")  # KataCR's hand-labelled real crops
        self.stats = defaultdict(int)
        self.dets = self.db.open_table("unit_detections")
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        CLASSES.write_text(json.dumps(self.classes, indent=1))

    # -- locating the play

    def spot(self, item: dict) -> tuple[float, float] | None:
        """Drop spot in the item's own video, frame pixels (unit position, not the label)."""
        if item["team"] == "friendly":
            v = self.videos[item["video_id"]]
            lay = Layout.load(v["layout_id"])
            p = locate_label(Path(v["path"]), lay, item["card"], item["t_video"])
            if p is None:
                return None
            tx, ty = _tiles(lay, *p)
            return _px(lay, tx, ty + LABEL_TO_UNIT_TILES)
        # paired opponent play: the partner's own label, mirrored across the arena
        pv = self.videos[item["partner_video"]]
        play = locate_label(Path(pv["path"]), Layout.load(pv["layout_id"]), item["card"], item["partner_t"])
        if play is None:
            return None
        pl, ml = Layout.load(pv["layout_id"]), Layout.load(self.videos[item["video_id"]]["layout_id"])
        tx, ty = _tiles(pl, *play)
        ty += LABEL_TO_UNIT_TILES
        # point-mirror: distance past the partner's river becomes distance before ours
        return _px(ml, 18 - tx, river_tiles(ml) - (ty - river_tiles(pl)))

    # -- one labelled play -> training frames

    def label(self, item: dict, rng: random.Random) -> list[dict]:
        v = self.videos[item["video_id"]]
        lay = Layout.load(v["layout_id"])
        spot = self.spot(item)
        if spot is None:
            self.stats["drop_no_spot"] += 1
            return []
        t0 = item["t_video"]
        rows = self.dets.search().where(
            f"match_id = '{item['match_id']}' AND t_video >= {t0 - 0.5} AND t_video <= {t0 + 12}").limit(10**6).to_arrow().to_pylist()
        if not rows:
            self.stats["drop_no_detections"] += 1
            return []
        by_t = defaultdict(list)
        for d in rows:
            by_t[round(d["t_video"], 3)].append(d)
        times = sorted(by_t)
        spell = self.ctype.get(item["card"]) == "spell" and item["card"] not in SPAWNING_SPELLS \
            and item["entity"] not in ROLLING  # a rolling ball is boxed as an object, not an effect area
        if spell and item["source"] == "paired":
            return []  # spot-check round 1: mirrored spell spots were 0/9 right
        entities = [item["entity"]] if item["entity"] not in MULTI else MULTI[item["entity"]]
        targets = []  # (t, box, entity, how, sim)
        if spell:
            r = SPELL_RADIUS.get(item["card"], 2.5) * (lay.katacr_arena[2] - lay.katacr_arena[0]) / 18
            for t in [x for x in times if 0.3 <= x - t0 <= 2.0][:2]:
                targets.append((t, (spot[0] - r, spot[1] - r, spot[0] + r, spot[1] + r), entities[0], "spell-radius", None))
        else:
            ax0, ay0 = lay.katacr_arena[:2]
            tile = (lay.katacr_arena[2] - ax0) / 18
            rel = [{**d, "x0": d["x0"] - ax0, "x1": d["x1"] - ax0, "y0": d["y0"] - ay0, "y1": d["y1"] - ay0}
                   for d in rows if d["cls"] not in NON_UNITS]
            sx, sy = _tiles(lay, *spot)
            tracks = [tr for tr in track_units(rel, self.card_names, tile, lambda c: False)
                      if -0.3 <= tr.first_t - t0 <= 3.0 and len(tr.dets) >= 2
                      and ((tr.dets[0][1] - sx) ** 2 + (tr.dets[0][2] - sy) ** 2) ** 0.5
                      <= (PAIRED_RADIUS_TILES if item["source"] == "paired" else SPOT_RADIUS_TILES)
                      + (1.5 if UNITS.get(item["card"], 1) > 1 else 0)]  # groups spread out (evo hogs fly apart)
            tracks.sort(key=lambda tr: ((tr.dets[0][1] - sx) ** 2 + (tr.dets[0][2] - sy) ** 2))
            if not tracks:
                self.stats["drop_no_track_near_spot"] += 1
            used = set()
            n_units = UNITS.get(item["card"], 1)
            if n_units > 1:  # identical units: one entity, every unit born with the play
                entities = entities * n_units
            for tr in tracks[:len(entities) + 1]:
                picked, last = [], -1e9
                for (t, _, _), b in zip(tr.dets, tr.boxes):
                    if t < t0 + MIN_AFTER_PLAY_S:
                        continue  # still the drag preview (Cole, 2026-10-01: boxes on a hovered Goblin Drill)
                    if t - last >= FRAME_GAP_S and len(picked) < FRAMES_PER_PLAY:
                        picked.append((t, (b[0] + ax0, b[1] + ay0, b[2] + ax0, b[3] + ay0)))
                        last = t
                if not picked:
                    continue
                f = read_frame(Path(v["path"]), picked[0][0])
                s = lay.scale(f.shape[1])
                x0, y0, x1, y1 = (int(c * s) for c in picked[0][1])
                crop = f[max(y0, 0):y1, max(x0, 0):x1]
                sims = {e: self.anchor.score(crop, e) for e in entities if (e not in used or n_units > 1)}
                known = {e: sc for e, sc in sims.items() if sc is not None}
                if known:
                    e, sc = max(known.items(), key=lambda kv: kv[1])
                    if sc < MIN_SIM:
                        self.stats["drop_low_sim"] += 1
                        continue
                else:
                    e, sc = next(iter(sims)), None
                used.add(e)
                for t, b in picked:
                    targets.append((t, b, e, "track", sc))
                if n_units == 1 and len(used) == len(entities):
                    break
        if not spell and UNITS.get(item["card"], 1) > 1 and targets:
            # one frame per sampled time, with every unit's box at that time (each unit's nearest detection)
            group = [tr for tr in tracks[:UNITS[item["card"]] + 1]]
            ent, how0, sc0 = targets[0][2], targets[0][3], targets[0][4]
            frames = []
            for t in sorted({round(x[0], 2) for x in targets if x[2] == ent})[:FRAMES_PER_PLAY]:
                boxes = []
                for tr in group:
                    near = [(abs(tt - t), b) for (tt, _, _), b in zip(tr.dets, tr.boxes) if abs(tt - t) <= 0.2]
                    if near:
                        b = min(near)[1]
                        boxes.append((b[0] + ax0, b[1] + ay0, b[2] + ax0, b[3] + ay0))
                if len(boxes) < UNITS[item["card"]] - 1:
                    self.stats["group_incomplete_dropped"] += 1
                    continue  # a partly-boxed group teaches the unboxed units as background
                old = by_t[t] if t in by_t else by_t[min(times, key=lambda x: abs(x - t))]
                frames.append(self._write(item, v, lay, t, boxes, ent, how0, sc0, old))
            return frames
        return [self._write(item, v, lay, t, box, e, how, sim, by_t[t] if t in by_t else by_t[min(times, key=lambda x: abs(x - t))])
                for t, box, e, how, sim in targets]

    def refine(self, f: np.ndarray, box, scale: float):
        """Tighten a tracked box to the unit with SAM (the old detector sizes boxes for the card it thought
        it saw). Returns (box, ok): falls back to the original if the mask is implausible."""
        x0, y0, x1, y1 = (c * scale for c in box)
        w, h = x1 - x0, y1 - y0
        pad = [max(x0 - 0.15 * w, 0), max(y0 - 0.15 * h, 0), min(x1 + 0.15 * w, f.shape[1]), min(y1 + 0.15 * h, f.shape[0])]
        res = self.sam(f, bboxes=[pad], verbose=False)[0]
        if res.masks is None or not len(res.masks.data):
            return box, False
        ys, xs = np.where(res.masks.data[0].cpu().numpy())
        if len(xs) < 30:
            return box, False
        nb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        area = (nb[2] - nb[0]) * (nb[3] - nb[1])
        if not 0.25 <= area / max(w * h, 1) <= 1.6:  # grabbed a neighbour / the floor, or a sliver
            return box, False
        return tuple(c / scale for c in nb), True

    def relabel(self, f: np.ndarray, boxes: list, old_cls: list[str], scale: float) -> list[str]:
        """kNN vote over labelled real crops; overrides the old name only when decisive."""
        from collections import Counter

        crops, keep = [], []
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            c = f[max(int(y0 * scale), 0):int(y1 * scale), max(int(x0 * scale), 0):int(x1 * scale)]
            if c.shape[0] >= 8 and c.shape[1] >= 8:
                crops.append(c); keep.append(k)
        out = list(old_cls)
        if not crops:
            return out
        for k, q in zip(keep, self.anchor.emb(crops)):
            votes = Counter(r["cls"] for r in self.knn.search(q.tolist()).metric("cosine").limit(KNN_K).to_arrow().to_pylist())
            top, n = votes.most_common(1)[0]
            if top != old_cls[k] and n >= KNN_DECISIVE and top in self.cidx:
                out[k] = top
                self.stats["grey_relabelled"] += 1
            self.stats["grey_total"] += 1
        return out

    def _write(self, item, v, lay, t, box, entity, how, sim, old, entities: list[str] | None = None) -> dict:
        """`box`: one box or a list; `entities`: per-box entity when a frame holds several (click frames)."""
        f = read_frame(Path(v["path"]), t)
        scale = lay.scale(f.shape[1])
        boxes = box if isinstance(box, list) else [box]
        if how == "track":
            refined = [self.refine(f, b, scale) for b in boxes]
            boxes = [b for b, _ in refined]
            ok = all(o for _, o in refined)
            how = "track+sam" if ok else "track"
            self.stats["sam_ok" if ok else "sam_fallback"] += 1
        box = boxes[0]
        crop, (ox, oy), sx, sy = UnitDetector.arena(None, f, lay)
        img = cv2.resize(crop, INPUT_WH, interpolation=cv2.INTER_AREA)
        W, H = INPUT_WH

        def yolo(b):
            x0, y0, x1, y1 = (b[0] - ox) / sx, (b[1] - oy) / sy, (b[2] - ox) / sx, (b[3] - oy) / sy
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if x1 - x0 < 3 or y1 - y0 < 3:
                return None
            return f"{(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"

        ents = entities or [entity] * len(boxes)
        pairs = [(e, y) for e, y in zip(ents, (yolo(b) for b in boxes)) if y and e in self.cidx]
        if not pairs:
            return {}
        mains = [y for _, y in pairs]
        lines = [f"{self.cidx[e]} {y}" for e, y in pairs]
        n_old = 0
        grey = [d for d in old if d["conf"] >= OLD_MIN_CONF and d["cls"] in self.cidx
                and (d["cls"] in KEEP_OLD or max(_iou((d["x0"], d["y0"], d["x1"], d["y1"]), b) for b in boxes) <= 0.3)]
        units = [d for d in grey if d["cls"] not in KEEP_OLD]
        new_cls = self.relabel(f, [(d["x0"], d["y0"], d["x1"], d["y1"]) for d in units], [d["cls"] for d in units], scale)
        names = {id(d): c for d, c in zip(units, new_cls)}
        for d in grey:
            y = yolo((d["x0"], d["y0"], d["x1"], d["y1"]))
            if y:
                lines.append(f"{self.cidx[names.get(id(d), d['cls'])]} {y}")
                n_old += 1
        name = f"{item['match_id']}_{t:07.2f}_{entity}"
        cv2.imwrite(str(self.out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (self.out / "labels" / f"{name}.txt").write_text("\n".join(lines))
        return {"name": name, "video_id": item["video_id"], "match_id": item["match_id"], "t_video": round(t, 2),
                "entity": entity, "card": item["card"], "source": item["source"], "how": how,
                "sim": None if sim is None else round(sim, 3), "main_box": [round(c, 1) for c in box], "n_main": len(mains),
                "old_boxes": n_old}


def collect_items(db_dir: Path = Path("data/lancedb")) -> list[dict]:
    """Trusted labels to auto-label: POV plays of new cards and evo/hero forms, and paired opponent plays."""
    db = lancedb.connect(db_dir)
    new = {"Berserker", "Boss Bandit", "Goblin Curse", "Goblin Demolisher", "Goblin Machine", "Suspicious Bush", "Vines",
           "Void", "Ronin", "Rune Giant", "Spirit Empress", "Minion Giant", "Goblinstein"}
    ev = db.open_table("events").search().limit(10**7).select(
        ["video_id", "match_id", "t_video", "kind", "card", "form", "confidence", "measured_cost"]).to_arrow().to_pylist()
    items = []
    for e in ev:
        if e["kind"] != "play" or not e["card"]:
            continue
        if e["card"] == "Spirit Empress":
            ent = spirit_empress_form(e)
            if ent:
                items.append({"video_id": e["video_id"], "match_id": e["match_id"], "t_video": e["t_video"], "card": e["card"],
                              "entity": ent, "team": "friendly", "source": "POV"})
            continue
        if e["confidence"] != "high":
            continue
        if e["card"] in new or e["form"] in ("evo", "hero"):
            ent = e["card"] if e["card"] in MULTI and e["form"] in (None, "normal") else FORM_CLASS.get(slug(e["card"], e["form"]), slug(e["card"], e["form"]))
            items.append({"video_id": e["video_id"], "match_id": e["match_id"], "t_video": e["t_video"], "card": e["card"],
                          "entity": ent, "team": "friendly", "source": "POV"})
    import glob
    for f in glob.glob("data/ground_truth/opponent_truth_auto_*.json"):
        tr = json.loads(Path(f).read_text())
        partner = tr["paired_with"].rsplit("_m", 1)[0]
        sign = 1 if tr["matches"][0].startswith("cole") else -1  # offset is cole minus friend
        for i in tr["items"]:
            if i["label"] not in new and i.get("form") not in ("evo", "hero"):
                continue
            ent = i["label"] if i["label"] in MULTI and i.get("form") in (None, "normal") else FORM_CLASS.get(slug(i["label"], i.get("form")), slug(i["label"], i.get("form")))
            items.append({"video_id": tr["video_id"], "match_id": i["match_id"], "t_video": i["t_video"], "card": i["label"],
                          "entity": ent, "team": "enemy", "source": "paired", "partner_video": partner,
                          # pairing offset = t_friend - t_cole, so the partner's own play is at t ± offset
                          "partner_t": round(i["t_video"] + sign * tr["offset_s"], 2)})
    return items


def label_clicks(al: "AutoLabeler", clicks_path: Path, log=print) -> list[dict]:
    """Cole's clicks as real-frame labels: SAM from the click point gives the unit's box (only the box is
    needed here, not a clean cut-out, so spill-over that ruined a slice matters far less)."""
    data = json.loads(clicks_path.read_text())
    out = []
    frames = defaultdict(list)
    for c in data["clicks"]:
        frames[(c["video_id"], round(c["t"], 2))].append(c)
    pending = []  # (key, box, entity, click)
    for c in data["clicks"]:
        v = al.videos.get(c["video_id"])
        if v is None:
            continue
        lay = Layout.load(v["layout_id"])
        f = read_frame(Path(v["path"]), c["t"])
        crop_w = c.get("crop_w")
        if crop_w is None:
            s = lay.scale(f.shape[1])
            ax0, ay0, ax1, ay1 = lay.katacr_arena
            cx0, cy0 = max(int(ax0 * s), 0), max(int(ay0 * s), 0)
            crop_w, crop_h = min(int(ax1 * s), f.shape[1]) - cx0, min(int(ay1 * s), f.shape[0]) - cy0
        else:
            crop_h = c["crop_h"]
        x, y = c["offset_x"] + c["u"] * crop_w, c["offset_y"] + c["v"] * crop_h
        res = al.sam(f, points=[[x, y]], labels=[1], verbose=False)[0]
        if res.masks is None or not len(res.masks.data):
            continue
        mask = res.masks.data[0].cpu().numpy().astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        k = lab[int(min(y, f.shape[0] - 1)), int(min(x, f.shape[1] - 1))]  # the piece under the click
        if k == 0:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else 0
        if k == 0:
            continue
        bx, by, bw, bh = st[k][:4]
        s = lay.scale(f.shape[1])
        box = (bx / s, by / s, (bx + bw) / s, (by + bh) / s)
        tile = (lay.katacr_arena[2] - lay.katacr_arena[0]) / 18
        if bw / s > 6 * tile or bh / s > 6 * tile:  # mask ran across the arena
            continue
        match = c.get("match_id")
        if not match:  # click pages built from times don't carry the match; find it
            ms = al.db.open_table("matches").search().where(f"video_id = '{c['video_id']}'").to_arrow().to_pylist()
            match = next((m["match_id"] for m in ms if m["t_start"] - 5 <= c["t"] <= m["t_end"] + 5), None)
            if match is None:
                continue
        rows = al.dets.search().where(f"match_id = '{match}' AND t_video >= {c['t'] - 0.15} AND t_video <= {c['t'] + 0.15}") \
            .limit(10**4).to_arrow().to_pylist()
        pending.append(((c["video_id"], round(c["t"], 2)), box, c["entity"], c, match, rows, v, lay))
    by_frame = defaultdict(list)
    for p in pending:
        by_frame[p[0]].append(p)
    for key, ps in by_frame.items():  # one training frame per clicked frame, every clicked unit boxed
        c, match, rows, v, lay = ps[0][3], ps[0][4], ps[0][5], ps[0][6], ps[0][7]
        item = {"video_id": c["video_id"], "match_id": match, "card": c.get("card") or data.get("card"), "source": "click"}
        r = al._write(item, v, lay, c["t"], [p[1] for p in ps], ps[0][2], "click", None, rows, entities=[p[2] for p in ps])
        if r:
            out.append(r)
    log(f"{clicks_path.name}: {len(out)} frames from {len(data['clicks'])} clicks")
    return out
