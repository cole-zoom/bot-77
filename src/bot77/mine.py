"""Mine training candidates with LanceDB vector search.

Every unit box in `unit_detections` (whatever class the old detector gave it) is cropped from
the video once a second, embedded with DINOv2, and stored in the LanceDB table `unit_crops`
with its match, time, team, box and old-detector class. Approved slices are embedded the same
way and used as queries: the nearest crops across all footage are candidates for that card,
including ones the old detector mislabelled. Candidates go to the gallery for approval.

This is the project's founding idea (mine moments from pro footage with LanceDB) applied to
training data. See decisions.md, 2026-09-28.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import lancedb
import numpy as np
import pyarrow as pa
import torch

from bot77.detect.names import NON_UNITS
from bot77.layout import Layout
from bot77.video import iter_frames

CROP_FPS = 1
MIN_CONF = 0.3
MIN_SIDE_PX = 24
PAD = 0.1
EMBED_SIZE = 224
BG = (128, 128, 128)  # slices and crops are embedded on the same neutral background
DIM = 384  # DINOv2 ViT-S/14

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Embedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(self.device)

    @torch.no_grad()
    def __call__(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([_square(im) for im in images_bgr])[:, :, :, ::-1] / 255.0  # RGB
        x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float()
        x = ((x - _MEAN) / _STD).to(self.device)
        v = self.model(x).float().cpu().numpy()
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _square(im: np.ndarray) -> np.ndarray:
    """Letterbox to a square on the neutral background, then resize."""
    h, w = im.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), BG, np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = im
    return cv2.resize(canvas, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)


def slice_to_bgr(rgba: np.ndarray) -> np.ndarray:
    a = rgba[:, :, 3:] / 255.0
    return (rgba[:, :, :3] * a + np.array(BG) * (1 - a)).astype(np.uint8)


SCHEMA = pa.schema([
    ("crop_id", pa.string()), ("video_id", pa.string()), ("match_id", pa.string()), ("t_video", pa.float32()),
    ("team", pa.string()), ("old_cls", pa.string()), ("conf", pa.float32()),
    ("x0", pa.float32()), ("y0", pa.float32()), ("x1", pa.float32()), ("y1", pa.float32()),
    ("image", pa.binary()), ("vector", pa.list_(pa.float32(), DIM)),
])


def build_crops(video_id: str, db_dir: Path = Path("data/lancedb"), log=print, batch: int = 64) -> int:
    db = lancedb.connect(db_dir)
    video = db.open_table("videos").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist()[0]
    layout = Layout.load(video["layout_id"])
    matches = sorted(db.open_table("matches").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist(),
                     key=lambda m: m["index"])
    emb = Embedder()
    total = 0
    for m in matches:
        dets = db.open_table("unit_detections").search().where(f"match_id = '{m['match_id']}'").limit(10**7).to_arrow().to_pylist()
        by_t: dict[int, list[dict]] = defaultdict(list)
        for d in dets:
            if d["cls"] in NON_UNITS or d["conf"] < MIN_CONF:
                continue
            if min(d["x1"] - d["x0"], d["y1"] - d["y0"]) < MIN_SIDE_PX:
                continue
            by_t[round(d["t_video"])].append(d)  # one sample per second
        if not by_t:
            continue
        rows, imgs = [], []

        def flush():
            nonlocal rows, imgs, total
            if not rows:
                return
            vecs = emb(imgs)
            for r, v in zip(rows, vecs):
                r["vector"] = v.tolist()
            _add(db, rows)
            total += len(rows)
            rows, imgs = [], []

        t0, t1 = min(by_t), max(by_t)
        seen = set()
        for t, f in iter_frames(Path(video["path"]), CROP_FPS, start=float(t0), duration=float(t1 - t0 + 1)):
            key = round(t)
            if key in seen or key not in by_t:
                continue
            seen.add(key)
            s = layout.scale(f.shape[1])
            for k, d in enumerate(by_t[key]):
                w, h = d["x1"] - d["x0"], d["y1"] - d["y0"]
                x0, y0 = int(max((d["x0"] - w * PAD) * s, 0)), int(max((d["y0"] - h * PAD) * s, 0))
                x1, y1 = int(min((d["x1"] + w * PAD) * s, f.shape[1])), int(min((d["y1"] + h * PAD) * s, f.shape[0]))
                crop = f[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                rows.append({"crop_id": f"{m['match_id']}_{key:05d}_{k:02d}", "video_id": video_id, "match_id": m["match_id"],
                             "t_video": float(key), "team": d["team"], "old_cls": d["cls"], "conf": d["conf"],
                             "x0": d["x0"], "y0": d["y0"], "x1": d["x1"], "y1": d["y1"],
                             "image": cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()})
                imgs.append(crop)
                if len(rows) >= batch:
                    flush()
        flush()
        log(f"{m['match_id']}: crops so far {total}")
    return total


def _add(db, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    if "unit_crops" in db.table_names(limit=10_000):
        db.open_table("unit_crops").add(table)
    else:
        db.create_table("unit_crops", table)


def search(slice_files: list[Path], db_dir: Path = Path("data/lancedb"), k: int = 60, where: str | None = None,
           exclude_near: list[tuple[str, float]] = (), per_second: int = 1) -> list[dict]:
    """Nearest unit crops to any of the given approved slices (max similarity over queries).
    `exclude_near`: (match_id, t_video) pairs already used as slices; crops within 2 s are skipped.
    At most `per_second` result per match-second, so one long-lived unit doesn't fill the page."""
    db = lancedb.connect(db_dir)
    table = db.open_table("unit_crops")
    emb = Embedder()
    queries = emb([slice_to_bgr(cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) for p in slice_files])
    best: dict[str, dict] = {}
    for q in queries:
        qb = table.search(q.tolist()).metric("cosine").limit(k * 4)
        if where:
            qb = qb.where(where, prefilter=True)
        for r in qb.to_arrow().to_pylist():
            sim = 1 - r["_distance"]
            if r["crop_id"] not in best or sim > best[r["crop_id"]]["similarity"]:
                best[r["crop_id"]] = {**{k2: v for k2, v in r.items() if k2 not in ("vector", "_distance")}, "similarity": sim}
    out, per = [], defaultdict(int)
    for r in sorted(best.values(), key=lambda r: -r["similarity"]):
        if any(r["match_id"] == mid and abs(r["t_video"] - t) <= 2 for mid, t in exclude_near):
            continue
        key = (r["match_id"], int(r["t_video"]))
        if per[key] >= per_second:
            continue
        per[key] += 1
        out.append(r)
        if len(out) >= k:
            break
    return out


def write_candidates(results: list[dict], card: str, entity: str, out_dir: Path) -> list[dict]:
    """Save search hits as files plus a gallery manifest."""
    d = out_dir / entity
    d.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in results:
        f = d / f"{r['crop_id']}.jpg"
        f.write_bytes(r["image"])
        manifest.append({"file": str(f), "card": card, "entity": entity, "match_id": r["match_id"], "video_id": r["video_id"],
                         "t_video": r["t_video"], "team": r["team"], "old_cls": r["old_cls"],
                         "box": [r["x0"], r["y0"], r["x1"], r["y1"]], "similarity": round(r["similarity"], 3),
                         "caption": f"{r['match_id'][-3:]} · {r['team']} · old: {r['old_cls']} · sim {r['similarity']:.2f}"})
    return manifest
