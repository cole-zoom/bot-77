"""Run the unit detector over processed matches and store every detection in LanceDB."""

from __future__ import annotations

import time
from pathlib import Path

import lancedb
import pyarrow as pa

from bot77.detect.units import UnitDetector
from bot77.layout import Layout
from bot77.pipeline import _replace
from bot77.video import iter_frames

DETECT_FPS = 4

SCHEMA = pa.schema([
    ("match_id", pa.string()), ("video_id", pa.string()), ("t_video", pa.float32()),
    ("cls", pa.string()), ("team", pa.string()), ("conf", pa.float32()),
    ("x0", pa.float32()), ("y0", pa.float32()), ("x1", pa.float32()), ("y1", pa.float32()),
])


def detect_video(video_id: str, db_dir: Path, matches: list[int] | None = None, fps: float = DETECT_FPS, log=print,
                 detector=None, table: str = "unit_detections") -> int:
    db = lancedb.connect(db_dir)
    video = db.open_table("videos").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist()[0]
    layout = Layout.load(video["layout_id"])
    ms = sorted(db.open_table("matches").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist(),
                key=lambda m: m["index"])
    detector = detector or UnitDetector()
    rows = []
    for mi, m in enumerate(ms):
        if matches and m["index"] not in matches:
            continue
        t0 = max(m["t_start"], ms[mi - 1]["t_end"] if mi else 0.0, 0.0)
        span = m["t_end"] - t0
        start = time.time()
        n = 0
        for t, f in iter_frames(Path(video["path"]), fps, start=t0, duration=span):
            for d in detector.detect(f, layout):
                rows.append({"match_id": m["match_id"], "video_id": video_id, "t_video": t, "cls": d.name,
                             "team": d.team, "conf": d.conf, "x0": d.box[0], "y0": d.box[1], "x1": d.box[2], "y1": d.box[3]})
            n += 1
        log(f"{m['match_id']}: {n} frames in {time.time() - start:.0f}s")
    # replace only the matches we ran
    ran = {r["match_id"] for r in rows}
    if table in db.table_names(limit=10_000) and ran:
        t = db.open_table(table)
        t.delete(" OR ".join(f"match_id = '{m}'" for m in ran))
        t.add(pa.Table.from_pylist(rows, schema=SCHEMA))
    elif rows:
        db.create_table(table, pa.Table.from_pylist(rows, schema=SCHEMA))
    return len(rows)
