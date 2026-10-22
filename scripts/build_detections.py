"""Phase 1b of docs/training-architecture.md (2026-10-22): one `detections` table for every detector run, instead of a
table per run, plus a `runs` table that says what each run is.

detections: run_id, match_id, video_id, t, cls, card (bot77.detect.names.class_to_card; null for towers / non-units),
            team (friendly / enemy as on the video), conf, x0..y1 (video pixels, as before), x, y (arena tiles of the box
            centre, bottom player's frame: (centre - arena origin) / tile, tile = arena width / 18).
The old unit_detections_* tables stay until every reader is switched (phases 2-3), then go.

  uv run python scripts/build_detections.py
"""
from datetime import timedelta

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from bot77.detect.names import class_to_card
from bot77.layout import Layout

RUNS = [  # run_id, source table, what it is, role
    ("legacy", "unit_detections", "early detector run (old classes) over YouTube + POV games, the source of crop mining",
     "mining / slices / autolabel (old code paths)"),
    ("mine_v4", "unit_detections_mine4", "V3 epoch-10 on 4 fps arena crops around trusted plays (data engine v4, RunPod)",
     "source of the v4 autolabels"),
    ("v5l_e15", "unit_detections_v5l_e15", "detector V5 (epoch 15): all 524 keyed replays + check games; the POV default",
     "current: board states, placements, playbook"),
    ("v6l_best", "unit_detections_v6l_best", "detector V6 (best): the bot's live detector; the bot's replays",
     "current for the bot's games"),
    ("v7l_best", "unit_detections_v7l_best", "detector V7 (best): check sets only; not adopted", "history"),
    ("v8l_best", "unit_detections_v8l_best", "detector V8 (best): check sets only; no gain on dev, not adopted", "history"),
]
db = lancedb.connect("data/lancedb")
names = [c["name"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name"]).to_arrow().to_pylist()]
arena = {}
for v in db.open_table("videos").to_arrow().to_pylist():
    if v["layout_id"]:
        try:
            ax0, ay0, ax1, _ = Layout.load(v["layout_id"]).katacr_arena
            arena[v["video_id"]] = (ax0, ay0, (ax1 - ax0) / 18)
        except Exception:  # noqa: BLE001 -- a layout without an arena: positions stay null
            pass
card_of: dict[str, str | None] = {}
COLS = ["match_id", "video_id", "t_video", "cls", "team", "conf", "x0", "y0", "x1", "y1"]


def convert(t: pa.Table, run_id: str) -> pa.Table:
    cls = t.column("cls").to_pylist()
    for c in set(cls) - card_of.keys():
        card_of[c] = class_to_card(c, names)[0]
    vids = t.column("video_id").to_pylist()
    a = np.array([arena.get(v, (np.nan, np.nan, np.nan)) for v in vids], dtype=np.float64)
    cx = (t.column("x0").to_numpy() + t.column("x1").to_numpy()) / 2
    cy = (t.column("y0").to_numpy() + t.column("y1").to_numpy()) / 2
    x, y = (cx - a[:, 0]) / a[:, 2], (cy - a[:, 1]) / a[:, 2]
    return pa.table({"run_id": pa.array([run_id] * t.num_rows), "match_id": t.column("match_id"), "video_id": t.column("video_id"),
                     "t": t.column("t_video"), "cls": t.column("cls"), "card": pa.array([card_of[c] for c in cls], pa.string()),
                     "team": t.column("team"), "conf": t.column("conf"),
                     "x0": t.column("x0"), "y0": t.column("y0"), "x1": t.column("x1"), "y1": t.column("y1"),
                     "x": pa.array(x.astype(np.float32), mask=np.isnan(x)), "y": pa.array(y.astype(np.float32), mask=np.isnan(y))})


out, counts = None, {}
for run_id, table, what, role in RUNS:
    src = db.open_table(table)
    matches = sorted(set(src.search().select(["match_id"]).limit(10**9).to_arrow().column("match_id").unique().to_pylist()))
    n = 0
    for i in range(0, len(matches), 25):
        chunk = matches[i:i + 25]
        ids = ",".join("'" + m.replace("'", "''") + "'" for m in chunk)
        t = convert(src.search().where(f"match_id IN ({ids})").select(COLS).limit(10**8).to_arrow(), run_id)
        n += t.num_rows
        if out is None:
            out = db.create_table("detections", t, mode="overwrite")
        else:
            out.add(t)
    counts[run_id] = (n, src.count_rows(), len(matches))
    print(f"{run_id}: {n:,} rows from {table} ({src.count_rows():,}) over {len(matches)} matches "
          f"{'OK' if n == src.count_rows() else 'MISMATCH'}", flush=True)
out.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)  # one version, not one per chunk
db.create_table("runs", pa.Table.from_pylist([
    {"run_id": r, "source_table": tb, "what": w, "role": ro, "rows": counts[r][0], "matches": counts[r][2]}
    for r, tb, w, ro in RUNS]), mode="overwrite")
d = db.open_table("detections")
print(f"detections: {d.count_rows():,} rows; positions missing (no layout arena): "
      f"{d.count_rows('x IS NULL'):,}; non-unit classes (card null): {d.count_rows('card IS NULL'):,}")
