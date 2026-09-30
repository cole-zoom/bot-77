"""Opponent plays v1: deploy events from stored unit detections, with the opponent's deck
inferred per match, written to LanceDB with thumbnails for review.

See docs/opponent-plays-spec.md. Built on KataCR's detectors [Wu et al. 2025].
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import lancedb
import pyarrow as pa

from bot77.calibrate import read_frame
from bot77.layout import Layout
from bot77.pipeline import _jpeg, _replace
from bot77.spawns import apply_deck, deploys, drop_pov_units, infer_deck, resolve_composites

THUMB_DELAY_S = 0.5

SCHEMA = pa.schema([
    ("event_id", pa.string()), ("match_id", pa.string()), ("video_id", pa.string()),
    ("t_video", pa.float32()), ("t_match", pa.float32()), ("card", pa.string()), ("evo", pa.bool_()),
    ("units", pa.int32()), ("x_tiles", pa.float32()), ("y_tiles", pa.float32()),
    ("votes", pa.string()), ("classes", pa.list_(pa.string())), ("subunit_only", pa.bool_()),
    ("deck", pa.list_(pa.string())), ("thumb_zoom", pa.binary()), ("thumb_arena", pa.binary()),
])


def opponent_events(video_id: str, db_dir: Path, matches: list[int] | None = None, log=print,
                    det_table: str = "unit_detections", out_table: str = "opponent_events") -> int:
    db = lancedb.connect(db_dir)
    video = db.open_table("videos").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist()[0]
    layout = Layout.load(video["layout_id"])
    cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
    names = [c["name"] for c in cards]
    ctype = {c["name"]: c["type"] for c in cards}
    ax0, ay0, ax1, ay1 = layout.katacr_arena
    tile = (ax1 - ax0) / 18
    ms = sorted(db.open_table("matches").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist(),
                key=lambda m: m["index"])
    hud = db.open_table("hud_states")
    rows = []
    for m in ms:
        if matches and m["index"] not in matches:
            continue
        dets = db.open_table(det_table).search().where(f"match_id = '{m['match_id']}'").limit(10**7).to_arrow().to_pylist()
        if not dets:
            continue
        ev = deploys(dets, names, tile, lambda c: ctype.get(c) == "building", lambda c: ctype.get(c) == "spell", (ax0, ay0))
        pov = [(p["t_video"], p["card"]) for p in db.open_table("events").search()
               .where(f"match_id = '{m['match_id']}'").limit(10**5).select(["t_video", "card"]).to_arrow().to_pylist()
               if p["card"]]
        enemy, dropped = drop_pov_units(resolve_composites([e for e in ev if e.team == "enemy"]), pov)
        deck = infer_deck(enemy)
        apply_deck(enemy, set(deck))
        clock = sorted(hud.search().where(f"match_id = '{m['match_id']}'").limit(10**7).select(["t_video", "t_match"])
                       .to_arrow().to_pylist(), key=lambda r: r["t_video"])
        for k, e in enumerate(enemy):
            f = read_frame(Path(video["path"]), e.t + THUMB_DELAY_S)
            px, py = ax0 + e.x * tile, ay0 + e.y * tile
            s = layout.scale(f.shape[1])
            r = int(2.5 * tile * s)
            cx, cy = int(px * s), int(py * s)
            zoom = f[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r].copy()
            arena = f[max(int(ay0 * s), 0):int(ay1 * s), max(int(ax0 * s), 0):int(ax1 * s)].copy()
            cv2.circle(arena, (cx - max(int(ax0 * s), 0), cy - max(int(ay0 * s), 0)), int(1.3 * tile * s), (0, 0, 255), 4)
            t_match = next((c["t_match"] for c in clock if c["t_video"] >= e.t), None)
            rows.append({
                "event_id": f"{m['match_id']}_o{k + 1:03d}", "match_id": m["match_id"], "video_id": video_id,
                "t_video": e.t, "t_match": t_match, "card": e.card, "evo": e.evo, "units": e.units,
                "x_tiles": e.x, "y_tiles": e.y, "votes": json.dumps(e.card_votes), "classes": e.classes,
                "subunit_only": e.subunit_only, "deck": deck,
                "thumb_zoom": _jpeg(zoom, 200) if zoom.size else None, "thumb_arena": _jpeg(arena, 240),
            })
        log(f"{m['match_id']}: {len(enemy)} opponent deploys ({len(dropped)} dropped as our own units); inferred deck {deck}")
    if rows:
        ran = {r["match_id"] for r in rows}
        if out_table in db.table_names(limit=10_000):
            t = db.open_table(out_table)
            t.delete(" OR ".join(f"match_id = '{mid}'" for mid in ran))
            t.add(pa.Table.from_pylist(rows, schema=SCHEMA))
        else:
            db.create_table(out_table, pa.Table.from_pylist(rows, schema=SCHEMA))
    return len(rows)
