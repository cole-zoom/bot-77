"""Process one video end to end: matches -> per-frame HUD state -> play events -> LanceDB."""

from __future__ import annotations

import bisect
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import lancedb
import numpy as np
import pyarrow as pa

from bot77.events import FrameState, Play, detect_plays
from bot77.layout import Layout
from bot77.readers.ability import read_ability
from bot77.readers.clock import ClockRead, ocr_text, read_clock
from bot77.readers.elixir import read_elixir
from bot77.readers.hand import SLOTS, HandReader, infer_deck
from bot77.segment import segment_matches
from bot77.video import iter_frames, probe

HUD_FPS = 10
CLOCK_FPS = 1
DECK_SAMPLES = 12
THUMB_W = 360
PLAY_THUMB_DELAY_S = 1.0  # the arena a second after the drop shows what was placed

VIDEOS = pa.schema([
    ("video_id", pa.string()), ("creator", pa.string()), ("layout_id", pa.string()), ("mode", pa.string()),
    ("path", pa.string()), ("url", pa.string()), ("width", pa.int32()), ("height", pa.int32()),
    ("duration_s", pa.float32()), ("processed_at", pa.string()),
])
MATCHES = pa.schema([
    ("match_id", pa.string()), ("video_id", pa.string()), ("index", pa.int32()),
    ("t_start", pa.float32()), ("t_end", pa.float32()), ("end_phase", pa.string()), ("end_time_left_s", pa.int32()),
    ("opponent_name", pa.string()), ("opponent_rating", pa.int32()),
    ("deck", pa.list_(pa.int64())), ("deck_names", pa.list_(pa.string())), ("card_snapshot", pa.string()),
])
HUD = pa.schema([
    ("match_id", pa.string()), ("t_video", pa.float32()), ("t_match", pa.float32()),
    ("hand", pa.list_(pa.int64())), ("hand_form", pa.list_(pa.string())), ("hand_state", pa.list_(pa.string())),
    ("greyed", pa.list_(pa.bool_())), ("selected", pa.list_(pa.bool_())),
    ("next", pa.int64()), ("elixir", pa.float32()), ("multiplier", pa.int8()),
    ("ability", pa.list_(pa.string())),  # left, right button: ready | dark | absent
])
EVENTS = pa.schema([
    ("event_id", pa.string()), ("match_id", pa.string()), ("video_id", pa.string()),
    ("t_video", pa.float32()), ("t_match", pa.float32()), ("kind", pa.string()),
    ("card_id", pa.int64()), ("card", pa.string()), ("form", pa.string()), ("slot", pa.string()),
    ("t_drag", pa.float32()),
    ("elixir_before", pa.float32()), ("elixir_after", pa.float32()), ("measured_cost", pa.float32()),
    ("confidence", pa.string()), ("notes", pa.list_(pa.string())),
    ("thumb_hand", pa.binary()), ("thumb_arena", pa.binary()),
])


def _jpeg(img: np.ndarray, width: int = THUMB_W) -> bytes:
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def _bounds(layout: Layout, names: list[str], frame: np.ndarray, pad: int = 10) -> tuple[slice, slice]:
    boxes = [layout.box(n, frame.shape[1], frame.shape[0]) for n in names]
    x0, y0 = min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad
    x1, y1 = max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad
    return slice(max(y0, 0), y1), slice(max(x0, 0), x1)


def _arena(layout: Layout, frame: np.ndarray) -> np.ndarray:
    """Everything between the top HUD and the hand bar."""
    ys, xs = _bounds(layout, ["opp_left_hp", "opp_right_hp", "my_left_hp", "my_right_hp"], frame, pad=0)
    hand_top = layout.box("slot1_tab", frame.shape[1], frame.shape[0])[1]
    x_pad = (xs.stop - xs.start) // 3
    return frame[max(ys.start - 60, 0):hand_top, max(xs.start - x_pad, 0):xs.stop + x_pad]


def _parse_int(text: str) -> int | None:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def process_video(video: Path, layout_id: str, cards: list[dict], db_dir: Path, *, creator: str = "",
                  mode: str = "ladder", url: str = "", card_snapshot: str = "", log=print) -> dict:
    layout = Layout.load(layout_id)
    info = probe(video)
    video_id = video.stem
    full = HandReader.from_cards(layout, cards)
    by_id = {c["id"]: c for c in cards}
    costs = {cid: c["elixir"] for cid, c in by_id.items() if c["elixir"] is not None}
    names = {cid: c["name"] for cid, c in by_id.items()}

    # 1. Clock pass -> matches, plus opponent info sampled along the way.
    t0 = time.time()
    clock: list[tuple[float, ClockRead]] = []
    opp: list[tuple[float, str, str]] = []
    for t, f in iter_frames(video, CLOCK_FPS):
        clock.append((t, read_clock(f, layout, mode)))
        if int(t) % 15 == 0:
            opp.append((t, ocr_text(layout.crop(f, "opp_name"), 1),
                        ocr_text(layout.crop(f, "opp_rating"), 1) if "opp_rating" in layout.regions else ""))
    matches = segment_matches(clock)
    log(f"clock pass: {len(clock)} frames, {len(matches)} matches ({time.time() - t0:.0f}s)")

    clock_t = [t for t, _ in clock]

    def multiplier_at(t: float) -> int:
        i = bisect.bisect_right(clock_t, t) - 1
        for j in (i, i + 1, i - 1):
            if 0 <= j < len(clock) and clock[j][1].multiplier:
                return clock[j][1].multiplier
        return 1

    match_rows, hud_rows, event_rows = [], [], []
    for mi, m in enumerate(matches):
        match_id = f"{video_id}_m{mi + 1:02d}"
        # Matches overlap by a few seconds when end screens are cut out: start after the previous one.
        prev_end = matches[mi - 1].t_end if mi else 0.0
        t_first, t_last = max(m.t_start, prev_end, 0.0), m.t_end

        # 2. Deck from a spread of frames with the full reader.
        span = t_last - t_first
        sample = [full.read(f) for _, f in iter_frames(video, DECK_SAMPLES / span, start=t_first, duration=span)]
        deck = infer_deck(sample)
        reader = full.restrict(set(deck))

        # 3. HUD pass at 10 fps.
        t0 = time.time()
        states: list[FrameState] = []
        hand_crops: dict[int, np.ndarray] = {}
        for t, f in iter_frames(video, HUD_FPS, start=t_first, duration=span):
            hand = reader.read(f)
            buttons = (read_ability(f, layout, "ability_left"), read_ability(f, layout, "ability_right"))
            states.append(FrameState(t, hand, read_elixir(f, layout), multiplier_at(t), buttons))
            hud_rows.append(_hud_row(match_id, t, m.elapsed_at(t), hand, states[-1]))
        log(f"match {mi + 1}: {len(states)} frames ({time.time() - t0:.0f}s), deck {[names[c] for c in deck]}")

        # 4. Plays.
        champs, uses = _ability_cards(deck, by_id, states)
        cuts = [t for (pt, pr), (t, r) in zip(m.reads, m.reads[1:]) if r.elapsed_s - pr.elapsed_s > (t - pt) + 3]
        plays = detect_plays(states, costs, names, champs, uses, cuts)
        thumbs = _thumbnails(video, layout, plays)
        for k, p in enumerate(plays):
            event_rows.append(_event_row(f"{match_id}_e{k + 1:03d}", match_id, video_id, m.elapsed_at(p.t), p, thumbs.get(k)))

        opp_here = [o for o in opp if m.t_start <= o[0] <= m.t_end]
        name, rating = (opp_here[len(opp_here) // 2][1:] if opp_here else ("", ""))
        match_rows.append({
            "match_id": match_id, "video_id": video_id, "index": mi + 1, "t_start": m.t_start, "t_end": m.t_end,
            "end_phase": m.last.phase, "end_time_left_s": m.last.time_left_s, "opponent_name": name,
            "opponent_rating": _parse_int(rating), "deck": deck, "deck_names": [names[c] for c in deck],
            "card_snapshot": card_snapshot,
        })

    db = lancedb.connect(db_dir)
    _replace(db, "videos", VIDEOS, [{
        "video_id": video_id, "creator": creator, "layout_id": layout_id, "mode": mode, "path": str(video),
        "url": url or f"https://www.youtube.com/watch?v={video_id}", "width": info["width"], "height": info["height"],
        "duration_s": info["duration"], "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }], video_id)
    _replace(db, "matches", MATCHES, match_rows, video_id)
    _replace(db, "hud_states", HUD, hud_rows, video_id, key="match_id", prefix=True)
    _replace(db, "events", EVENTS, event_rows, video_id)
    return {"matches": len(match_rows), "events": len(event_rows), "hud_states": len(hud_rows)}


def redetect_video(video_id: str, cards: list[dict], db_dir: Path, log=print) -> dict:
    """Re-run play detection from the stored per-frame HUD states (no video decoding, apart from
    thumbnails). Use after changing the detector."""
    from bot77.readers.elixir import ElixirRead
    from bot77.readers.hand import SlotRead

    db = lancedb.connect(db_dir)
    video = db.open_table("videos").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist()[0]
    layout = Layout.load(video["layout_id"])
    by_id = {c["id"]: c for c in cards}
    costs = {cid: c["elixir"] for cid, c in by_id.items() if c["elixir"] is not None}
    names = {cid: c["name"] for cid, c in by_id.items()}
    matches = sorted(db.open_table("matches").search().where(f"video_id = '{video_id}'").to_arrow().to_pylist(),
                     key=lambda m: m["index"])
    hud = db.open_table("hud_states")
    event_rows = []
    for mi, m in enumerate(matches):
        prev_end = matches[mi - 1]["t_end"] if mi else 0.0
        t_first = max(m["t_start"], prev_end, 0.0)
        rows = hud.search().where(f"match_id = '{m['match_id']}' AND t_video >= {t_first}").limit(1_000_000) \
            .to_arrow().to_pylist()
        rows.sort(key=lambda r: r["t_video"])
        states = []
        for r in rows:
            hand = {}
            for k, slot in enumerate(SLOTS):
                cid, st = r["hand"][k], r["hand_state"][k]
                hand[slot] = SlotRead(st, cid if cid >= 0 else None, names.get(cid), r["hand_form"][k] or None,
                                      greyed=r["greyed"][k], selected=r["selected"][k])
            hand["next"] = SlotRead("card", r["next"], names.get(r["next"]), "normal") if r["next"] >= 0 else SlotRead("unknown")
            el = r["elixir"]
            buttons = tuple(r.get("ability") or ("absent", "absent"))
            states.append(FrameState(r["t_video"], hand, ElixirRead(el, int(el) if el is not None else None, True, 1.0),
                                     r["multiplier"], buttons))
        champs, uses = _ability_cards(m["deck"], by_id, states)
        cuts = [b["t_video"] for a, b in zip(rows, rows[1:])
                if b["t_match"] - a["t_match"] > (b["t_video"] - a["t_video"]) + 2]
        plays = detect_plays(states, costs, names, champs, uses, cuts)
        thumbs = _thumbnails(Path(video["path"]), layout, plays)
        hud_t = [r["t_video"] for r in rows]
        for k, p in enumerate(plays):
            j = min(max(bisect.bisect_left(hud_t, p.t), 0), len(rows) - 1)  # match time from the stored clock mapping
            event_rows.append(_event_row(f"{m['match_id']}_e{k + 1:03d}", m["match_id"], video_id, rows[j]["t_match"], p, thumbs.get(k)))
        log(f"{m['match_id']}: {len(plays)} events")
    events = db.open_table("events")
    if "t_drag" not in events.schema.names:  # schema changed: rebuild the table
        keep = events.search().where(f"video_id != '{video_id}'").limit(10_000_000).to_arrow().to_pylist()
        for r in keep:
            r.setdefault("t_drag", None)
        db.create_table("events", pa.Table.from_pylist(keep + event_rows, schema=EVENTS), mode="overwrite")
    else:
        _replace(db, "events", EVENTS, event_rows, video_id)
    return {"events": len(event_rows)}


def _ability_cards(deck: list[int], by_id: dict, states: list[FrameState]) -> tuple[dict, dict]:
    """Cards whose ability button can appear: champions always, hero-capable cards only if they
    were actually played in hero form this match (e.g. Barbarian Barrel has a hero form, but
    a normal Barbarian Barrel has no ability)."""
    hero_seen = {s.hand[slot].card_id for s in states for slot in SLOTS if s.hand[slot].form == "hero"}
    costs = {cid: by_id[cid]["ability_cost"] for cid in deck
             if by_id[cid]["ability_cost"] is not None and (by_id[cid]["is_champion"] or cid in hero_seen)}
    uses = {cid: by_id[cid]["ability_uses_per_deploy"] or 1 for cid in costs}
    return costs, uses


def _hud_row(match_id: str, t: float, t_match: float, hand: dict, st: FrameState) -> dict:
    r = [hand[s] for s in SLOTS]
    return {
        "match_id": match_id, "t_video": t, "t_match": t_match,
        "hand": [x.card_id if x.state == "card" else -1 for x in r],
        "hand_form": [x.form or "" for x in r], "hand_state": [x.state for x in r],
        "greyed": [bool(x.greyed) for x in r], "selected": [bool(x.selected) for x in r],
        "next": hand["next"].card_id if hand["next"].state == "card" else -1,
        "elixir": st.elixir.elixir, "multiplier": st.multiplier, "ability": list(st.abilities),
    }


def _event_row(event_id: str, match_id: str, video_id: str, t_match: float, p: Play, thumbs) -> dict:
    d = asdict(p)
    return {
        "event_id": event_id, "match_id": match_id, "video_id": video_id, "t_video": p.t, "t_match": t_match,
        "t_drag": p.t_drag,
        "kind": p.kind, "card_id": p.card_id, "card": p.name, "form": p.form, "slot": p.slot,
        "elixir_before": p.elixir_before, "elixir_after": p.elixir_after, "measured_cost": p.measured_cost,
        "confidence": p.confidence, "notes": d["notes"],
        "thumb_hand": thumbs[0] if thumbs else None, "thumb_arena": thumbs[1] if thumbs else None,
    }


def _thumbnails(video: Path, layout: Layout, plays: list[Play]) -> dict[int, tuple[bytes, bytes]]:
    """Hand bar just before each drop, and the arena a second after it."""
    from bot77.calibrate import read_frame

    out = {}
    hand_regions = ["next", *SLOTS, "elixir_bar"]
    for k, p in enumerate(plays):
        before = read_frame(video, max(p.t - 0.3, 0))
        after = read_frame(video, p.t + PLAY_THUMB_DELAY_S)
        ys, xs = _bounds(layout, hand_regions, before)
        out[k] = (_jpeg(before[ys, xs]), _jpeg(_arena(layout, after)))
    return out


def _replace(db, name: str, schema: pa.Schema, rows: list[dict], video_id: str, key: str = "video_id",
             prefix: bool = False) -> None:
    """Replace this video's rows in a table (creating it if needed)."""
    table = pa.Table.from_pylist(rows, schema=schema)
    if name not in db.table_names():
        db.create_table(name, table)
        return
    t = db.open_table(name)
    if set(t.schema.names) != set(schema.names):  # schema changed: rebuild, keeping other videos' rows
        where_other = f"NOT ({key} LIKE '{video_id}%')" if prefix else f"{key} != '{video_id}'"
        keep = t.search().where(where_other).limit(100_000_000).to_arrow().to_pylist()
        for r in keep:
            for col in schema.names:
                r.setdefault(col, None)
        db.create_table(name, pa.Table.from_pylist(keep + rows, schema=schema), mode="overwrite")
        return
    where = f"{key} LIKE '{video_id}%'" if prefix else f"{key} = '{video_id}'"
    t.delete(where)
    if rows:
        t.add(table)
