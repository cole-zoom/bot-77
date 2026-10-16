"""Opponent plays v1: deploy events from stored unit detections, with the opponent's deck
inferred per match, written to LanceDB with thumbnails for review.

See docs/opponent-plays-spec.md. Built on KataCR's detectors [Wu et al. 2025].
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import lancedb
import pyarrow as pa

from bot77.calibrate import read_frame
from bot77.layout import Layout
from bot77.pipeline import _jpeg, _replace
from bot77.spawns import apply_deck, deploys, drop_pov_units, infer_deck, resolve_composites

CHAMPIONS = {"Archer Queen", "Golden Knight", "Skeleton King", "Mighty Miner", "Monk", "Little Prince", "Goblinstein",
             "Boss Bandit"}

SPELL_TEAM = os.environ.get("BOT77_SPELL_TEAM", "enemy") or None  # "" = old screen-half rule (dev A/B)

THUMB_DELAY_S = 0.5


def _split_dets(rules):
    if rules is None:
        return 0
    from bot77.playrules import merged_rules
    return merged_rules(rules)["split_min_dets"]

SCHEMA = pa.schema([
    ("event_id", pa.string()), ("match_id", pa.string()), ("video_id", pa.string()),
    ("t_video", pa.float32()), ("t_match", pa.float32()), ("card", pa.string()), ("evo", pa.bool_()),
    ("t_seen", pa.float32()),  # first unit seen (t_video = estimated moment the card left the hand)
    ("units", pa.int32()), ("x_tiles", pa.float32()), ("y_tiles", pa.float32()),
    ("votes", pa.string()), ("classes", pa.list_(pa.string())), ("subunit_only", pa.bool_()),
    ("deck", pa.list_(pa.string())), ("thumb_zoom", pa.binary()), ("thumb_arena", pa.binary()),
])


def opponent_events(video_id: str, db_dir: Path, matches: list[int] | None = None, log=print,
                    det_table: str = "unit_detections", out_table: str = "opponent_events", decode_params=None, thumbs: bool = True,
                    rules: dict | None = None) -> int:
    """decode_params: a bot77.decode.DecodeParams to pick plays with the game-rules decoder (None: v1 behaviour)."""
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
        ev = deploys(dets, names, tile, lambda c: ctype.get(c) == "building", lambda c: ctype.get(c) == "spell", (ax0, ay0),
                     spell_team=SPELL_TEAM,
                     split_min_dets=_split_dets(rules))
        pov_all = [(p["t_video"], p["card"]) for p in db.open_table("events").search()
                   .where(f"match_id = '{m['match_id']}' AND kind = 'play'").limit(10**5).select(["t_video", "card"])
                   .to_arrow().to_pylist()]
        pov = [(p["t_video"], p["card"]) for p in db.open_table("events").search()
               .where(f"match_id = '{m['match_id']}'").limit(10**5).select(["t_video", "card"]).to_arrow().to_pylist()
               if p["card"]]
        enemy, dropped = drop_pov_units(resolve_composites([e for e in ev if e.team == "enemy"]), pov)
        abil = []
        if rules is not None:  # play rules v2 (bot77.playrules); {} = defaults
            from bot77.playrules import apply_rules, merged_rules, replay_rules
            if video["layout_id"].startswith("replay"):
                rules = replay_rules(rules)
            cr = merged_rules(rules)["clock"]
            if cr["on"]:
                from bot77.clocks import fuse_clocks
                enemy = fuse_clocks(enemy, dets, names, tile, (ax0, ay0), pov, lambda c: ctype.get(c) == "spell", cr)
            enemy = apply_rules(enemy, pov, ctype, rules)
            mr = merged_rules(rules)
            if mr["spell_probs"]:  # spell stage (bot77.spells): classifier spell plays replace/join the detector's
                from bot77.spells import load_probs, merge_spells, spell_peaks
                pr = load_probs(mr["spell_probs"], m["match_id"])
                if pr is not None:
                    peaks = spell_peaks(*pr, thr=mr["spell_thr"], group_s=mr["spell_group_s"],
                                        class_thr=mr["spell_class_thr"])
                    own_ab = [(p["t_video"], p["card"]) for p in db.open_table("events").search()
                              .where(f"match_id = '{m['match_id']}' AND kind = 'ability'").limit(10**4)
                              .select(["t_video", "card"]).to_arrow().to_pylist() if p["card"]]
                    enemy = merge_spells(enemy, peaks, pov, lambda c: ctype.get(c) == "spell", any_own=mr["spell_any_own"],
                                         own_abilities=own_ab, ability_thr=mr["spell_ability_thr"],
                                         unnamed_own=[t for t, c in pov_all if not c] if mr["spell_unnamed_own"] else ())
        # spell-stage ability plays stay out of deck inference and card re-picking; they rejoin after
        abil = [e for e in enemy if e.card == "__ability" and "spell-stage" in e.classes]
        enemy = [e for e in enemy if not (e.card == "__ability" and "spell-stage" in e.classes)]
        deck = infer_deck(enemy, w_unclocked=merged_rules(rules)["deck_w_unclocked"] if rules is not None else 1.0,
                          is_spell=lambda c: ctype.get(c) == "spell")
        if decode_params is not None:
            from bot77.decode import decode
            enemy = decode(enemy, deck, lambda c: ctype.get(c) == "spell", decode_params, mirror_in_deck=True)
        else:
            apply_deck(enemy, set(deck))
            if rules is not None and merged_rules(rules)["cycle_guess"]:
                # a clock with no recognisable unit: guess from the cycle (cards in hand = deck minus the last 4 plays),
                # the one that has waited longest
                hist = []
                for e in sorted(enemy, key=lambda e: e.t):
                    if e.card is None and getattr(e, "from_clock", False):
                        cand = [c for c in deck if ctype.get(c) != "spell" and c not in [h for _, h in hist[-4:]]]
                        last = {c: max([t for t, h in hist if h == c], default=-1e9) for c in cand}
                        e.card = min(cand, key=lambda c: last[c]) if cand else None
                    if e.card:
                        hist.append((e.t, e.card))
            if rules is not None and merged_rules(rules)["drop_cardless"]:
                enemy = [e for e in enemy if e.card is not None]
        if rules is not None and merged_rules(rules)["mirror_repeat"]:
            # Mirror copies the card just played, so it is always the very next play (Cole, 2026-10-15); and a card
            # can't come back by itself sooner than 4 plays later (8-card cycle). Same card twice in a row = Mirror.
            # Only when both reads are sure: a misread first card makes a false repeat (held-back replays, 2026-10-15).
            mr_ = merged_rules(rules)
            sure = lambda e: (sum(e.card_votes.values()) >= mr_["mirror_min_votes"] and
                              e.card_votes.get(e.card, 0) >= mr_["mirror_min_share"] * sum(e.card_votes.values()))
            prev = None
            for e in sorted(enemy, key=lambda e: e.t):
                if (prev is not None and e.card and e.card == prev.card and e.card not in CHAMPIONS  # can't copy a champion
                        and sure(e) and sure(prev)):
                    e.card = "Mirror"
                prev = e
        enemy = sorted(enemy + abil, key=lambda e: e.t)
        clock = sorted(hud.search().where(f"match_id = '{m['match_id']}'").limit(10**7).select(["t_video", "t_match"])
                       .to_arrow().to_pylist(), key=lambda r: r["t_video"])
        for k, e in enumerate(enemy):
            t_match = next((c["t_match"] for c in clock if c["t_video"] >= e.t), None)
            if not thumbs:  # development-set scoring: seeking the video for every play is most of the runtime
                rows.append({"event_id": f"{m['match_id']}_o{k + 1:03d}", "match_id": m["match_id"], "video_id": video_id,
                             "t_video": e.t, "t_match": t_match, "card": e.card, "evo": e.evo, "units": e.units,
                             "t_seen": getattr(e, "t_seen", e.t), "x_tiles": e.x, "y_tiles": e.y, "votes": json.dumps(e.card_votes), "classes": e.classes,
                             "subunit_only": e.subunit_only, "deck": deck, "thumb_zoom": None, "thumb_arena": None})
                continue
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
                "t_seen": getattr(e, "t_seen", e.t), "x_tiles": e.x, "y_tiles": e.y, "votes": json.dumps(e.card_votes), "classes": e.classes,
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
