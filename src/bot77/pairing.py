"""Pair two recordings of the same friendly battle (both players recording) and turn each side's
POV plays into exact opponent labels for the other.

The match clock runs at real speed in every game, so ANY two overlapping matches line up at some
constant offset — the clock alone can't pair them (first attempt, 2026-09-29). The recordings' wall-clock
start times fix it: the same game must sit at offset = start_A - start_B (plus a little clock skew). Then every POV play in B is an
opponent play in A at t_A = t_B - offset (and vice versa). The POV detector is 100% on 490 labelled
plays (decisions.md), so these labels need no human review.
"""

from __future__ import annotations

import json
from pathlib import Path

import lancedb
import numpy as np

MAX_SPREAD_S = 1.5  # clock-derived offsets within a true pair agree to about this
MAX_SKEW_S = 20  # the two devices' wall clocks / recording starts may differ by this much
MIN_COMMON_S = 30  # the two matches must share at least this much match time


def _clock(db, match_id: str) -> dict[int, float]:
    """match second -> first video time it was on screen."""
    rows = db.open_table("hud_states").search().where(f"match_id = '{match_id}'").limit(10**7) \
        .select(["t_video", "t_match"]).to_arrow().to_pylist()
    out: dict[int, float] = {}
    for r in sorted(rows, key=lambda r: r["t_video"]):
        if r["t_match"] is not None:
            out.setdefault(int(r["t_match"]), r["t_video"])
    return out


def pair_matches(video_a: str, video_b: str, start_offset_s: float, db_dir: Path = Path("data/lancedb")) -> list[dict]:
    """`start_offset_s`: recording start of A minus start of B, in seconds of wall-clock time."""
    db = lancedb.connect(db_dir)
    ms = lambda v: db.open_table("matches").search().where(f"video_id = '{v}'").to_arrow().to_pylist()
    pairs = []
    for ma in ms(video_a):
        ca = _clock(db, ma["match_id"])
        best = None
        for mb in ms(video_b):
            cb = _clock(db, mb["match_id"])
            common = sorted(set(ca) & set(cb))
            if len(common) < MIN_COMMON_S:
                continue
            offs = np.array([cb[e] - ca[e] for e in common])
            med = float(np.median(offs))
            spread = float(np.median(np.abs(offs - med)))
            if (spread <= MAX_SPREAD_S and abs(med - start_offset_s) <= MAX_SKEW_S
                    and (best is None or abs(med - start_offset_s) < abs(best["offset_s"] - start_offset_s))):
                best = {"a": ma["match_id"], "b": mb["match_id"], "offset_s": med, "spread_s": spread, "common_s": len(common)}
        if best:
            pairs.append(best)
    return pairs


def opponent_truth_from_pair(pair: dict, db_dir: Path = Path("data/lancedb")) -> tuple[dict, dict]:
    """(truth for A's opponent, truth for B's opponent), in the opponent-truth format."""
    db = lancedb.connect(db_dir)

    def pov(mid):
        return db.open_table("events").search().where(f"match_id = '{mid}'").limit(10**5) \
            .select(["t_video", "kind", "card", "form", "confidence"]).to_arrow().to_pylist()

    def build(target_mid, source_mid, sign):
        vid = target_mid.rsplit("_m", 1)[0]
        items = []
        for e in pov(source_mid):
            label = "__ability" if e["kind"] == "ability" else e["card"]
            if not label:
                continue
            items.append({"match_id": target_mid, "t_video": round(e["t_video"] - sign * pair["offset_s"], 2),
                          "label": label, "form": e["form"], "source": f"POV of {source_mid}",
                          "source_confidence": e["confidence"]})
        items.sort(key=lambda x: x["t_video"])
        return {"video_id": vid, "side": "opponent", "matches": [target_mid], "items": items,
                "paired_with": source_mid, "offset_s": pair["offset_s"], "auto": True}

    return build(pair["a"], pair["b"], 1), build(pair["b"], pair["a"], -1)


def write_pair_truth(video_a: str, video_b: str, start_offset_s: float, out_dir: Path = Path("data/ground_truth"),
                     db_dir: Path = Path("data/lancedb"), log=print) -> list[dict]:
    pairs = pair_matches(video_a, video_b, start_offset_s, db_dir)
    for p in pairs:
        ta, tb = opponent_truth_from_pair(p, db_dir)
        for t in (ta, tb):
            mid = t["matches"][0]
            (out_dir / f"opponent_truth_auto_{mid}.json").write_text(json.dumps(t, indent=1))
        log(f"{p['a']} <-> {p['b']}: offset {p['offset_s']:.2f}s (spread {p['spread_s']:.2f}), "
            f"{p['common_s']} s in common, {len(ta['items'])} / {len(tb['items'])} opponent plays")
    return pairs
