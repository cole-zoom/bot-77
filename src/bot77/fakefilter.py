"""Learned fake-play filter (failure mining, 2026-10-16).

Hand rules for fakes trade too many real plays (the elixir check alone: held-back -80 fakes for -45 right, because real
plays also come in tight pairs and detector votes don't say which of two is the second read). Instead, a small
gradient-boosted classifier learns P(fake) per logged opponent play from the answer keys: trained on the train-split
replays (scripts/train_fake_filter.py), threshold picked on held-back + dev, test scored once.

Features use only what the pipeline knows when it writes a play (same function for training rows and live events):
the play's own evidence (votes, their mix, units, delay from first unit to play time, spot), its card (cost, type,
in the inferred deck and how high), its neighbours (gaps, same card nearby, plays within 4 s), and the opponent's
elixir (simulated as in bot77.elixir, before this play).
"""
from __future__ import annotations

import json
import math

import numpy as np

from bot77.elixir import CAP, _rate_fn

TYPES = ["troop", "building", "spell", "ability", "mirror", "none"]
FEATURES = (["votes", "top_share", "n_cards_voted", "units", "subunit_only", "delay", "x", "y", "past_river", "cost"]
            + [f"type_{t}" for t in TYPES]
            + ["spell_stage", "evo", "in_deck", "deck_rank", "card_count", "dt_prev", "dt_next", "dt_same_prev",
               "dt_same_next", "n_within_4s", "elixir_before", "elixir_after", "t_match", "multiplier", "prev_cost",
               "own_next", "own_prev", "own_same_near"])
RIVER = 14.9


def as_row(e) -> dict:
    """A pipeline Deploy as an events-table row."""
    return {"t_video": e.t, "card": e.card, "votes": e.card_votes, "units": e.units, "t_seen": getattr(e, "t_seen", e.t),
            "x_tiles": e.x, "y_tiles": e.y, "classes": list(e.classes), "subunit_only": e.subunit_only,
            "evo": bool(getattr(e, "evo", False))}


def features(rows: list[dict], deck: list[str], timeline: list[tuple[float, float | None, int]],
             cards: dict[str, dict], own: list[tuple[float, str | None]] = ()) -> np.ndarray:
    """rows: one match's plays (events-table rows or as_row(Deploy)); timeline: [(t_video, t_match, multiplier)];
    cards: name -> {"type", "elixir", "ability_cost"}; own: our own logged plays [(t_video, card)] (POV: the card we
    were dragging shows a placement preview the spell stage can read as an opponent spell, 2026-10-18; replays: the
    bottom player's plays). Returns len(rows) x len(FEATURES)."""
    rows = sorted(rows, key=lambda r: r["t_video"])
    ts = [r["t_video"] for r in rows]
    tl = sorted(timeline)
    gain = _rate_fn([(t, m) for t, _, m in tl])
    start = 5.0 if tl and tl[0][1] is not None and tl[0][1] < 3 else CAP
    clock = lambda t: next(((tm, mu) for tv, tm, mu in tl if tv >= t), (tl[-1][1], tl[-1][2]) if tl else (None, 1))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["card"]] = counts.get(r["card"], 0) + 1
    out, elixir, prev_t, last_card = [], start, (tl[0][0] if tl else (ts[0] if ts else 0.0)), None
    for i, r in enumerate(rows):
        v = r["votes"] if isinstance(r["votes"], dict) else json.loads(r["votes"] or "{}")
        tot = float(sum(v.values())) or 0.0
        card = r["card"]
        if card == "__ability":
            typ = "ability"
            names = [c[: -len(" ability")] for c in r["classes"] or [] if c.endswith(" ability")]
            cost = (cards.get(names[0], {}).get("ability_cost") if names else None) or 1
        elif card == "Mirror":
            typ, cost = "mirror", (cards.get(last_card, {}).get("elixir") or 0) + 1 if last_card else 1
        elif card is None:
            typ, cost = "none", 0
        else:
            typ, cost = cards.get(card, {}).get("type", "troop"), cards.get(card, {}).get("elixir") or 0
        elixir = min(CAP, elixir + gain(prev_t, r["t_video"]))
        prev_t = r["t_video"]
        before = elixir
        elixir -= cost
        same = [ts[j] for j in range(len(rows)) if j != i and rows[j]["card"] == card]
        tm, mu = clock(r["t_video"])
        f = {"votes": tot, "top_share": (v.get(card, 0) / tot) if tot else 0.0, "n_cards_voted": len(v),
             "units": r["units"] or 0, "subunit_only": float(bool(r["subunit_only"])),
             "delay": (r["t_seen"] or r["t_video"]) - r["t_video"], "x": r["x_tiles"], "y": r["y_tiles"],
             "past_river": r["y_tiles"] - RIVER, "cost": cost,
             **{f"type_{t}": float(typ == t) for t in TYPES},
             "spell_stage": float("spell-stage" in (r["classes"] or [])), "evo": float(bool(r.get("evo"))),
             "in_deck": float(card in deck), "deck_rank": deck.index(card) if card in deck else 9,
             "card_count": counts.get(card, 0),
             "dt_prev": ts[i] - ts[i - 1] if i else 60.0, "dt_next": ts[i + 1] - ts[i] if i + 1 < len(ts) else 60.0,
             "dt_same_prev": min([ts[i] - t for t in same if t < ts[i]] or [60.0]),
             "dt_same_next": min([t - ts[i] for t in same if t > ts[i]] or [60.0]),
             "n_within_4s": sum(1 for j, t in enumerate(ts) if j != i and abs(t - ts[i]) <= 4),
             "elixir_before": before, "elixir_after": elixir, "t_match": tm if tm is not None else -1.0,
             "multiplier": mu or 1, "prev_cost": out[-1][FEATURES.index("cost")] if out else 0.0,
             "own_next": min([t - r["t_video"] for t, _ in own if t >= r["t_video"]] or [60.0]),
             "own_prev": min([r["t_video"] - t for t, _ in own if t < r["t_video"]] or [60.0]),
             "own_same_near": float(any(c == card and abs(t - r["t_video"]) <= 8 for t, c in own))}
        if elixir < 0:
            elixir = 0.0  # don't carry an impossible debt into the next plays
        if card not in ("__ability", "Mirror", None):
            last_card = card
        out.append([float(f[k]) if f[k] is not None and not (isinstance(f[k], float) and math.isnan(f[k])) else -1.0
                    for k in FEATURES])
    return np.array(out, dtype=np.float32).reshape(len(rows), len(FEATURES))


def order(rows: list[dict]) -> list[int]:
    """Indices that sort rows the way features() does."""
    return sorted(range(len(rows)), key=lambda k: rows[k]["t_video"])
