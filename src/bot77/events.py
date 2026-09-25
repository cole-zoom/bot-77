"""Turn per-frame hand + elixir reads into play events.

What a play looks like on screen (measured on Alina's footage):
1. The card leaves its slot when the player starts dragging it (slot shows empty).
2. Elixir drops by the card's cost when it's released on the arena, typically ~0.2-1.5 s
   later. The drop animates over ~0.3 s.
3. The next card advances and slides into the vacated slot.
A cancelled drag is step 1 followed by the same card coming back, with no elixir drop.

So the elixir drop is the moment of the play, and the card is the one that recently left
the hand whose cost matches the drop. A drop with no card leaving the hand is likely a
champion / hero ability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bot77.readers.elixir import ElixirRead
from bot77.readers.hand import SLOTS, SlotRead

REGEN_S_PER_ELIXIR = 2.8  # at x1; x2 and x3 divide this
DROP_MIN = 0.6  # elixir fall that counts as spending
COST_TOLERANCE = 0.4  # |measured drop - card cost| allowed
VACATE_LOOKBACK_S = 4.0  # a card must have left the hand within this long before the drop
NEXT_ADVANCE_S = 2.0  # the next card should change within this long after the drop


@dataclass(frozen=True)
class FrameState:
    t: float  # video time
    hand: dict[str, SlotRead]
    elixir: ElixirRead
    multiplier: int = 1


@dataclass
class Play:
    t: float  # video time of the elixir drop
    kind: str  # play | ability
    card_id: int | None
    name: str | None
    form: str | None
    slot: str | None
    elixir_before: float
    elixir_after: float
    measured_cost: float
    confidence: str  # high | medium | low
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- smoothing


def _slot_series(states: list[FrameState], slot: str) -> list[tuple[int, str] | None]:
    """Per frame: (card_id, form) for a card, None for empty. Unknown frames (occlusion,
    mid-slide) hold the previous value. One-frame flicker between two reads of the same
    card is removed; a card that shows for a single frame between empties is kept (it slid
    in and was dragged straight back out)."""
    raw: list[tuple[int, str] | None | str] = []
    for s in states:
        r = s.hand[slot]
        raw.append((r.card_id, r.form) if r.state == "card" else None if r.state == "empty" else "?")
    held, last = [], None
    for v in raw:
        last = last if v == "?" else v
        held.append(last)
    out = list(held)
    for i in range(1, len(held) - 1):
        if held[i - 1] == held[i + 1] != held[i] and held[i - 1] is not None:
            out[i] = held[i - 1]
    return out


RISE_SLACK = 0.15  # elixir allowed above the regen-limited rise, for read noise
RISE_CONFIRM_S = 0.5  # a faster rise that holds this long is real (e.g. an elixir gain)


def _elixir_series(states: list[FrameState]) -> np.ndarray:
    e = np.array([np.nan if s.elixir.elixir is None else s.elixir.elixir for s in states], float)
    idx = np.arange(len(e))
    ok = ~np.isnan(e)
    if ok.sum() >= 2:
        e = np.interp(idx, idx[ok], e[ok])
    # median of 3 removes single-frame glitches (overlays on the bar)
    padded = np.pad(e, 1, mode="edge")
    e = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)

    # Elixir can't rise faster than regen. When a whole point ticks over, the bar briefly
    # reads ~0.9 high (new pink segment + old sliver); capping the rise removes those spikes,
    # which would otherwise look like a 1-elixir spend when they fall back.
    t = np.array([s.t for s in states])
    out = e.copy()
    above_since = None
    for i in range(1, len(e)):
        cap = out[i - 1] + (t[i] - t[i - 1]) * states[i].multiplier / REGEN_S_PER_ELIXIR + RISE_SLACK
        if e[i] > cap:
            above_since = t[i] if above_since is None else above_since
            if t[i] - above_since < RISE_CONFIRM_S:
                out[i] = cap
                continue
        else:
            above_since = None
        out[i] = min(e[i], 10.0)
    return out


# ---------------------------------------------------------------- drops


@dataclass
class _Drop:
    i_start: int  # last frame before elixir starts falling
    i_end: int  # frame where the fall bottoms out
    amount: float  # regen-corrected elixir spent


DROP_PAUSE_S = 0.35  # the drop animation pauses briefly at whole numbers; bridge pauses up to this
DROP_MAX_S = 1.2


def find_drops(states: list[FrameState], e: np.ndarray) -> list[_Drop]:
    """A drop runs from the last frame before the bar starts falling to the last frame that
    fell, bridging short pauses. Regen is added back over that span only, so frames where the
    bar sits flat afterwards (e.g. pinned at 0) don't inflate the measured cost."""
    t = np.array([s.t for s in states])
    drops, i, n = [], 1, len(e)
    while i < n:
        if e[i] < e[i - 1] - 0.05:
            start, last_fall, j = i - 1, i, i
            while j + 1 < n and t[j + 1] - t[start] < DROP_MAX_S:
                if e[j + 1] < e[j] - 0.01:
                    last_fall = j + 1
                elif e[j + 1] > e[j] + 0.03 or t[j + 1] - t[last_fall] > DROP_PAUSE_S:
                    break
                j += 1
            regen = (t[last_fall] - t[start]) * states[start].multiplier / REGEN_S_PER_ELIXIR
            amount = e[start] - e[last_fall] + regen
            if amount >= DROP_MIN:
                drops.append(_Drop(start, last_fall, float(amount)))
            i = last_fall + 1
        else:
            i += 1
    return drops


# ---------------------------------------------------------------- attribution


def _vacated(series: list, i_drop: int, times: list[float], lookback_s: float):
    """If the slot is empty at frame i_drop and held a card within `lookback_s` before,
    return (card, frame index it was last seen)."""
    if series[i_drop] is not None:
        return None
    j = i_drop - 1
    while j >= 0 and times[i_drop] - times[j] <= lookback_s:
        if series[j] is not None:
            return series[j], j
        j -= 1
    return None


def detect_plays(states: list[FrameState], costs: dict[int, int], names: dict[int, str],
                 ability_costs: dict[int, int] | None = None,
                 ability_uses: dict[int, int] | None = None) -> list[Play]:
    """`costs`: card_id -> elixir; `ability_costs`: card_id -> ability elixir for the deck's
    heroes / champions; `ability_uses`: card_id -> uses per deploy (default 1)."""
    ability_costs = ability_costs or {}
    ability_uses = ability_uses or {}
    uses_left: dict[int, int] = {}  # champion / hero -> ability uses left since its last deploy
    times = [s.t for s in states]
    e = _elixir_series(states)
    series = {slot: _slot_series(states, slot) for slot in SLOTS}
    nxt = [(s.hand["next"].card_id if s.hand["next"].state == "card" else None) for s in states]

    plays: list[Play] = []
    used: set[tuple[str, int]] = set()  # (slot, last-seen frame) already attributed
    for d in find_drops(states, e):
        cost = round(d.amount)
        cands = []
        for slot in SLOTS:
            v = _vacated(series[slot], d.i_end, times, VACATE_LOOKBACK_S)
            if v and (slot, v[1]) not in used:
                (card_id, form), last_seen = v
                cands.append((slot, card_id, form, last_seen))
        exact = [c for c in cands if abs(costs.get(c[1], -99) - d.amount) <= COST_TOLERANCE]
        base = dict(t=times[d.i_start + 1], elixir_before=round(float(e[d.i_start]), 2),
                    elixir_after=round(float(e[d.i_end]), 2), measured_cost=round(d.amount, 2))

        if exact:
            slot, card_id, form, last_seen = max(exact, key=lambda c: c[3])  # most recently vacated
            used.add((slot, last_seen))
            notes = []
            prev_next = nxt[max(last_seen, 0)]
            advanced = any(nxt[k] != prev_next for k in range(d.i_end, len(states))
                           if times[k] - times[d.i_end] <= NEXT_ADVANCE_S)
            refilled = prev_next is not None and any(
                series[slot][k] is not None and series[slot][k][0] == prev_next
                for k in range(d.i_end, len(states)) if times[k] - times[d.i_end] <= NEXT_ADVANCE_S + 1)
            if not advanced:
                notes.append("next card did not advance")
            if not refilled:
                notes.append("slot not refilled with the previous next card")
            confidence = "high" if advanced and refilled else "medium" if advanced or refilled else "low"
            plays.append(Play(kind="play", card_id=card_id, name=names.get(card_id), form=form, slot=slot,
                              confidence=confidence, notes=notes, **base))
            if card_id in ability_costs:
                uses_left[card_id] = ability_uses.get(card_id, 1)
            continue

        pair = _pair_match(cands, costs, d.amount)
        if pair:
            for slot, card_id, form, last_seen in pair:
                used.add((slot, last_seen))
                plays.append(Play(kind="play", card_id=card_id, name=names.get(card_id), form=form, slot=slot,
                                  confidence="medium", notes=["two cards released together; drop split by cost"], **base))
                if card_id in ability_costs:
                    uses_left[card_id] = ability_uses.get(card_id, 1)
            continue

        ability = [cid for cid, c in ability_costs.items() if abs(c - d.amount) <= COST_TOLERANCE]
        if not cands and ability:
            cid = ability[0]
            if uses_left.get(cid, 0) > 0:
                uses_left[cid] -= 1
                plays.append(Play(kind="ability", card_id=cid, name=names.get(cid), form=None, slot=None,
                                  confidence="medium", notes=["no card left the hand; cost matches ability"], **base))
            else:
                plays.append(Play(kind="ability", card_id=cid, name=names.get(cid), form=None, slot=None,
                                  confidence="low", notes=["cost matches ability, but no unused deploy of it"], **base))
            continue

        note = (f"drop of {d.amount:.2f} matches no recently vacated card"
                + (f" (vacated: {', '.join(names.get(c[1], '?') for c in cands)})" if cands else ""))
        plays.append(Play(kind="play", card_id=None, name=None, form=None, slot=None,
                          confidence="low", notes=[note], **base))
    return plays


def _pair_match(cands: list, costs: dict[int, int], amount: float):
    """Two vacated cards whose costs add up to the drop."""
    best = None
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            total = costs.get(cands[a][1], -99) + costs.get(cands[b][1], -99)
            err = abs(total - amount)
            if err <= COST_TOLERANCE and (best is None or err < best[0]):
                best = (err, [cands[a], cands[b]])
    return best[1] if best else None
