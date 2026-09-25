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

Lessons from Cole's review of the first two matches:
- Right after a play the bar's fractional sliver vanishes and the spent segments drain
  pale, so measured costs can be off by up to ~0.7. When exactly one card left the hand,
  and its slot refills with the previous next card, trust the hand over the cost.
- That sliver snap on its own looks like a small spend (e.g. 3.64 -> 3.03). A real spend
  is a whole number of elixir, so it always lowers the whole-elixir count; a snap doesn't.
- The emote panel covers the elixir bar; elixir is ignored while the hand is covered.

Round 2 (matches 3-4): "Mighty Miner ability, then Hog" merges into one 5-elixir drop, and
at triple elixir regen is fast enough that an ability can leave the whole-elixir count
unchanged. So when the ability button is readable, ability uses come from the button
itself (ready -> dark), and their cost is taken out of any drop they overlap before cards
are matched.
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
MID_SLIDE_SETTLE_S = 1.0  # a card dragged mid-slide doesn't appear in a slot within this long after the drop
SMALL_UNEXPLAINED = 1.5
PAIR_TOLERANCE = 0.7  # two cards released together: two sources of measurement error
END_GUARD_S = 1.5  # the bar reads 0 when the game ends; ignore unexplained drops this close to the end
CUT_GUARD_S = 1.5  # elixir jumps at jump cuts in edited videos; ignore unexplained drops this close to one
MISMATCH_MIN_FRACTION = 0.5  # a cost-mismatch guess needs the drop to be at least this share of the card's cost
ABILITY_STATE_MIN_FRAMES = 3  # a button state must hold this long to count (drags flicker over it)
ABILITY_BUTTON_LEAD_S = 0.5  # the button can change this long before the elixir drop starts ...
ABILITY_BUTTON_LAG_S = 1.5  # ... or this long after it ends  # drops below this with no card leaving the hand and no usable ability are ignored


@dataclass(frozen=True)
class FrameState:
    t: float  # video time
    hand: dict[str, SlotRead]
    elixir: ElixirRead
    multiplier: int = 1
    abilities: tuple[str, str] = ("absent", "absent")  # left, right button: ready | dark | absent


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
    t_drag: float | None = None  # when the card left the hand (orders cards released together)


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


OCCLUDED_SLOTS = 3  # this many unreadable slots means something (e.g. the emote panel) covers the hand bar


def _occluded(s: FrameState) -> bool:
    return sum(s.hand[slot].state == "unknown" for slot in SLOTS) >= OCCLUDED_SLOTS


def _button_series(states: list[FrameState], side: int) -> list[str]:
    """Ability button state per frame, with states shorter than ABILITY_STATE_MIN_FRAMES
    (a dragged card passing over the button, the press animation) replaced by what was
    there before."""
    raw = [s.abilities[side] for s in states]
    out = list(raw)
    i = 0
    while i < len(raw):
        j = i
        while j < len(raw) and raw[j] == raw[i]:
            j += 1
        if j - i < ABILITY_STATE_MIN_FRAMES and i > 0:
            for k in range(i, j):
                out[k] = out[i - 1]
        i = j
    return out


def _elixir_series(states: list[FrameState]) -> np.ndarray:
    e = np.array([np.nan if s.elixir.elixir is None or _occluded(s) else s.elixir.elixir for s in states], float)
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
    pieces: list[tuple[int, int, float]] = field(default_factory=list)  # falls separated by pauses


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
            piece_start, pieces = start, []
            while j + 1 < n and t[j + 1] - t[start] < DROP_MAX_S:
                if e[j + 1] < e[j] - 0.01:
                    if j + 1 - last_fall > 1:  # falling resumed after a pause: close the piece
                        pieces.append((piece_start, last_fall))
                        piece_start = j
                    last_fall = j + 1
                elif e[j + 1] > e[j] + 0.03 or t[j + 1] - t[last_fall] > DROP_PAUSE_S:
                    break
                j += 1
            pieces.append((piece_start, last_fall))
            amount = _spent(states, e, t, start, last_fall)
            if amount >= DROP_MIN:
                drops.append(_Drop(start, last_fall, amount,
                                   [(a, b, _spent(states, e, t, a, b)) for a, b in pieces]))
            i = last_fall + 1
        else:
            i += 1
    return drops


def _spent(states, e, t, a: int, b: int) -> float:
    return float(e[a] - e[b] + (t[b] - t[a]) * states[a].multiplier / REGEN_S_PER_ELIXIR)


def _is_snap(before: float, after: float, amount: float) -> bool:
    """The fractional sliver vanishing (no whole elixir lost), not a spend."""
    return amount < 1.2 and int(before + 1e-6) == int(after + 0.02)


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
                 ability_uses: dict[int, int] | None = None,
                 cut_times: list[float] | None = None) -> list[Play]:
    """`costs`: card_id -> elixir; `ability_costs`: card_id -> ability elixir for the champions
    and hero-form cards in play; `ability_uses`: card_id -> uses per deploy (default 1);
    `cut_times`: video times of jump cuts (edited videos)."""
    ctx = _Context(states, costs, names, ability_costs or {}, ability_uses or {})
    ctx.cut_times = cut_times or []
    drops = find_drops(states, ctx.e)
    if ctx.button_mode:
        ctx.add_button_abilities(drops)
    for d in drops:
        ctx.explain(d)
    ctx.add_hand_only_plays()
    ctx.plays.sort(key=lambda p: (p.t, p.t_drag or p.t))
    return ctx.plays


class _Context:
    def __init__(self, states, costs, names, ability_costs, ability_uses):
        self.states, self.costs, self.names = states, costs, names
        self.ability_costs, self.ability_uses = ability_costs, ability_uses
        self.times = [s.t for s in states]
        self.e = _elixir_series(states)
        self.series = {slot: _slot_series(states, slot) for slot in SLOTS}
        self.nxt = [(s.hand["next"].card_id if s.hand["next"].state == "card" else None) for s in states]
        self.plays: list[Play] = []
        self.used: set[tuple[str, int]] = set()  # (slot, last-seen frame) already attributed
        self.uses_left: dict[int, int] = {}  # champion / hero -> ability uses left since its last deploy
        self.buttons = {k: _button_series(states, k) for k in (0, 1)}
        self.button_mode = any(v != "absent" for ser in self.buttons.values() for v in ser)

    # -- explaining one drop, most specific explanation first

    def explain(self, d: _Drop) -> None:
        if d.amount < DROP_MIN:
            return  # fully accounted for by a button ability use
        cands = self._candidates(d.i_end)
        base = self._base(d.i_start, d.i_end, d.amount)
        exact = [c for c in cands if abs(self.costs.get(c[1], -99) - d.amount) <= COST_TOLERANCE]
        if exact:
            self._card(max(exact, key=lambda c: c[3]), d.i_end, base)  # most recently vacated
            return
        pair = _pair_match(cands, self.costs, d.amount, PAIR_TOLERANCE)
        if pair:
            for c in sorted(pair, key=lambda c: c[3]):
                self._card(c, d.i_end, base, note="two cards released together; drop split by cost", cap="medium")
            return
        if len(d.pieces) > 1 and self._explain_pieces(d):
            return
        if not self.button_mode and self._ability(d.i_start, d.i_end, d.amount, cands, base):
            return
        if not cands and d.amount < SMALL_UNEXPLAINED:
            # A sliver snap, or an ability with no unused deploy. Every one of these was "not a
            # play" in review; real 1-elixir cards are caught from the hand instead.
            return
        plausible = [c for c in cands if d.amount >= MISMATCH_MIN_FRACTION * self.costs.get(c[1], 99)]
        if plausible:
            c = min(plausible, key=lambda c: abs(self.costs.get(c[1], 99) - d.amount))
            note = f"cost mismatch: measured {d.amount:.2f}, {self.names.get(c[1])} costs {self.costs.get(c[1])}"
            self._card(c, d.i_end, base, note=note, cap="medium" if len(cands) == 1 else "low")
            return
        if cands and d.amount < SMALL_UNEXPLAINED:
            return  # e.g. a card hovered over the board while the sliver snapped
        t_end = self.times[d.i_end]
        if self.times[-1] - t_end <= END_GUARD_S or any(abs(t_end - c) <= CUT_GUARD_S for c in self.cut_times):
            return  # the bar resetting at game end, or elixir jumping across a jump cut
        self.plays.append(Play(kind="play", card_id=None, name=None, form=None, slot=None, confidence="low",
                               notes=[f"drop of {d.amount:.2f} with no card leaving the hand"], **base))

    def _explain_pieces(self, d: _Drop) -> bool:
        """A drop merged from separate spends: each piece must be a card or an ability."""
        plan = []
        for a, b, amount in d.pieces:
            cands = [c for c in self._candidates(b) if all(c[:2] != p[1][:2] for p in plan if p[0] == "card")]
            exact = [c for c in cands if abs(self.costs.get(c[1], -99) - amount) <= COST_TOLERANCE]
            if exact:
                plan.append(("card", max(exact, key=lambda c: c[3]), a, b, amount))
            elif not self.button_mode and self._usable_ability(amount) is not None:
                plan.append(("ability", self._usable_ability(amount), a, b, amount))
            else:
                return False
        for kind, what, a, b, amount in plan:
            base = self._base(a, b, amount)
            if kind == "card":
                self._card(what, b, base, note="split from a merged drop", cap="medium")
            else:
                self._ability(a, b, amount, [], base, note="split from a merged drop")
        return True

    # -- building events

    def _candidates(self, i: int) -> list:
        out = []
        for slot in SLOTS:
            v = _vacated(self.series[slot], i, self.times, VACATE_LOOKBACK_S)
            if v and (slot, v[1]) not in self.used:
                (card_id, form), last_seen = v
                out.append((slot, card_id, form, last_seen))
        mid_slide = self._dragged_mid_slide(i)
        if mid_slide and (mid_slide[0], mid_slide[3]) not in self.used:
            out.append(mid_slide)
        return out

    def _dragged_mid_slide(self, i: int):
        """A card can be dragged out while it's still sliding from "next" into an empty slot, so
        it never reads as a card in the slot. Signature: "next" changed from X within the
        lookback, X hasn't been seen in any slot since, and a slot is empty at the drop."""
        k = i
        while k > 0 and self.times[i] - self.times[k] <= VACATE_LOOKBACK_S and self.nxt[k - 1] == self.nxt[k]:
            k -= 1
        if k == 0 or self.times[i] - self.times[k] > VACATE_LOOKBACK_S or self.nxt[k - 1] is None:
            return None
        card = self.nxt[k - 1]
        # ... and it doesn't settle into a slot shortly after either (then it wasn't dragged mid-slide)
        until = next((j for j in range(i, len(self.times)) if self.times[j] - self.times[i] > MID_SLIDE_SETTLE_S), len(self.times))
        seen = any(self.series[slot][j] is not None and self.series[slot][j][0] == card
                   for slot in SLOTS for j in range(k, until))
        empty = [slot for slot in SLOTS if self.series[slot][i] is None]
        if seen or not empty:
            return None
        return (empty[0], card, "normal", k - 1)

    def _base(self, a: int, b: int, amount: float) -> dict:
        return dict(t=self.times[min(a + 1, len(self.times) - 1)], elixir_before=round(float(self.e[a]), 2),
                    elixir_after=round(float(self.e[b]), 2), measured_cost=round(amount, 2))

    def _refill_evidence(self, slot: str, last_seen: int, i_after: int) -> tuple[bool, bool]:
        prev_next = self.nxt[max(last_seen, 0)]
        window = [k for k in range(i_after, len(self.states)) if self.times[k] - self.times[i_after] <= NEXT_ADVANCE_S + 1]
        advanced = any(self.nxt[k] != prev_next for k in window if self.times[k] - self.times[i_after] <= NEXT_ADVANCE_S)
        refilled = prev_next is not None and any(
            self.series[slot][k] is not None and self.series[slot][k][0] == prev_next for k in window)
        return advanced, refilled

    def _card(self, cand, i_after: int, base: dict, note: str | None = None, cap: str = "high") -> None:
        slot, card_id, form, last_seen = cand
        self.used.add((slot, last_seen))
        advanced, refilled = self._refill_evidence(slot, last_seen, i_after)
        notes = [note] if note else []
        if not advanced:
            notes.append("next card did not advance")
        if not refilled:
            notes.append("slot not refilled with the previous next card")
        conf = "high" if advanced and refilled else "medium" if advanced or refilled else "low"
        order = ["low", "medium", "high"]
        conf = order[min(order.index(conf), order.index(cap))]
        self.plays.append(Play(kind="play", card_id=card_id, name=self.names.get(card_id), form=form, slot=slot,
                               confidence=conf, notes=notes, t_drag=self.times[min(last_seen + 1, len(self.times) - 1)],
                               **base))
        if card_id in self.ability_costs:
            self.uses_left[card_id] = self.ability_uses.get(card_id, 1)

    def _usable_ability(self, amount: float) -> int | None:
        for cid, c in self.ability_costs.items():
            if abs(c - amount) <= COST_TOLERANCE and self.uses_left.get(cid, 0) > 0:
                return cid
        return None

    def _ability(self, a: int, b: int, amount: float, cands: list, base: dict, note: str | None = None) -> bool:
        if cands or _is_snap(base["elixir_before"], base["elixir_after"], amount):
            return False
        cid = self._usable_ability(amount)
        if cid is None:
            return False
        self.uses_left[cid] -= 1
        notes = [note] if note else ["no card left the hand; cost matches ability"]
        self.plays.append(Play(kind="ability", card_id=cid, name=self.names.get(cid), form=None, slot=None,
                               confidence="medium", notes=notes, **base))
        return True

    # -- abilities from the button

    def add_button_abilities(self, drops: list[_Drop]) -> None:
        """Each ready -> dark change of an ability button is one use. Its cost comes out of the
        elixir drop it overlaps (e.g. ability + Hog released together reads as one 5 drop)."""
        side_card = next(iter(self.ability_costs)) if len(self.ability_costs) == 1 else None
        consumed: set[int] = set()  # a drop pays for at most one ability
        for side, ser in self.buttons.items():
            for i in range(1, len(ser)):
                if not (ser[i - 1] == "ready" and ser[i] == "dark"):
                    continue
                # The button also goes dark when the ability becomes unaffordable (and flickers in
                # the last seconds of a match). A use spends elixir, so it needs a drop near the
                # change (the button lags the drop by up to ~1 s) that the ability's cost fits into:
                # alone, or together with a card that left the hand.
                t = self.times[i]
                cost = self.ability_costs.get(side_card, 1)
                near = [(k, d) for k, d in enumerate(drops) if k not in consumed
                        and self.times[d.i_start] - ABILITY_BUTTON_LEAD_S <= t <= self.times[d.i_end] + ABILITY_BUTTON_LAG_S
                        and self._ability_fits(d, cost)]
                if not near:
                    continue
                k, d = min(near, key=lambda kd: abs(self.times[kd[1].i_end] - t))
                consumed.add(k)
                d.amount -= cost
                d.pieces = [(a, b, amt) for a, b, amt in d.pieces if abs(amt - cost) > COST_TOLERANCE] or d.pieces
                base = self._base(d.i_start, d.i_end, cost)
                base["t"] = t
                self.plays.append(Play(kind="ability", card_id=side_card, name=self.names.get(side_card), form=None,
                                       slot=None, confidence="high",
                                       notes=[f"{'left' if side == 0 else 'right'} ability button went from ready to dark"],
                                       **base))

    def _ability_fits(self, d: _Drop, cost: int) -> bool:
        if self.e[d.i_start] < cost - 0.2 or d.amount < cost - COST_TOLERANCE:
            return False
        cands = self._candidates(d.i_end)
        if not cands:
            return True
        if any(abs(self.costs.get(c[1], -99) - d.amount) <= COST_TOLERANCE for c in cands):
            return False  # a card explains the whole drop; the button went dark because elixir fell below cost
        rest = d.amount - cost
        if rest < DROP_MIN:
            return True  # ability alone while a card was being dragged
        return any(abs(self.costs.get(c[1], -99) - rest) <= COST_TOLERANCE for c in cands) \
            or _pair_match(cands, self.costs, rest) is not None

    # -- plays seen only in the hand

    def add_hand_only_plays(self) -> None:
        """A card replaced in its slot by the previous next card was played, even if no elixir
        drop was matched to it (e.g. the bar was covered)."""
        for slot in SLOTS:
            ser = self.series[slot]
            i = 1
            while i < len(ser):
                if ser[i] is None and ser[i - 1] is not None:
                    last_seen, card = i - 1, ser[i - 1]
                    k = i
                    while k < len(ser) and ser[k] is None and self.times[k] - self.times[i] <= VACATE_LOOKBACK_S:
                        k += 1
                    prev_next = self.nxt[last_seen]
                    if (k < len(ser) and ser[k] is not None and ser[k][0] == prev_next != card[0]
                            and (slot, last_seen) not in self.used):
                        self.used.add((slot, last_seen))
                        t = self.times[i]
                        self.plays.append(Play(
                            t=t, kind="play", card_id=card[0], name=self.names.get(card[0]), form=card[1], slot=slot,
                            elixir_before=round(float(self.e[last_seen]), 2), elixir_after=round(float(self.e[k]), 2),
                            measured_cost=round(float(self.e[last_seen] - self.e[k]), 2), confidence="medium",
                            notes=["seen in the hand only: replaced by the previous next card, no elixir drop matched"],
                            t_drag=t))
                    i = k
                else:
                    i += 1


def _pair_match(cands: list, costs: dict[int, int], amount: float, tolerance: float = COST_TOLERANCE):
    """Two vacated cards whose costs add up to the drop."""
    best = None
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            total = costs.get(cands[a][1], -99) + costs.get(cands[b][1], -99)
            err = abs(total - amount)
            if err <= tolerance and (best is None or err < best[0]):
                best = (err, [cands[a], cands[b]])
    return best[1] if best else None
