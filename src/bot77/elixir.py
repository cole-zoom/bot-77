"""Opponent elixir check (failure mining, 2026-10-16): drop plays the opponent couldn't have afforded.

The opponent's elixir is simulated from the start of the match: +1 every 2.8 s times the on-screen multiplier
(hud_states.multiplier: 1x / 2x / 3x), capped at 10, minus each logged play's cost (cards.elixir; Mirror = the copied
card + 1; abilities = cards.ability_cost). Missed plays would only lower the real elixir, so the simulation is an
upper bound: when a play takes it below -tol, the logged plays since the opponent last sat at full elixir can't all be
real, and the one with the least evidence is dropped (then re-simulate). Most fakes sit within 4 s of a real play
(a second read of the same play, a spawned unit), which is exactly what an elixir burst exposes.
"""
from __future__ import annotations

REGEN_S = 2.8  # seconds per elixir at 1x
CAP = 10.0


def _rate_fn(timeline: list[tuple[float, int]]):
    """Elixir gained between two video times, integrating the multiplier timeline [(t_video, multiplier)]."""
    # collapse the per-frame timeline into segments where the multiplier changes
    segs: list[tuple[float, int]] = []
    for t, m in sorted(timeline):
        m = max(m or 1, 1)
        if not segs or segs[-1][1] != m:
            segs.append((t, m))
    if not segs:
        segs = [(float("-inf"), 1)]
    segs[0] = (float("-inf"), segs[0][1])

    def gain(a: float, b: float) -> float:
        g = 0.0
        for k, (s, m) in enumerate(segs):
            e = segs[k + 1][0] if k + 1 < len(segs) else float("inf")
            lo, hi = max(a, s), min(b, e)
            if hi > lo:
                g += (hi - lo) * m / REGEN_S
        return g

    return gain


def evidence(e) -> float:
    """How sure a play is: a deploy clock (a real placement) outweighs any vote count."""
    s = float(sum(e.card_votes.values())) if e.card_votes else 0.0
    if getattr(e, "clock_t", None) is not None:
        s += 100.0
    return s


def check(events: list, timeline: list[tuple[float, int]], cost, start: float, tol: float = 1.0,
          burst_s: float = 12.0, policy: str = "pair", pair_s: float = 4.0) -> tuple[list, list]:
    """(kept, dropped). `cost(e)` -> elixir or None (unknown: free). `start`: elixir at the first timeline frame
    (5 at a match start; 10 when the recording starts mid-match, the safe assumption).
    `policy`: which play of an overspent burst goes. "last": the latest one without a deploy clock (fakes follow a real
    play: a second read, a spawned unit; held-back trace 2026-10-16: a real Furnace lost to a fake Mega Knight after
    it under "evidence"); "evidence": the one with the fewest detector votes; "pair" (default): only plays with another
    play within pair_s are suspects (84% of fakes, failure mining), the weaker of them goes, and an overspend with no
    such suspect is accepted (the play times or a card's cost are off, not a fake)."""
    gain = _rate_fn(timeline)
    t0 = min([t for t, _ in timeline] or [min((e.t for e in events), default=0.0)])
    kept = sorted(events, key=lambda e: e.t)
    dropped = []
    while True:
        elixir, prev, full_at, drop = start, t0, t0, None
        for i, e in enumerate(kept):
            elixir = elixir + gain(prev, e.t)
            if elixir >= CAP:
                elixir, full_at = CAP, e.t  # sat at full: leaked elixir can't be spent later
            prev = e.t
            c = cost(e)
            if c is None:
                continue
            elixir -= c
            if elixir >= -tol:
                continue
            # candidates: plays since the opponent was last full (or within burst_s), up to the overspend
            cands = [j for j in range(i + 1) if kept[j].t >= min(full_at, e.t - burst_s) and cost(kept[j])]
            if policy == "pair":
                # only a play with another one within pair_s can be a second read / spawn of it; keep the stronger
                cands = [j for j in cands if any(k != j and abs(kept[k].t - kept[j].t) <= pair_s for k in range(len(kept)))
                         and getattr(kept[j], "clock_t", None) is None]
                if not cands:
                    elixir = 0.0  # can't tell which play is wrong: accept it and go on
                    continue
                drop = min(cands, key=lambda j: (evidence(kept[j]), -kept[j].t))
            elif policy == "last":
                unclocked = [j for j in cands if getattr(kept[j], "clock_t", None) is None]
                drop = max(unclocked or cands, key=lambda j: kept[j].t)
            else:
                drop = min(cands, key=lambda j: (evidence(kept[j]), -kept[j].t))
            break
        if drop is None:
            return kept, dropped
        dropped.append(kept.pop(drop))
