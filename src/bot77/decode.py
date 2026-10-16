"""Opponent play decoder: pick the most likely *legal* sequence of opponent plays from the deploy candidates
(decisions.md, 2026-10-10). Game rules used:

- the deck is 8 cards; every play is one of them;
- the cycle: a played card goes to the back of the queue, so it comes back to hand only after 4 other plays
  (applied as a penalty, not a ban: a missed play would otherwise make a legal repeat look illegal);
- Mirror plays a copy of the previous card, so it looks exactly like that card; a "repeat" right after the same card
  is Mirror when Mirror fits the deck;
- a candidate with weak evidence can be dropped as not a play (a unit walking in, a spawn, a misdetection).

Dynamic programming over (event index, last 4 plays kept). Scores: card log-likelihood from the detection votes, a
keep score from the candidate's evidence (fit on the development set, never on the benchmark), the cycle penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CYCLE = 4  # a card returns only after this many other plays


@dataclass
class DecodeParams:
    keep_bias: float = 0.0  # added to every candidate's keep score
    keep_per_det: float = 0.08  # per detection in the candidate (capped)
    keep_det_cap: int = 25
    keep_per_unit: float = 0.3
    keep_subunit_only: float = -2.0
    keep_spell: float = 0.5
    keep_off_deck: float = -1.5  # none of the candidate's votes is a deck card
    vote_smoothing: float = 1.0
    off_deck_logp: float = -4.0
    cycle_penalty: float = 3.0  # per repeat inside the cycle window
    mirror_prior: float = -1.0  # log prior of calling a repeat Mirror (when Mirror is plausible)
    extra: dict = field(default_factory=dict)


def keep_score(e, deck: set[str], is_spell, p: DecodeParams) -> float:
    n = sum(e.card_votes.values())
    s = p.keep_bias + p.keep_per_det * min(n, p.keep_det_cap) + p.keep_per_unit * min(e.units, 6)
    if e.subunit_only:
        s += p.keep_subunit_only
    if e.card and is_spell(e.card):
        s += p.keep_spell
    if not any(c in deck for c in e.card_votes):
        s += p.keep_off_deck
    return s


def card_logp(e, deck: list[str], p: DecodeParams) -> dict[str, float]:
    tot = sum(v for c, v in e.card_votes.items() if c in deck) + p.vote_smoothing * len(deck)
    return {c: math.log((e.card_votes.get(c, 0) + p.vote_smoothing) / tot) if tot else p.off_deck_logp for c in deck}


def decode(events: list, deck: list[str], is_spell=lambda c: False, p: DecodeParams | None = None,
           mirror_in_deck: bool = False) -> list:
    """events: Deploy-like objects sorted by time (card_votes, units, subunit_only, card). Returns the kept events with
    .card set (possibly 'Mirror'); dropped events are left out."""
    p = p or DecodeParams()
    if not events or not deck:
        return events
    deck_set = set(deck)
    cards = list(deck) + (["Mirror"] if mirror_in_deck and "Mirror" not in deck else [])
    # state: tuple of the last CYCLE kept cards (most recent last) -> (score, back-pointer chain)
    beam: dict[tuple, tuple[float, tuple]] = {(): (0.0, ())}
    for i, e in enumerate(events):
        ks = keep_score(e, deck_set, is_spell, p)
        lp = card_logp(e, [c for c in cards if c != "Mirror"], p)
        nxt: dict[tuple, tuple[float, tuple]] = {}

        def push(state, score, chain):
            if state not in nxt or score > nxt[state][0]:
                nxt[state] = (score, chain)

        for state, (score, chain) in beam.items():
            push(state, score, chain)  # drop this candidate
            for c in cards:
                if c == "Mirror":
                    if not state:
                        continue
                    prev = state[-1]
                    s = score + ks + lp.get(prev, p.off_deck_logp) + p.mirror_prior
                else:
                    s = score + ks + lp[c]
                s -= p.cycle_penalty * (c in state)
                push((state + (c,))[-CYCLE:], s, chain + ((i, c),))
        # keep the beam small
        beam = dict(sorted(nxt.items(), key=lambda kv: -kv[1][0])[:400])
    best = max(beam.values(), key=lambda v: v[0])[1]
    out = []
    for i, c in best:
        e = events[i]
        e.card = c
        out.append(e)
    return out
