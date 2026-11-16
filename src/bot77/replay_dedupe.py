"""The same battle recorded twice (2026-11-16). The replay runner skips battles whose Battle Log thumbnail it has
recorded before, but on the phone the bot's own deck fills half of every thumbnail, so different games score 0.86-0.93
against the 0.90 "recorded before" bar and the same game can come back under a new id (21 pairs of 1012 replays,
both hands' plays identical). A replay's hand plays settle it: the same game has the same plays at the same clock.

The later recording of each pair is listed in config/duplicate_replays.json (scripts/find_duplicate_replays.py) and
gets split 'duplicate' in build_sources.py, so one copy is used.
"""
from __future__ import annotations

import re

SAME_CLOCK_S = 1.0  # a play's match clock in two recordings of one game (detector jitter)
SAME_SHARE = 0.8  # share of each recording's plays found in the other (same games 0.89-1.0, different < 0.6)
MIN_PLAYS = 6


def _share(a: list[tuple], b: list[tuple]) -> float:
    used, n = set(), 0
    for t, card, side in a:
        for i, (u, c, s) in enumerate(b):
            if i not in used and c == card and s == side and abs(u - t) <= SAME_CLOCK_S:
                used.add(i); n += 1
                break
    return n / max(len(a), 1)


def same_game(a: list[tuple], b: list[tuple]) -> bool:
    """a, b: (match clock, card, side) hand plays of two replays."""
    if min(len(a), len(b)) < MIN_PLAYS:
        return False
    return min(_share(a, b), _share(b, a)) >= SAME_SHARE


def recorded_order(match_id: str) -> tuple:
    """Recording order of a replay id (tv_/bl_YYYY-MM-DD_rNN_m01): day, then index."""
    m = re.search(r"(\d{4}-\d\d-\d\d)_r(\d+)", match_id)
    return (m[1], int(m[2]), match_id) if m else ("", 0, match_id)


def duplicates(plays: dict[str, list[tuple]]) -> dict[str, str]:
    """{later recording: first recording} over replays' hand plays."""
    ids = sorted((g for g in plays if len(plays[g]) >= MIN_PLAYS), key=recorded_order)
    sig = {g: {(c, s) for _, c, s in plays[g]} for g in ids}
    out: dict[str, str] = {}
    for i, a in enumerate(ids):
        if a in out:
            continue
        for b in ids[i + 1:]:
            if b in out or len(sig[a] & sig[b]) < SAME_SHARE * max(len(sig[a]), len(sig[b])):
                continue
            if same_game(plays[a], plays[b]):
                out[b] = a
    return out
