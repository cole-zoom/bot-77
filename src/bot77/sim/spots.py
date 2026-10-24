"""Where a play landed, for the plays that have no spot in `plays` (docs/simulator.md S2/S3).

`plays` (source hand) gives the exact card and time; only 43% of the human replays' plays carry a spot (the box centre
of the unit's first detection, from the older matcher). Here: the first new detection of that card for that team in
the 2.5 s after the play, not near a copy that was already there in the 1.5 s before; the centroid of that frame's new
bodies is the spot (box centre, bottom frame, like `plays.x / y`). Spells only have a spot when the detector has a
class for them (Fireball, Zap, The Log, ... not Arrows).
"""
from __future__ import annotations

import collections
from functools import lru_cache

from bot77.sim.scenario import _rows

RUN = "v5l_e15"
AFTER_S, BEFORE_S, NEAR = 2.5, 1.5, 2.0
TEAM = {"bottom": "friendly", "top": "enemy"}


@lru_cache(maxsize=64)
def _dets(game_id: str) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
    out = collections.defaultdict(list)
    for d in _rows("detections", f"match_id = '{game_id}' AND run_id = '{RUN}'", ["t", "card", "team", "x", "y"],
                   limit=10**7):
        if d["card"]:
            out[(d["card"], d["team"])].append((d["t"], d["x"], d["y"]))
    for v in out.values():
        v.sort()
    return out


def recover(game_id: str, play: dict) -> tuple[float, float] | None:
    """(x, y) box centre in the bottom frame for a play (card, side, t = video time), or None."""
    if play.get("x") is not None:
        return play["x"], play["y"]
    dets = _dets(game_id).get((play["card"], TEAM[play["side"]]), [])
    t = play["t"]
    before = [(x, y) for tt, x, y in dets if t - BEFORE_S <= tt < t]
    new = [(tt, x, y) for tt, x, y in dets if t <= tt <= t + AFTER_S
           and all((x - bx) ** 2 + (y - by) ** 2 > NEAR ** 2 for bx, by in before)]
    if not new:
        return None
    first = new[0][0]
    pts = [(x, y) for tt, x, y in new if tt == first]
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
