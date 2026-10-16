"""Deploy clocks as a play signal (decisions.md, 2026-10-11).

Placing a troop or building shows a small timer at the spot until the unit lands, whatever the card. The detector
already finds it (class "clock"), so a clock that starts is a play, even when the unit itself is never recognised.
Spells show no clock. Dev set (V4 e15): 91 of the 239 missed plays have an enemy clock start within 1.5 s, and 153 of
the 206 fake plays have none.

Positions are arena tiles (same frame as bot77.spawns), times are video seconds.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from bot77.detect.names import class_to_card
from bot77.spawns import RIVER_Y_TILES, Deploy

CLOCK_GAP_S, CLOCK_TILES = 0.6, 1.5  # a clock detection continues a clock seen this recently, this close


@dataclass
class Clock:
    t: float  # first seen
    last_t: float
    x: float
    y: float
    team: str
    n: int
    conf: float  # best detection confidence


def clock_starts(dets: list[dict], tile_px: float, origin: tuple[float, float], min_conf: float) -> list[Clock]:
    """Clocks from per-frame detections. `team` is the detector's call unless the clock is clearly on one half:
    troops and buildings go on the owner's half, so the side is the better guess there."""
    ox, oy = origin
    rows = sorted((d for d in dets if d["cls"] == "clock" and d["conf"] >= min_conf), key=lambda d: d["t_video"])
    out: list[Clock] = []
    for d in rows:
        x = ((d["x0"] + d["x1"]) / 2 - ox) / tile_px
        y = ((d["y0"] + d["y1"]) / 2 - oy) / tile_px
        for c in out:
            if d["t_video"] - c.last_t <= CLOCK_GAP_S and abs(c.x - x) + abs(c.y - y) <= CLOCK_TILES:
                c.last_t, c.x, c.y, c.n, c.conf = d["t_video"], x, y, c.n + 1, max(c.conf, d["conf"])
                break
        else:
            out.append(Clock(d["t_video"], d["t_video"], x, y, d["team"], 1, d["conf"]))
    return out


def side_team(c: Clock, margin: float) -> str:
    if c.y < RIVER_Y_TILES - margin:
        return "enemy"
    if c.y > RIVER_Y_TILES + margin:
        return "friendly"
    return c.team


def fuse_clocks(enemy: list[Deploy], dets: list[dict], card_names: list[str], tile_px: float,
                origin: tuple[float, float], pov_plays: list[tuple[float, str]], is_spell, r: dict) -> list[Deploy]:
    """r: the "clock" rules (bot77.playrules.DEFAULT_RULES).
    1. Enemy clocks: side of the river (or the detector's team near it); a clock right after one of OUR plays near our
       half is ours.
    2. A deploy with a clock start near it in time and space takes the clock's time (the moment it was placed).
    3. A clock with no deploy becomes a new deploy, its card voted by the detections at that spot over the next
       seconds (any team: freshly placed units get the wrong badge), or None.
    4. Optionally, troop/building deploys with no clock are dropped."""
    clocks = [c for c in clock_starts(dets, tile_px, origin, r["min_conf"]) if c.n >= r["min_dets"]]
    clocks = [c for c in clocks if c.y >= r["min_y"]]  # the HUD trophy icon above the arena reads as a clock
    if r["unit_team_s"] > 0:  # whose units land on the clock: the badge is reliable once the unit is down
        ox, oy = origin
        for c in clocks:
            n = Counter()
            for d in dets:
                if c.t + 0.5 <= d["t_video"] <= c.t + r["unit_team_s"] and d["cls"] != "clock" and d["conf"] >= 0.5:
                    x = ((d["x0"] + d["x1"]) / 2 - ox) / tile_px
                    y = ((d["y0"] + d["y1"]) / 2 - oy) / tile_px
                    if abs(x - c.x) + abs(y - c.y) <= 2.0:
                        n[d["team"]] += d["conf"]
            c.unit_team = n.most_common(1)[0][0] if n and n.most_common(1)[0][1] >= r["unit_team_min"] else None
    clocks = [c for c in clocks if (getattr(c, "unit_team", None) or
                                    (side_team(c, r["side_margin"]) if r["use_side"] else c.team)) == "enemy"]
    if r["pov_window"]:  # our clock (placed past the river after a tower falls, or Miner/Drill anywhere)
        lo, hi = r["pov_window"]
        clocks = [c for c in clocks if not any(lo <= c.t - t <= hi for t, _ in pov_plays)]
    used = set()
    for e in sorted(enemy, key=lambda e: e.t):
        if is_spell(e.card):
            continue
        cand = [(abs(e.t - r["unit_lag_s"] - c.t), k) for k, c in enumerate(clocks) if k not in used
                and -r["win_before_s"] <= e.t - c.t <= r["win_after_s"]
                and abs(c.x - e.x) + abs(c.y - e.y) <= r["tiles"]]
        if cand:
            k = min(cand)[1]
            used.add(k)
            e.clock_t = clocks[k].t
    out = list(enemy)
    if r["drop_unclocked_repeat_s"] > 0:  # a same-card unit with no clock soon after a play is that unit walking back
        keep = []
        for e in sorted(out, key=lambda e: e.t):
            if (getattr(e, "clock_t", None) is None and not is_spell(e.card) and
                    any(k.card == e.card and 0 < e.t - k.t <= r["drop_unclocked_repeat_s"] for k in keep)):
                continue
            keep.append(e)
        out = keep
    if r["drop_unclocked"]:
        out = [e for e in out if is_spell(e.card) or getattr(e, "clock_t", None) is not None
               or e.card in r["keep_unclocked"]]
    if r["new_from_clock"]:
        ox, oy = origin
        for k, c in enumerate(clocks):
            if k in used:
                continue
            votes, classes = Counter(), set()
            for d in dets:
                if not (c.t <= d["t_video"] <= c.t + r["vote_s"]) or d["conf"] < r["vote_conf"] or d["cls"] == "clock":
                    continue
                x = ((d["x0"] + d["x1"]) / 2 - ox) / tile_px
                y = ((d["y0"] + d["y1"]) / 2 - oy) / tile_px
                if abs(x - c.x) + abs(y - c.y) > r["vote_tiles"]:
                    continue
                card, _ = class_to_card(d["cls"], card_names)
                if card is None or is_spell(card):
                    continue
                votes[card] += d["conf"]
                classes.add(d["cls"])
            if not votes and not r["keep_cardless"]:
                continue
            e = Deploy("enemy", c.t + r["unit_lag_s"], c.x, c.y, votes.most_common(1)[0][0] if votes else None,
                       dict(votes), 1, False, False, sorted(classes))
            e.clock_t = c.t
            e.from_clock = True
            out.append(e)
    return sorted(out, key=lambda e: e.t)
