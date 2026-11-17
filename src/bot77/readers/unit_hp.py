"""A unit's HP fraction from its HP bar (the detector's `bar` box, next to its `bar-level` badge), 2026-11-15.

The bar shows once a unit is damaged: a fill from the left over a translucent track, the level badge covering the
bar's left end. Measured on ten v5l_e15 bars of the bot's replays (red, pink, light-blue and white fills, over grass,
sand and units): every fill column has median V >= 148, every track column <= 141 (the track is dark and
translucent). Hue- or split-based rules were tried first and misread white fills over sand and red fills over light
arena (read 1.0 and 0.92 for ~0.7 and ~0.3). Fraction = the contiguous bright run from the left / the bar's length
(from the badge's right edge to the box's right edge less its outline).
"""
from __future__ import annotations

import cv2
import numpy as np

OUTLINE = 3  # px of dark outline inside the box's right edge
GAP = 2      # non-fill columns allowed inside the fill run (antialiasing, a damage tick)


FILL_V = 145  # median V of a column: fill above, track below


def fill_mask(hsv_strip: np.ndarray) -> np.ndarray:
    """Per column: is it bar fill (bright), not the dark track?"""
    return np.median(hsv_strip[:, :, 2].astype(np.float32), axis=0) >= FILL_V


SLACK = 6  # px the box may sit off the bar vertically (the unit moved between the detector's frame and this one)
BAND = 3   # rows read


def band_rows(frame: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> tuple[int, int] | None:
    """The BAND rows near the box that look most like a bar: the largest spread of brightness along the row (bright
    fill against dark track). Sand or grass rows are flat; the bar's outline rows are flat and dark."""
    lo, hi = max(0, y0 - SLACK), min(frame.shape[0], y1 + SLACK)
    if hi - lo < BAND or x1 - x0 < 8:
        return None
    v = cv2.cvtColor(frame[lo:hi, x0:x1], cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    best, at = -1.0, None
    for y in range(0, hi - lo - BAND + 1):
        col = np.median(v[y: y + BAND], axis=0)
        spread = float(np.percentile(col, 90) - np.percentile(col, 10))
        dark = float(np.percentile(col, 10)) < FILL_V  # the track must be in it, unless the bar is nearly full
        if dark and spread > best:
            best, at = spread, y
    if at is None:  # nearly full bar: no dark track in reach; the box's middle rows
        mid = (y0 + y1) // 2
        return mid - 1, mid + 2
    return lo + at, lo + at + BAND


def bar_fill(frame: np.ndarray, bar: tuple[float, float, float, float],
             badge: tuple[float, float, float, float] | None = None) -> float | None:
    """HP fraction 0-1 of the bar box (x0, y0, x1, y1, frame pixels), or None if it doesn't read as a bar."""
    x0, y0, x1, y1 = (int(round(v)) for v in bar)
    if badge is not None and badge[0] <= x0 + 4 and badge[2] > x0:  # the badge covers the bar's left end
        x0 = int(round(badge[2])) + 1
    r = x1 - OUTLINE
    h = y1 - y0
    if r - x0 < 8 or h < 3 or x0 < 0 or y0 < 0:
        return None
    band = band_rows(frame, x0, r, y0, y1)
    if band is None:
        return None
    strip = frame[band[0]: band[1], x0: r]
    fill = fill_mask(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV))
    head = np.flatnonzero(fill[: max(3, len(fill) // 5)])
    if len(head) == 0:  # a fill always starts at the left
        return None
    end, gap = head[0], 0
    for i in range(head[0], len(fill)):
        if fill[i]:
            end, gap = i, 0
        else:
            gap += 1
            if gap > GAP:
                break
    return float((end + 1) / len(fill))


# Bar <-> unit pairing (moved from bot77.sim.validate 2026-11-16 so the live reader shares it)
def _badge_of(bar: dict, bars: list[dict]) -> dict | None:
    cy = (bar["box"][1] + bar["box"][3]) / 2
    for q in bars:
        if q["cls"] == "bar-level" and abs((q["box"][1] + q["box"][3]) / 2 - cy) < 15 \
                and q["box"][0] <= bar["box"][0] + 4 and q["box"][2] > bar["box"][0]:
            return q
    return None


def _bar_team(bar: dict, bars: list[dict]) -> str:
    """A bar's team is its level badge's when it has one: the detector calls a bar's team by its fill colour, and a
    just-hit bar is white (bl_2026-10-23_r25 50.5 s: our Giant's bar read 'opp', its badge 'own')."""
    badge = _badge_of(bar, bars)
    return badge["team"] if badge else bar["team"]


def bar_of(unit_box: tuple, team: str, bars: list[dict]) -> tuple[tuple, tuple | None] | None:
    """The HP bar (and its level badge) over a unit: same team, the bar's centre within the unit box's width, its
    bottom from 45 px above the box's top to 30 px into it (the bar hangs over the unit's head)."""
    x0, y0, x1, _ = unit_box
    best, bd = None, 1e9
    for b in bars:
        if b["cls"] != "bar" or _bar_team(b, bars) != team:
            continue
        bx0, by0, bx1, by1 = b["box"]
        cx = (bx0 + bx1) / 2
        if x0 - 10 <= cx <= x1 + 10 and y0 - 45 <= by1 <= y0 + 30:
            d = abs(cx - (x0 + x1) / 2) + abs(by1 - y0)
            if d < bd:
                best, bd = b, d
    if best is None:
        return None
    badge = _badge_of(best, bars)
    return best["box"], (badge["box"] if badge else None)


def attach_bar_hp(frame: np.ndarray, units: list[dict], bars: list[dict]) -> list[dict]:
    """Copies of `units` (dicts with box, team), each with hp_frac read off the HP bar over its head where one is
    found; `bars` are the frame's `bar` / `bar-level` detections (cls, team, box). Each bar goes to one unit,
    closest pairs first (overlapping units would otherwise share one)."""
    out, free = [dict(u) for u in units], list(bars)
    # each bar to one unit, closest pairs first (overlapping units would otherwise share one)
    pairs = []
    for i, u in enumerate(out):
        hit = bar_of(u["box"], u["team"], free) if u.get("box") else None
        if hit:
            pairs.append((abs((hit[0][0] + hit[0][2]) / 2 - (u["box"][0] + u["box"][2]) / 2) + abs(hit[0][3] - u["box"][1]), i))
    for _, i in sorted(pairs):
        u = out[i]
        hit = bar_of(u["box"], u["team"], free)
        if not hit:
            continue
        free = [b for b in free if b["box"] != hit[0]]
        f = bar_fill(frame, hit[0], hit[1])
        if f is not None:
            u["hp_frac"], u["hp_bar"] = f, hit  # hp_bar: (bar box, badge box), for checking by eye
    return out
