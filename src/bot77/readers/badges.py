"""Find unit level badges on the arena and tell the teams apart by colour.

Every troop and building carries a small level badge with its HP bar to the right: red for
the opponent's units, blue for ours. A badge is a saturated, roughly square blob with white
digits inside. Tower badges (fixed positions) are excluded by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bot77.layout import Layout

# HSV ranges (OpenCV hue is 0-179)
RED_LO, RED_HI = 8, 168  # red wraps around: hue < RED_LO or > RED_HI
BLUE = (95, 118)
MIN_S, MIN_V = 150, 150
SIZE_PX = (12, 44)  # badge side length at the source resolution; generous across layouts
MIN_SQUARENESS = 0.6
MIN_DIGIT_FRACTION = 0.08  # white-ish pixels inside the badge (the level digits)


@dataclass(frozen=True)
class Badge:
    team: str  # enemy | friendly
    x: float  # badge centre, source-frame pixels
    y: float
    size: float


def _team_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sat = (s > MIN_S) & (v > MIN_V)
    return {
        "enemy": (((h < RED_LO) | (h > RED_HI)) & sat).astype(np.uint8),
        "friendly": ((h >= BLUE[0]) & (h <= BLUE[1]) & sat).astype(np.uint8),
    }


def find_badges(frame: np.ndarray, layout: Layout) -> list[Badge]:
    x0, y0, x1, y1 = layout.box("arena", frame.shape[1], frame.shape[0])
    arena = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 200)).astype(np.uint8)
    lo, hi = SIZE_PX
    out = []
    for team, mask in _team_masks(hsv).items():
        # close the digit holes so a badge is one blob; the HP bar may still be attached
        blob = cv2.morphologyEx(mask | white * mask.max(), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if not lo <= bh <= hi:
                continue
            side = bh
            if bw > side * 1.35:  # badge merged with its HP bar: the badge is the left square
                bw = side
            if not lo <= bw <= hi or min(bw, bh) / max(bw, bh) < MIN_SQUARENESS:
                continue
            inner = white[by + side // 5: by + bh - side // 5, bx + bw // 5: bx + bw - bw // 5]
            if inner.size == 0 or inner.mean() < MIN_DIGIT_FRACTION:
                continue
            ring = mask[by:by + bh, bx:bx + bw].mean()
            if ring < 0.3:  # mostly the badge colour
                continue
            out.append(Badge(team, x0 + bx + bw / 2, y0 + by + bh / 2, float(side)))
        del blob
    return out
