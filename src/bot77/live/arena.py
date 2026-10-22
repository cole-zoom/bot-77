"""Fit the arena crop of a new capture (B v0 calibration, 2026-10-21). bot77.autolayout places the HUD regions within
~10 px but the arena crop only to ~50 px, which shifts placements by up to a tile and changes what the detector sees.
Towers never move, so: detect the four princess towers with the rough crop, then scale / shift the crop so they sit
at their tiles in Cole's hand-calibrated layout (our towers check the fit)."""
from __future__ import annotations

import numpy as np

# princess tower centres in arena tiles, measured on Cole's recordings with the hand-made cole_iphone_mirroring layout
QUEEN_X = (3.55, 14.5)
QUEEN_Y_ENEMY, QUEEN_Y_OURS = 5.68, 20.78
CROP_ASPECT = (1776 - 265) / (1059 - 23)  # the reference crop's height / width (KataCR input proportions)


def tower_centres(frames: list[np.ndarray], layout, detector) -> dict[str, tuple[float, float]]:
    """Median pixel centres of the four princess towers ({team}_{L|R})."""
    pts: dict[str, list] = {}
    for f in frames:
        for d in detector.detect(f, layout):
            if d.name not in ("queen-tower", "cannoneer-tower", "dagger-duchess-tower"):
                continue
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            pts.setdefault(f"{d.team}_{'L' if cx < f.shape[1] / 2 else 'R'}", []).append((cx, cy))
    return {k: (float(np.median([p[0] for p in v])), float(np.median([p[1] for p in v]))) for k, v in pts.items()}


def fit_arena(towers: dict[str, tuple[float, float]], frame_w: int, layout_w: int) -> tuple[list[int], float]:
    """(katacr_arena in layout pixels, check error in tiles): x from the enemy towers' spread, y from their row, our
    towers' row as the independent check."""
    el, er = towers["enemy_L"], towers["enemy_R"]
    tile = (er[0] - el[0]) / (QUEEN_X[1] - QUEEN_X[0])
    ax0 = el[0] - QUEEN_X[0] * tile
    ay0 = (el[1] + er[1]) / 2 - QUEEN_Y_ENEMY * tile
    ax1 = ax0 + 18 * tile
    ay1 = ay0 + CROP_ASPECT * 18 * tile
    err = 0.0
    ours = [towers[k] for k in ("friendly_L", "friendly_R") if k in towers]
    if ours:
        err = abs((np.mean([p[1] for p in ours]) - ay0) / tile - QUEEN_Y_OURS)
    s = layout_w / frame_w  # back to layout pixels
    return [round(ax0 * s), round(ay0 * s), round(ax1 * s), round(ay1 * s)], float(err)
