"""Which device recorded a replay video, so it is read with the right layouts (2026-11-15).

BlueStacks (to 2026-11-02) and the Galaxy A16 (from 2026-11-13) both record the replay viewer at 1080x1920, but the
A16 draws the arena smaller and lower: the opponent's princess HP bars (red) sit at y ~499 on BlueStacks and ~594 on
the A16 (config/layouts/replay_a16_*.json). Read off a frame mid-match, when the towers are standing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

LAYOUTS = {"bluestacks": ("replay_bottom", "replay_top"), "a16": ("replay_a16_bottom", "replay_a16_top")}
SPLIT_Y = 546  # between the two devices' red bar rows


def red_bar_rows(frame: np.ndarray) -> list[int]:
    """Top rows of the bar-shaped red blobs in the upper arena (the opponent's princess tower HP bars)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 120)).astype(np.uint8)
    y0, y1 = 350, 800
    n, _, st, _ = cv2.connectedComponentsWithStats(red[y0:y1], 8)
    return [int(st[i, 1]) + y0 for i in range(1, n)
            if 60 <= st[i, 2] <= 140 and 4 <= st[i, 3] <= 30 and st[i, 4] > 0.5 * st[i, 2] * st[i, 3]]


def device_of(frames: list[np.ndarray]) -> str | None:
    """'bluestacks' | 'a16' from a few frames, or None if no bar was found in any."""
    rows = [y for f in frames for y in red_bar_rows(f)]
    if not rows:
        return None
    return "bluestacks" if float(np.median(rows)) < SPLIT_Y else "a16"


def layouts_for(video: Path, times: tuple[float, ...] = (12.0, 30.0, 60.0)) -> tuple[str, str]:
    """(bottom layout, top layout) for a replay video. Raises if the device can't be told."""
    from bot77.calibrate import read_frame
    frames = []
    for t in times:
        try:
            frames.append(read_frame(video, t))
        except Exception:  # noqa: BLE001  (a short video)
            pass
    dev = device_of([f for f in frames if f is not None])
    if dev is None:
        raise RuntimeError(f"{video}: no opponent HP bar found, device unknown")
    return LAYOUTS[dev]
