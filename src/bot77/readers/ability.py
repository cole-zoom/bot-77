"""Read the champion / hero ability buttons above the hand.

Three states, by colour (measured on Alina's Mighty Miner button):
- ready:  bright cyan button with a pink elixir-cost pip
- dark:   navy button with a grey pip: ability used (stays dark until the champion leaves),
          or not affordable right now (goes back to ready when elixir refills)
- absent: no button (the champion / hero isn't on the field)
"""

from __future__ import annotations

import cv2
import numpy as np

from bot77.layout import Layout

SIDES = ("ability_left", "ability_right")
READY_CYAN, READY_PINK, DARK_NAVY = 0.08, 0.015, 0.2  # minimum pixel fractions


def read_ability(frame: np.ndarray, layout: Layout, side: str = "ability_right") -> str:
    if side not in layout.regions:
        return "absent"
    hsv = cv2.cvtColor(layout.crop(frame, side), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    cyan = ((h > 85) & (h < 110) & (s > 120) & (v > 170)).mean()
    pink = ((h > 140) & (h < 170) & (s > 120) & (v > 150)).mean()
    navy = ((h > 100) & (h < 130) & (s > 120) & (v > 40) & (v < 140)).mean()
    if cyan > READY_CYAN and pink > READY_PINK:
        return "ready"
    if navy > DARK_NAVY:
        return "dark"
    return "absent"
