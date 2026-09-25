"""Read elixir from the fill of the elixir bar, to a fraction of a point.

The bar has 10 equal segments. Whole elixir is drawn pink; progress toward the next
point is a lighter blue sliver in the following segment; the rest is dark blue. White
pixels (the elixir number, the selected card's cost outline, "Elixir bar is full!") are
ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bot77.layout import Layout

SEGMENTS = 10
PARTIAL_BRIGHTER = 1.2  # a partial column is this much brighter than the bar's empty level
MIN_READABLE = 0.5

@dataclass(frozen=True)
class ElixirRead:
    elixir: float | None  # e.g. 4.6; None when unreadable
    whole: int | None  # pink segments only
    consistent: bool  # floor(elixir) agrees with the pink segment count
    readable_frac: float  # fraction of bar columns that matched a known colour


def _is_pink(cols: np.ndarray) -> np.ndarray:
    b, g, r = cols[:, 0], cols[:, 1], cols[:, 2]
    return (r > 150) & (b > 150) & (g < 0.8 * np.minimum(r, b))


def _is_white(cols: np.ndarray) -> np.ndarray:
    return cols.min(axis=1) > 170  # number, cost outline, "Elixir bar is full!"


def read_elixir(frame: np.ndarray, layout: Layout) -> ElixirRead:
    """Colours are judged relative to the bar itself: captures differ in colour processing
    (e.g. one creator's empty segments are brighter blue than another's partial ones)."""
    bar = layout.crop(frame, "elixir_bar")
    h = bar.shape[0]
    cols = np.median(bar[int(h * 0.3):int(h * 0.7)].astype(np.float32), axis=0)  # BGR per column

    pink, white = _is_pink(cols), _is_white(cols)
    blue = ~pink & ~white & (cols[:, 0] > cols[:, 2] + 30)  # empty or partial: blue-dominant
    readable = float((pink | blue).mean())
    if readable < MIN_READABLE:
        return ElixirRead(None, None, False, readable)

    width = len(cols) / SEGMENTS
    pink_idx = np.where(pink)[0]
    pink_end = pink_idx.max() + 1 if len(pink_idx) else 0
    whole = min(int(pink_end / width + 0.15), SEGMENTS)  # tolerate a few px of misalignment

    # The partial sliver starts where the pink ends and is brighter than the empty level.
    brightness = cols.mean(axis=1)
    beyond = np.where(blue)[0]
    beyond = beyond[beyond >= pink_end]
    end = pink_end
    if len(beyond):
        partial_min = np.percentile(brightness[beyond], 10) * PARTIAL_BRIGHTER
        x, gap = pink_end, 0
        while x < len(cols) and gap <= 2:  # walk right, allowing a couple of stray columns
            if blue[x] and brightness[x] > partial_min:
                end, gap = x + 1, 0
            else:
                gap += 1
            x += 1
    elixir = min(end / width, SEGMENTS)
    consistent = whole <= elixir + 0.2 and elixir < whole + 1.2
    return ElixirRead(round(float(elixir), 2), whole, bool(consistent), readable)
