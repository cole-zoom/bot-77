"""Per-creator screen layouts: named HUD regions in source-pixel coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LAYOUT_DIR = Path("config/layouts")


@dataclass(frozen=True)
class Layout:
    layout_id: str
    frame_size: tuple[int, int]  # (width, height) the regions were marked on
    regions: dict[str, tuple[int, int, int, int]]  # name -> (x0, y0, x1, y1)
    hand: dict = field(default_factory=dict)  # hand-reader constants, in source pixels

    @classmethod
    def load(cls, layout_id: str) -> Layout:
        raw = json.loads((LAYOUT_DIR / f"{layout_id}.json").read_text())
        regions = {k: tuple(v) for k, v in raw["regions"].items()}
        return cls(raw["layout_id"], tuple(raw["frame_size"]), regions, raw.get("hand", {}))

    def scale(self, frame_w: int) -> float:
        return frame_w / self.frame_size[0]

    def box(self, name: str, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Region scaled to a frame of a different resolution (same aspect ratio)."""
        sx, sy = frame_w / self.frame_size[0], frame_h / self.frame_size[1]
        x0, y0, x1, y1 = self.regions[name]
        return round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)

    def crop(self, frame: np.ndarray, name: str) -> np.ndarray:
        x0, y0, x1, y1 = self.box(name, frame.shape[1], frame.shape[0])
        return frame[y0:y1, x0:x1]
