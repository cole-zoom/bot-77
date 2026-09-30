"""bot77's own unit detector (scripts/train_detector.py): a standard Ultralytics YOLO trained on
KataCR-style generated arenas [Wu et al. 2025, §3] with current classes. It has no faction
output; each unit's team comes from the colour of the level badge the model finds above it
(red = enemy, blue = friendly), with the POV play log as a later cross-check (bot77.spawns).
Same interface as bot77.detect.units.UnitDetector.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bot77.detect.units import INPUT_WH, Detection, UnitDetector
from bot77.layout import Layout

WEIGHTS = Path("data/models/bot77_detector/v1/weights/best.pt")
CONF = 0.35
BADGES = {"bar-level", "bar"}


def badge_team(frame: np.ndarray, box: tuple[float, float, float, float]) -> str | None:
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = frame[max(y0, 0):y1, max(x0, 0):x1]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 130) & (hsv[..., 2] > 110)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 168)) & sat).mean()
    blue = ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 118) & sat).mean()
    if max(red, blue) < 0.08:
        return None
    return "enemy" if red > blue else "friendly"


class Bot77Detector:
    def __init__(self, weights: Path = WEIGHTS, conf: float = CONF, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device or "mps"

    arena = UnitDetector.arena  # same detector crop as KataCR's

    def detect(self, frame: np.ndarray, layout: Layout) -> list[Detection]:
        crop, (ox, oy), sx, sy = self.arena(frame, layout)
        img = cv2.resize(crop, INPUT_WH, interpolation=cv2.INTER_AREA)
        r = self.model.predict(img, imgsz=896, conf=self.conf, device=self.device, verbose=False)[0]
        names = r.names
        raw = [(names[int(c)], float(p), tuple(b)) for b, c, p in
               zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        to_frame = lambda b: (ox + b[0] * sx, oy + b[1] * sy, ox + b[2] * sx, oy + b[3] * sy)
        badges = [(to_frame(b), badge_team(frame, to_frame(b))) for n, _, b in raw if n in BADGES]
        out = []
        for name, conf, b in raw:
            fb = to_frame(b)
            team = None
            if name in BADGES:
                team = badge_team(frame, fb)
            else:
                # nearest badge whose centre sits over the unit's top half
                cx, top, w, h = (fb[0] + fb[2]) / 2, fb[1], fb[2] - fb[0], fb[3] - fb[1]
                near = [(abs((bb[0] + bb[2]) / 2 - cx) + abs((bb[1] + bb[3]) / 2 - top), t) for bb, t in badges
                        if t and fb[0] - w * 0.3 <= (bb[0] + bb[2]) / 2 <= fb[2] + w * 0.3
                        and top - h * 0.6 <= (bb[1] + bb[3]) / 2 <= top + h * 0.5]
                if near:
                    team = min(near)[1]
            # units whose badge isn't visible: side of the river decides (buildings / troops are
            # placed on the owner's half; the POV cross-check fixes the rest later)
            if team is None:
                mid_y = (layout.katacr_arena[1] + layout.katacr_arena[3]) / 2 * layout.scale(frame.shape[1])
                team = "enemy" if (fb[1] + fb[3]) / 2 < mid_y else "friendly"
            out.append(Detection(name, team, conf, fb))
        return out
