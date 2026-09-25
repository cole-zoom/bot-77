"""Read the match clock: phase (regulation / overtime), time left, and elixir multiplier."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ocrmac import ocrmac
from PIL import Image

from bot77.layout import Layout
from bot77.readers.hand import normalise

REGULATION_S = 180
OVERTIME_S = 120
HUD_ASSETS = Path("assets/hud")
MULTIPLIER_MIN_SCORE = 0.5
OCR_UPSCALES = (3, 1, 2)  # tried in order: 3x suits white text, but breaks the red last-30s digits

_CLOCK = re.compile(r"(\d)\s*[:.]\s*(\d{2})")
_OCR_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "S": "5", "B": "8"})


@dataclass(frozen=True)
class ClockRead:
    phase: str | None  # regulation | overtime | None (no clock on screen)
    time_left_s: int | None
    multiplier: int | None  # 1, 2 or 3
    elapsed_s: int | None  # seconds since the match started


def ocr_text(image_bgr: np.ndarray, upscale: int = 3) -> str:
    big = image_bgr if upscale == 1 else cv2.resize(image_bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    results = ocrmac.OCR(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)), recognition_level="accurate").recognize()
    return " ".join(text for text, _conf, _box in results)


def parse_clock(text: str) -> int | None:
    m = _CLOCK.search(text.translate(_OCR_DIGIT_FIXES))
    if not m or int(m[2]) > 59:
        return None
    return int(m[1]) * 60 + int(m[2])


def parse_phase(label: str) -> str | None:
    label = label.lower().replace(" ", "")
    if "overtime" in label:
        return "overtime"
    if "time" in label or "left" in label:
        return "regulation"
    return None


# The top HUD's size relative to the hand bar varies with the capture's aspect ratio (Ian's
# is ~0.8x his hand's scale), so the multiplier is searched over a few sizes instead of
# trusting the layout's hand scale.
MULTIPLIER_SCALES = (0.55, 0.65, 0.75, 0.85, 1.0)


@lru_cache(maxsize=32)
def _multiplier_templates(scale: float) -> dict[int, np.ndarray]:
    out = {}
    for n in (2, 3):
        im = cv2.imread(str(HUD_ASSETS / f"multiplier_x{n}.png"), cv2.IMREAD_GRAYSCALE)
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out[n] = normalise(im, 5.0 * scale)
    return out


def read_multiplier(frame: np.ndarray, layout: Layout) -> tuple[int, float]:
    """(multiplier, score). No x2/x3 drop on screen means normal elixir (1)."""
    crop = layout.crop(frame, "multiplier")
    s = layout.scale(frame.shape[1])
    if abs(s - 1.0) > 1e-3:
        crop = cv2.resize(crop, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
    gray_raw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scores: dict[int, float] = {}
    for scale in MULTIPLIER_SCALES:
        gray = normalise(gray_raw, 5.0 * scale)
        for n, t in _multiplier_templates(scale).items():
            if t.shape[0] <= gray.shape[0] and t.shape[1] <= gray.shape[1]:
                score = cv2.minMaxLoc(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED))[1]
                scores[n] = max(scores.get(n, -1.0), score)
    if not scores:
        return 1, 0.0
    n, score = max(scores.items(), key=lambda kv: kv[1])
    return (n, score) if score >= MULTIPLIER_MIN_SCORE else (1, score)


def elapsed(phase: str, time_left_s: int, mode: str = "ladder") -> int | None:
    """Seconds since the match started. Only ladder rules are known: 3:00 regulation then
    2:00 overtime. Other modes (e.g. Princess Gambit's longer overtime) return None."""
    if mode != "ladder":
        return None
    if phase == "overtime":
        return REGULATION_S + OVERTIME_S - time_left_s
    return REGULATION_S - time_left_s


def _read_first(crop: np.ndarray, parse):
    for upscale in OCR_UPSCALES:
        value = parse(ocr_text(crop, upscale))
        if value is not None:
            return value
    return None


def read_clock(frame: np.ndarray, layout: Layout, mode: str = "ladder") -> ClockRead:
    phase = _read_first(layout.crop(frame, "timer_label"), parse_phase)
    left = _read_first(layout.crop(frame, "timer"), parse_clock)
    if phase is None or left is None:
        return ClockRead(phase, left, None, None)
    mult, _ = read_multiplier(frame, layout)
    return ClockRead(phase, left, mult, elapsed(phase, left, mode))
