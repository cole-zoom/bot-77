"""The home screen's account numbers, read between games (2026-11-14, Cole: log trophies as the bot grinds bot-77, so
card upgrades and climbing can be told apart from model changes).

Boxes are for the 1080x1920 home screen of the Galaxy A16 (bot-77, 11-14: trophies 1330, gold 134 342, gems 560, the
XP bar's number 145, all read at every OCR scale). The king level sits inside its badge too small to read; not kept.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from bot77.readers.clock import ocr_text

BOXES = {"name": (40, 455, 300, 505), "trophies": (440, 470, 600, 525), "gold": (520, 115, 740, 175),
         "gems": (840, 115, 1010, 175), "xp": (130, 115, 300, 175)}
LOG = Path("data/live/home_reads.jsonl")


def read_home(im: np.ndarray) -> dict:
    """{name, trophies, gold, gems, xp}: numbers as int, None where the box didn't read."""
    if im.shape[:2] != (1920, 1080):
        return {}
    out = {}
    for k, (x0, y0, x1, y1) in BOXES.items():
        text = ocr_text(im[y0:y1, x0:x1], 2).strip()
        if k == "name":
            out[k] = text or None
        else:
            digits = re.sub(r"\D", "", text)
            out[k] = int(digits) if digits else None
    return out


def log_home(im: np.ndarray, **extra) -> dict:
    r = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **read_home(im), **extra}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(r) + "\n")
    return r
