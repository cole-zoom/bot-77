"""Unit HP bars read within 0.1 of a reading by eye: blue / white / red fills, over grass and sand, the box off the
bar by a few px (bot77.readers.unit_hp)."""
import json
from pathlib import Path

import cv2
import pytest

from bot77.readers.unit_hp import bar_fill

FIX = Path(__file__).parent / "fixtures/unit_hp"
CASES = json.loads((FIX / "bars.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["file"] for c in CASES])
def test_bar_fill_matches_the_eye(case):
    im = cv2.imread(str(FIX / case["file"]))
    f = bar_fill(im, case["bar"], case["badge"])
    assert f is not None and abs(f - case["by_eye"]) <= 0.1


def test_no_bar_no_reading():
    im = cv2.imread(str(FIX / CASES[0]["file"]))
    im[:] = (60, 140, 90)   # flat grass
    assert bar_fill(im, CASES[0]["bar"], CASES[0]["badge"]) is None
