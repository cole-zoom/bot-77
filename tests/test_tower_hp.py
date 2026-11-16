import importlib.util
import sys
from pathlib import Path

import cv2
import pytest

spec = importlib.util.spec_from_file_location("build_tower_hp", Path(__file__).parents[1] / "scripts/build_tower_hp.py")
th = importlib.util.module_from_spec(spec)
_argv, sys.argv = sys.argv, ["build_tower_hp"]
spec.loader.exec_module(th)
sys.argv = _argv


def test_a_flat_enemy_king_box_is_grown_to_include_its_number():
    x0, y0, x1, y1 = th.crop_box("king-tower-bar", (430, 334, 647, 349))  # bl_2026-10-23_r18 58 s
    assert y1 == 353 and y0 <= 334 - 40
    assert th.crop_box("king-tower-bar", (433, 1583, 645, 1646)) == (429, 1579, 649, 1650)  # ours: as boxed
    assert th.crop_box("tower-bar", (728, 496, 881, 545)) == (724, 492, 885, 549)


@pytest.mark.skipif(sys.platform != "darwin", reason="ocrmac")
def test_the_enemy_kings_hp_reads_from_the_grown_crop():
    im = cv2.imread(str(Path(__file__).parent / "fixtures/tower_hp/opp_king_2091.png"))  # frame [260:380, 400:680]
    x0, y0, x1, y1 = th.crop_box("king-tower-bar", (430 - 400, 334 - 260, 647 - 400, 349 - 260))
    assert th.read_number(im[y0:y1, x0:x1])[0] == 2091
