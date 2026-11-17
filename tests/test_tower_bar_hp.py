"""bot77.readers.tower_hp on synthetic a16_live frames: bars drawn with the colours measured on the phone."""
import cv2
import numpy as np

from bot77.readers.tower_hp import BARS_A16, TowerHP, read_bar

FILL = {"own": (99, 143, 254), "opp": (171, 136, 250)}  # HSV of a bar's fill
TRACK = {"own": (107, 255, 94), "opp": (163, 119, 90)}


def frame_with(bars: dict[str, float], y_opp: int = 350) -> np.ndarray:
    hsv = np.zeros((1920, 1080, 3), np.uint8)
    hsv[...] = (13, 150, 175)  # wooden ground
    for name, frac in bars.items():
        zero, x0, x1, y0, y1 = BARS_A16[name]
        if name.startswith("opp"):
            y0, y1 = y_opp, y_opp + 13
        end = int(round(zero + frac * (x1 + 1 - zero)))
        team = name[:3]
        if not name.endswith("king"):
            hsv[y0:y1, x0:x1 + 1] = TRACK[team]
        hsv[y0:y1, x0:end] = FILL[team]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_fractions_read_back():
    for frac in (1.0, 0.73, 0.4, 0.12):
        im = frame_with({"own_l": frac, "own_r": frac, "own_king": frac})
        for name in ("own_l", "own_r", "own_king"):
            f, present = read_bar(im, name)
            assert present and abs(f - frac) < 0.02, (name, frac, f)


def test_their_bar_found_at_either_height_and_destroyed_after_gone():
    for y in (350, 368):
        tw = TowerHP(every=0.0)
        assert abs(tw.update(frame_with({"opp_l": 1.0, "opp_r": 1.0}, y), 0.0, 10)["opp_r"] - 1.0) < 0.02
        assert abs(tw.update(frame_with({"opp_l": 1.0, "opp_r": 0.6}, y), 0.5, 10)["opp_r"] - 0.6) < 0.02
        assert abs(tw.update(frame_with({"opp_l": 1.0, "opp_r": 0.3}, y), 1.0, 11)["opp_r"] - 0.3) < 0.02
        for t in (1.5, 2.0, 2.5, 3.0):
            out = tw.update(frame_with({"opp_l": 1.0}, y), t, 12)
        assert out["opp_r"] == 0.0 and abs(out["opp_l"] - 1.0) < 0.02
        assert tw.update(frame_with({"opp_l": 1.0, "opp_r": 1.0}, y), 5.0, 14)["opp_r"] == 0.0  # never comes back


def test_kings_full_until_their_bar_shows_and_hp_never_jumps_up():
    tw = TowerHP(every=0.0)
    assert tw.update(frame_with({}), 0.0, 10)["own_king"] == 1.0
    assert abs(tw.update(frame_with({"own_king": 0.8}), 0.5, 10)["own_king"] - 0.8) < 0.02
    assert abs(tw.update(frame_with({"own_king": 1.0}), 1.0, 11)["own_king"] - 0.8) < 0.02  # a one-read jump up


def test_a_fall_too_big_for_one_read_must_hold_a_second():
    tw = TowerHP(every=0.0)
    tw.update(frame_with({"own_l": 1.0}), 0.0, 10)
    assert abs(tw.update(frame_with({"own_l": 0.1}), 0.5, 10)["own_l"] - 1.0) < 0.02
    assert abs(tw.update(frame_with({"own_l": 0.1}), 1.0, 11)["own_l"] - 1.0) < 0.02
    assert abs(tw.update(frame_with({"own_l": 0.1}), 1.5, 11)["own_l"] - 0.1) < 0.02


def test_a_brief_overlay_is_not_a_destroyed_tower():
    """BlueStacks' "No card selected" over a bar for ~0.5 s read it near empty (bot_2026-10-31_012417)."""
    tw = TowerHP(every=0.0)
    tw.update(frame_with({"own_l": 1.0}), 0.0, 10)
    tw.update(frame_with({"own_l": 0.04}), 0.5, 10)
    tw.update(frame_with({"own_l": 0.04}), 1.0, 11)
    for k in range(10):
        f = tw.update(frame_with({"own_l": 1.0}), 1.5 + k * 0.5, 12 + k)
    assert abs(f["own_l"] - 1.0) < 0.02
