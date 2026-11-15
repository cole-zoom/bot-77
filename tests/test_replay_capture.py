import cv2
import numpy as np

from bot77.replay_capture import device_of, red_bar_rows


def _frame(bar_y: int) -> np.ndarray:
    f = np.full((1920, 1080, 3), (60, 90, 120), np.uint8)   # brownish arena
    for x in (250, 750):
        cv2.rectangle(f, (x, bar_y), (x + 105, bar_y + 16), (40, 40, 220), -1)   # red HP bar (BGR)
    return f


def test_the_red_bar_row_tells_the_device():
    assert red_bar_rows(_frame(499)) == [499, 499]
    assert device_of([_frame(499)]) == "bluestacks"   # BlueStacks replay viewer (bs replay_03, 12 s)
    assert device_of([_frame(594)]) == "a16"          # Galaxy A16 (bl_2026-11-15 r01, 20 s)
    assert device_of([np.zeros((1920, 1080, 3), np.uint8)]) is None
