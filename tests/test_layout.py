import numpy as np

from bot77.layout import Layout


def test_alina_layout_regions_inside_frame():
    layout = Layout.load("alina_portrait")
    w, h = layout.frame_size
    for name, (x0, y0, x1, y1) in layout.regions.items():
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, name


def test_box_scales_to_lower_resolution():
    layout = Layout.load("alina_portrait")
    x0, y0, x1, y1 = layout.regions["slot1"]
    assert layout.box("slot1", 540, 1170) == (round(x0 / 2), round(y0 / 2), round(x1 / 2), round(y1 / 2))


def test_crop_shape():
    layout = Layout.load("alina_portrait")
    frame = np.zeros((2340, 1080, 3), np.uint8)
    x0, y0, x1, y1 = layout.regions["timer"]
    assert layout.crop(frame, "timer").shape == (y1 - y0, x1 - x0, 3)
