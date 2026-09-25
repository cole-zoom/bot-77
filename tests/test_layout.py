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


def test_segment_accepts_jump_cuts_and_splits_new_matches():
    from bot77.readers.clock import ClockRead
    from bot77.segment import segment_matches

    def read(el):
        return ClockRead("regulation", 180 - el, 1, el)

    reads = [(t, read(10 + t)) for t in range(0, 40)]  # match A
    reads += [(40 + t, read(62 + t)) for t in range(0, 30)]  # jump cut: 12 s of footage removed
    reads += [(70, read(150))]  # a lone misread
    reads += [(71 + t, read(92 + t)) for t in range(0, 20)]
    reads += [(95 + t, read(3 + t)) for t in range(0, 30)]  # match B
    matches = segment_matches(reads)
    assert len(matches) == 2
    a, b = matches
    assert len(a.reads) == 90 and a.elapsed_at(50) == 72
    assert b.reads[0][1].elapsed_s == 3
