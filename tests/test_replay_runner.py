"""The replay runner gets past screens with nothing safe to press (2026-11-15, collect1 chunk 2: a chest opening)."""
import importlib.util
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHEST = ROOT / "data/raw/live_ui/unknown/replay_list_fail_1794736194.png"


def _runner():
    spec = importlib.util.spec_from_file_location("replay_runner", ROOT / "scripts/replay_runner.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_blind_screens_alternate_back_and_the_nudge():
    r = _runner()
    assert [r.blind_action(n) for n in range(1, 5)] == ["back", "nudge", "back", "nudge"]


@pytest.mark.skipif(not CHEST.exists(), reason="needs the saved chest screen")
def test_a_chest_opening_is_tapped_not_only_backed_out_of(monkeypatch):
    r = _runner()
    chest, log_screen = cv2.imread(str(CHEST)), None
    taps, backs = [], []

    class FakeDevice:
        s = "fake"
        def cap(self):
            return log_screen if taps else chest   # the chest opens on the first tap
        def tap(self, x, y):
            taps.append((x, y))

    log_screen = chest.copy()
    monkeypatch.setattr(r.time, "sleep", lambda s: None)
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **k: backs.append(a))
    monkeypatch.setattr(r, "seen", lambda im, name, slack=40: name == r.TITLE and im is log_screen)
    monkeypatch.setattr(r, "at_end", lambda im: False)
    r.TITLE = "log_title"
    assert r.open_log(FakeDevice()) is log_screen
    assert taps == [r.NUDGE] and len(backs) == 1
