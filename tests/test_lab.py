"""Lab battle scripts (config/lab/*.yaml, bot77.lab) and the compare helpers (bot77.sim.labcmp)."""
import importlib.util
from pathlib import Path

import pytest

from bot77 import lab

ROOT = Path(__file__).resolve().parents[1]


def test_tests_load_in_order():
    ts = lab.load_tests(ROOT / "config/lab")
    assert len(ts) == 20
    assert list(ts) == sorted(ts)
    for t in ts.values():
        assert t.placements and sorted(t.decks["bluestacks"]) == sorted(lab.BOMBER_DECK)
        assert len(set(t.decks["cole"])) == 8


def test_cole_tile_views():
    # princess towers map onto princess towers either way; rotate swaps the lanes, flip keeps them
    assert lab.cole_tile((3, 27), "rotate") == (14, 6)
    assert lab.cole_tile((3, 27), "flip") == (3, 6)
    assert lab.cole_tile((3, 20), "rotate") == (14, 13)
    assert "right tower's column" in lab.own_tile_words((14, 13))
    assert lab.clock_text(30) == "2:30"


def test_placements_legal_in_engine():
    cre = pytest.importorskip("clash_royale_env")  # noqa: F841
    from bot77.sim import cards as sc
    from bot77.sim.engine import Sim
    ts = lab.load_tests(ROOT / "config/lab")
    t0 = next(iter(ts.values()))
    sim = Sim.new([sc.engine_id(n) for n in t0.decks["bluestacks"]], [sc.engine_id(n) for n in t0.decks["cole"]])
    sim.reset()
    for t in ts.values():
        for p in t.placements:
            card = sc.engine_id(p.card)
            assert card is not None, p.card
            team = 0 if p.side == "bluestacks" else 1
            assert sim.env.is_valid_placement(card, float(p.tile[0]), float(p.tile[1]), team), (t.id, p)
        for kind, tile in t.dump_spots().items():
            card = sc.engine_id("Arrows" if kind == "spell" else "Goblin Hut")
            assert sim.env.is_valid_placement(card, float(tile[0]), float(tile[1]), 0), (t.id, kind, tile)


def _lab_play():
    spec = importlib.util.spec_from_file_location("lab_play", ROOT / "scripts/lab_play.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_game_clock_anchors_on_the_second_turning():
    c = _lab_play().GameClock()
    c.update(10, 100.0)
    assert c.t(100.0) == pytest.approx(10.5)
    c.update(10, 100.3)
    c.update(11, 100.6)  # the second turned here
    assert c.t(101.0) == pytest.approx(11.4)
    c.update(30, 101.2)  # a jump (a dropped stretch of frames): re-anchor
    assert c.t(101.2) == pytest.approx(30.5)


def test_body_events():
    pytest.importorskip("clash_royale_env")
    from bot77.sim.labcmp import _events
    ts = [i * 0.25 for i in range(20)]
    c = [0, 0, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0]
    e = _events(ts, c)
    assert e["first"] == 0.5 and e["max"] == 3 and e["drops"] == [1.5, 3.25, 3.25] and e["last"] == 3.0
    flicker = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  # a one-frame gap is not a death
    assert _events(ts[:10], flicker)["drops"] == []
