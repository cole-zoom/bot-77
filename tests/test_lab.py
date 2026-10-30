"""Lab battle scripts (config/lab/*.yaml, bot77.lab) and the compare helpers (bot77.sim.labcmp)."""
import importlib.util
from pathlib import Path

import pytest

from bot77 import lab

ROOT = Path(__file__).resolve().parents[1]


def test_tests_load_in_order():
    ts = lab.load_tests(ROOT / "config/lab")
    assert len(ts) == 24
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


def test_clock_text_overtime():
    assert lab.clock_text(150) == "0:30"
    assert lab.clock_text(198) == "OT 1:42"  # L20b ran here; the old clock read 0:00
    assert lab.clock_text(400) == "OT 0:00"


def test_shifted_moves_every_placement():
    t = lab.load_tests(ROOT / "config/lab")["L21"]
    s = t.shifted(121.4)
    assert [p.t for p in s.placements] == pytest.approx([p.t + 121.4 for p in t.placements])
    assert [p.t for p in t.placements] == [25.0, 29.5]  # the original is untouched
    assert s.end_t == pytest.approx(t.end_t + 121.4)


def _runner(test_id: str):
    """A LabRunner without a device or reader: just the plan and the tap bookkeeping."""
    from types import SimpleNamespace
    lp = _lab_play()
    t = lab.load_tests(ROOT / "config/lab")[test_id]
    r = lp.LabRunner.__new__(lp.LabRunner)
    r.test, r.dry_run, r.log = t, False, lambda *_: None
    r.queue, r.cole, r.end_t, r.shift, r.pending = t.side("bluestacks"), t.side("cole"), t.end_t, None, None
    r.cost = {1: 4.0}
    return lp, r, SimpleNamespace


def test_late_join_moves_the_plan():
    lp, r, _ = _runner("L20b")  # Mini P.E.K.K.A at 30, Ice Spirit at 40
    r.rebase(5.0)
    assert r.shift == 0.0 and r.queue[0].t == 30.0
    lp, r, _ = _runner("L20b")
    r.rebase(198.5)  # joined in overtime
    assert r.shift == pytest.approx(198.5 + lp.JOIN_LEAD_S - 30.0)
    assert r.queue[0].t == pytest.approx(198.5 + lp.JOIN_LEAD_S) and r.cole[0].t == pytest.approx(r.queue[0].t + 10)


def test_a_tap_counts_only_once_seen():
    lp, r, NS = _runner("L02")
    sched = r.queue[0]
    row = {"card": "Mini P.E.K.K.A"}
    r.pending = {"row": row, "card": 1, "slot": 2, "elixir": 9.9, "at": 100.0, "sched": sched}
    assert r.confirm(NS(hand=[5, 6, 1, 7], elixir=9.9), 100.5)  # nothing yet: wait
    assert r.queue and row.get("taken") is None
    assert not r.confirm(NS(hand=[5, 6, 1, 7], elixir=9.9), 100.0 + lp.CONFIRM_S)  # never showed: failed, retry
    assert row["taken"] is False and r.queue[0] is sched
    row2 = {"card": "Mini P.E.K.K.A"}
    r.pending = {"row": row2, "card": 1, "slot": 2, "elixir": 9.9, "at": 103.0, "sched": sched}
    assert not r.confirm(NS(hand=[5, 6, 8, 7], elixir=9.6), 103.6)  # the card left its slot: taken
    assert row2["taken"] is True and not r.queue


def test_session_map_points_at_lab_replays():
    import json
    import yaml
    tests = lab.load_tests(ROOT / "config/lab")
    lab_ids = set(json.loads((ROOT / "config/lab_replays.json").read_text())["replays"])
    for f in sorted((ROOT / "config/lab/sessions").glob("*.yaml")):
        for r in yaml.safe_load(f.read_text())["runs"]:
            assert r["test"] in tests, (f.name, r)
            assert r["replay"].removesuffix("_m01") in lab_ids, (f.name, r)  # never a ladder game
            assert r["status"] in ("ok", "void", "unread")
