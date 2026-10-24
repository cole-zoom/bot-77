"""bot77.sim (docs/simulator.md S1): the coordinate map, the card map, plays, and a scenario built from a moment."""
from pathlib import Path

import pytest

from bot77.sim import cards as sc
from bot77.sim.engine import Sim
from bot77.features.frames import FEET_DY, RIVER_Y
from bot77.sim.geometry import engine_to_ours, ours_to_engine

DECK = ["Giant", "Mini P.E.K.K.A", "Musketeer", "Goblin Hut", "Fireball", "Arrows", "Skeletons", "Minions"]


def new_sim() -> Sim:
    deck = [sc.engine_id(n) for n in DECK]
    s = Sim.new(deck, deck)
    s.reset()
    return s


def test_coordinates_round_trip_and_fixed_points():
    for x, y in [(0.0, 0.0), (3.45, 22.57), (9.0, RIVER_Y), (17.9, 27.3)]:
        ex, ey = ours_to_engine(x, y)
        assert engine_to_ours(ex, ey) == pytest.approx((x, y))
    assert ours_to_engine(9.0, RIVER_Y) == pytest.approx((8.5, 16.5))  # river centre
    # princess towers: our box centres (5.94 / 21.28) + their 1.29 box offset -> engine rows 27 / 6
    assert ours_to_engine(3.45, 21.28 + 1.29)[1] == pytest.approx(6.0, abs=0.05)
    assert ours_to_engine(14.5, 5.94 + 1.29)[1] == pytest.approx(27.0, abs=0.05)
    assert ours_to_engine(3.45, 0)[0] == pytest.approx(3.0, abs=0.05)


def test_card_map():
    assert sc.engine_cards()[sc.engine_id("Mini P.E.K.K.A")]["name"] == "Mini PEKKA"
    assert sc.engine_cards()[sc.engine_id("Mini P.E.K.K.A", "hero")]["name"].startswith("Hero Mini")
    assert sc.engine_cards()[sc.engine_id("Knight", "evo")]["evolution"]
    assert sc.engine_id("Goblin Hut") is not None
    assert sc.our_name(sc.engine_id("Mini P.E.K.K.A")) == "Mini P.E.K.K.A"


def test_giant_from_the_bridge_reaches_the_princess_tower():
    """Real Giants (session replays, v5l_e15) stand at box y 7.97 hitting the left princess tower and walk 0.73 of our
    units a second: from our bridge (feet 16.17) that's ~9.5 s to the tower."""
    s = new_sim()
    s.inject(sc.engine_id("Giant"), 3.45, RIVER_Y + FEET_DY, "own")
    stop_t = stop_y = None
    prev = None
    for _ in range(200):
        s.step(1)
        g = next(u for u in s.units() if u["card"] == "Giant")
        if prev is not None and abs(g["y"] - prev) < 1e-4:
            stop_t, stop_y = s.t, g["y"] - FEET_DY
            break
        prev = g["y"]
    assert stop_y == pytest.approx(7.97, abs=0.35)
    assert 8.5 < stop_t < 11.0
    s.step(30)
    assert s.towers()["opp_l"]["hp"] < s.towers()["opp_l"]["max"]


def test_arrows_kill_a_skeleton_group():
    s = new_sim()
    s.inject(sc.engine_id("Skeletons"), 9.0, 10.0, "opp")
    s.step(1)
    assert sum(u["card"] == "Skeletons" for u in s.units()) == 3
    r = s.play(sc.engine_id("Arrows"), 9.0, 10.0, "own")
    assert r.played
    s.step(20)
    assert not any(u["card"] == "Skeletons" and u["team"] == "opp" for u in s.units())


def test_forced_play_puts_card_in_hand_and_tops_up_elixir():
    s = new_sim()
    s.set_elixir("own", 1.0)
    r = s.play(sc.engine_id("Giant"), 9.0, 20.0, "own")
    assert r.played and r.elixir_added > 0
    assert any(u["card"] == "Giant" and u["team"] == "own" for u in s.units())
    r = s.play(sc.engine_id("Musketeer"), 9.0, 12.0, "own", force=False)  # their half: illegal, not forced
    assert not r.played and r.reason
    r = s.play(sc.engine_id("Musketeer"), 9.0, 14.0, "own")  # just past the river: moved back onto our half
    assert r.played and r.moved > 0


def test_snapshot_is_independent():
    s = new_sim()
    s.inject(sc.engine_id("Giant"), 3.45, 20.0, "own")
    s.step(1)  # injected units join the board on the next tick
    y0 = next(u for u in s.units() if u["card"] == "Giant")["y"]
    fork = s.snapshot()
    fork.step(50)
    assert s.tick < fork.tick
    assert next(u for u in s.units() if u["card"] == "Giant")["y"] == y0
    assert next(u for u in fork.units() if u["card"] == "Giant")["y"] < y0 - 2


@pytest.mark.skipif(not Path("data/lancedb/moments.lance").exists(), reason="needs the local LanceDB")
def test_scenario_from_moment():
    from bot77.sim.scenario import scenario_from_moment
    sc_ = scenario_from_moment("tv_2026-10-12_r46_m01:top:179.80:play")
    s = sc_.sim
    assert s.t == pytest.approx(sc_.moment["t_match"] + 0.1, abs=0.06)  # plus the tick that commits the units
    assert sc_.notes["hand_set"]
    assert sc_.notes["engine_multiplier"] == sc_.notes["moment_multiplier"] == 2
    cards = {(u["card"], u["team"]) for u in s.units()}
    assert {("Mega Knight", "own"), ("Mother Witch", "opp"), ("Dart Goblin", "opp")} <= cards
