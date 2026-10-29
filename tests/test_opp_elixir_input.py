"""Plan P5: the opponent's estimated elixir as observation scalar 10 (docs/plan-elixir-spells.md).

Needs the engine build with the input (`set_elixir_spent`); skipped on an older build.
"""
import numpy as np
import pytest

import clash_royale_env as cre
from bot77.sim import cards as sc
from bot77.sim.engine import has_opp_elixir_input
from bot77.sim.mirror import mirror, observation

pytestmark = pytest.mark.skipif(not has_opp_elixir_input(), reason="engine build without the P5 input")
DECK = [2, 81, 5, 43, 9, 95, 7, 3]


def _est_index() -> int:
    return cre.ClashRoyaleEnv.EXTRA_SCALARS_START + 10


def test_the_mirror_shows_the_opponent_estimate_it_was_given():
    hand = DECK[:4]
    for t, est in ((10.0, 2.0), (95.0, 7.5), (150.0, 0.0)):
        sim = mirror(DECK, DECK, t, hand, 6.0, est, [], None, [])
        obs = observation(sim)
        # one tick of regen after the commit step: 0.035 x the phase multiplier
        assert obs[_est_index()] == pytest.approx(min(10.0, est + 0.035 * cre.elixir_multiplier_at_tick(sim.tick)) / 10,
                                                  abs=2e-3)
        # and our own bar's estimate (what the opponent's view sees) is our elixir
        assert sim.env.get_elixir_estimate(0) == pytest.approx(sim.env.get_elixir_for_team(0), abs=1e-3)


def test_an_unclipped_live_estimate_reads_clipped():
    sim = mirror(DECK, DECK, 60.0, DECK[:4], 5.0, 13.0, [], None, [])
    assert observation(sim)[_est_index()] == pytest.approx(1.0)
    sim = mirror(DECK, DECK, 60.0, DECK[:4], 5.0, -2.0, [], None, [])
    assert observation(sim)[_est_index()] == pytest.approx(0.0035 * cre.elixir_multiplier_at_tick(sim.tick), abs=1e-3)


def test_the_tempo_pair_halves_differ_in_the_estimate():
    from bot77.rl import situations as sit
    seen = {}
    for name in ("chill", "press_calm"):
        tpl = sit.BY_NAME[name]
        spec = tpl(np.random.default_rng(3))
        g = cre.ClashRoyaleEnv(DECK, DECK, 3600, cre.TowerTroopType.NONE, cre.TowerTroopType.NONE)
        g.seed(0)
        g.reset()
        s = sit.Situation(g, spec, DECK, np.random.default_rng(3))
        obs = np.asarray(g.get_observation_for_team(0), dtype=np.float32)
        seen[name] = (obs[_est_index()], spec["opp_elixir"])
        # their bar as the situation left it (the spec's, plus any tick of regen its set-up stepped)
        assert obs[_est_index()] == pytest.approx(min(10.0, g.get_elixir_for_team(1)) / 10, abs=1e-4)
        assert abs(g.get_elixir_for_team(1) - spec["opp_elixir"]) < 0.1
        assert s.spent0 == pytest.approx(float(g.get_elixir_spent(0)))
    assert seen["chill"][0] >= 0.8 and seen["press_calm"][0] <= 0.3


def test_a_full_bar_at_tick_zero_reads_full():
    """A defence template starts at tick 0 with their bar at 10: the spend goes negative so the estimate says 10."""
    from bot77.sim.engine import sync_spent_to_bars
    g = cre.ClashRoyaleEnv(DECK, DECK, 3600, cre.TowerTroopType.NONE, cre.TowerTroopType.NONE)
    g.seed(0)
    g.reset()
    g.set_elixir_for_team(1, 10.0)
    assert g.get_elixir_estimate(1) == pytest.approx(5.0)
    assert sync_spent_to_bars(g)
    assert g.get_elixir_spent(1) == pytest.approx(-5.0)
    assert g.get_elixir_estimate(1) == pytest.approx(10.0)


def test_a_real_play_lowers_the_estimate_by_its_cost():
    sim = mirror(DECK, DECK, 30.0, DECK[:4], 8.0, 8.0, [], None, [])
    before = sim.env.get_elixir_estimate(1)
    card = next(c for c in DECK if not sc.engine_cards()[c].get("is_spell", False) and sc.cost(c) <= 8)
    assert sim.env.set_hand_for_team(1, [card] + [c for c in DECK if c != card][:3])
    sim.env.step_self_play(4, 0.0, 0.0, 0, 9.0, 8.0, 1)
    assert sim.env.get_elixir_spent(1) > 0  # it was played
    assert sim.env.get_elixir_estimate(1) == pytest.approx(before + 0.035 - sc.cost(card), abs=1e-3)
