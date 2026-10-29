"""The spell guard and the live full-bar guard (docs/plan-elixir-spells.md P1)."""
from pathlib import Path
from types import SimpleNamespace as U

import pytest

import clash_royale_env as cre
from bot77.rl import guards

FIREBALL, ARROWS, SKELETON_ARMY, KNIGHT = 7, 3, 12, 0
ROOT = Path(__file__).resolve().parents[1]


def _engine_with(*units):
    g = cre.ClashRoyaleEnv([2, 81, 5, 43, 9, 95, 7, 3], [2, 81, 5, 43, 9, 95, 7, 3], 3600,
                           cre.TowerTroopType.NONE, cre.TowerTroopType.NONE)
    g.seed(0)
    for card, x, y in units:
        g.inject(card, x, y, 1, -1.0, 0)
    g.step_self_play(4, 0.0, 0.0, 4, 0.0, 0.0, 1)
    return g


def test_radii_come_from_the_engine():
    assert guards.spell_radius(FIREBALL) == pytest.approx(2.5)
    assert guards.spell_radius(ARROWS) == pytest.approx(3.5)


def test_spell_on_empty_grass_is_vetoed():
    g = _engine_with()
    # behind our own towers and in the centre of our side: nothing of theirs there
    assert guards.spell_vetoed(FIREBALL, 8.5, 6.0, g.get_units())
    assert guards.spell_vetoed(ARROWS, 3.0, 4.0, g.get_units())
    # our own units don't count
    assert guards.spell_vetoed(FIREBALL, 3.0, 6.0, [U(team=0, x=3.0, y=6.0, hp=500, tower=False)])


def test_arrows_on_a_skeleton_army_go_through():
    g = _engine_with((SKELETON_ARMY, 4.0, 12.0))
    assert not guards.spell_vetoed(ARROWS, 4.0, 12.0, g.get_units())
    assert not guards.spell_vetoed(ARROWS, 6.5, 12.0, g.get_units())  # off-centre, still within 3.5 + body


def test_fireball_on_an_enemy_tower_goes_through():
    g = _engine_with()
    tower = next(u for u in g.get_units() if u.team == 1 and u.tower and u.x < 8)
    assert not guards.spell_vetoed(FIREBALL, tower.x, tower.y, g.get_units())
    assert not guards.spell_vetoed(FIREBALL, tower.x, tower.y - 3.5, g.get_units())  # 2.5 + the tower's 1.5 reach
    assert guards.spell_vetoed(FIREBALL, tower.x, tower.y - 4.5, g.get_units())


def test_other_cards_are_never_vetoed():
    assert not guards.spell_vetoed(KNIGHT, 8.5, 6.0, [])


def test_clone_env_turns_a_whiffed_spell_into_a_wait(monkeypatch):
    import os
    import sys
    import numpy as np
    monkeypatch.setenv("CLASH_DECK", "2,81,5,43,9,95,7,3")
    monkeypatch.setenv("RL_CLONE_CKPT", str(ROOT / "data/models/bc_v2/frozen.pth"))
    monkeypatch.setenv("RL_PRO_DECKS", str(ROOT / "data/cache/rl/pro_decks.json"))
    monkeypatch.setenv("RL_SITUATION_SHARE", "1")
    if not Path(os.environ["RL_CLONE_CKPT"]).exists():
        pytest.skip("no clone checkpoint (data/ is local)")
    sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
    monkeypatch.chdir(ROOT / "third_party" / "clash-royale-env")
    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    env = CloneEnv({"scenario_seed": 0, "decision_ticks": 1})
    env.forced_template = sit.arrows_value
    env.rng = np.random.default_rng(1)
    env.scenario_rng = env.rng
    env.reset()
    env.game.set_hand_for_team(0, [ARROWS, FIREBALL, 2, 81])
    env.game.set_elixir_for_team(0, 10.0)
    act = {"card_index": 0, "target_x": np.array([8.5], np.float32), "target_y": np.array([1.0], np.float32),
           "activate_ability_slot1": 0, "activate_ability_slot2": 0}
    spent = env.game.get_elixir_spent(0)
    _, _, _, _, info = env.step(act)
    assert info["spell_veto"] == 1.0 and env.game.get_elixir_spent(0) == spent


def test_full_bar_guard_plays_after_three_seconds_calm():
    ckpt = ROOT / "data/models/bc_defence_v2/epoch03_cal50.pth"
    if not ckpt.exists():
        pytest.skip("no school checkpoint (data/ is local)")
    import lancedb
    from bot77.live.selfplay import SelfPlayPolicy
    from bot77.live.state import LiveState
    cards = lancedb.connect(str(ROOT / "data/lancedb")).open_table("cards").search().where(
        "kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
    by = {c["name"]: c["id"] for c in cards}
    hand = [by["Giant"], by["Mini P.E.K.K.A"] if "Mini P.E.K.K.A" in by else by["Mini PEKKA"], by["Bomber"], by["Mega Minion"]]
    pol = SelfPlayPolicy(cards, str(ckpt), sample=False, decision_s=0.1, full_bar_guard=True)  # greedy: always waits
    acts = []
    for i in range(50):  # 5 s at 10 elixir, empty board
        t = 60.0 + i / 10
        st = LiveState(t, hand, by["Goblin Hut"], 10.0, 120, 60, 1, "regulation", [])
        acts.append((t, pol.decide(st, [])))
    played = [(t, a) for t, a in acts if a is not None]
    assert played, "the guard never fired"
    assert played[0][0] >= 63.0 - 1e-6 and played[0][1].reason == "full-bar guard"
    off = SelfPlayPolicy(cards, str(ckpt), sample=False, decision_s=0.1)
    assert all(off.decide(LiveState(60.0 + i / 10, hand, by["Goblin Hut"], 10.0, 120, 60, 1, "regulation", []), []) is None
               for i in range(50))
