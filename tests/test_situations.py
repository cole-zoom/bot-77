"""The situations' scripted opponent and full-game runs (bot77.rl.situations, bot77.rl.clone_env)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("clash_royale_env")

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"


@pytest.fixture(scope="module")
def mods():
    if not (ROOT / "data/models/bc_v2/frozen.pth").exists() or not (ROOT / "data/cache/rl/pro_decks.json").exists():
        pytest.skip("the frozen clone and the pro deck pool are local data")
    os.environ.update({"CLASH_DECK": "2,81,5,43,9,95,7,3", "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    sys.path.insert(0, str(FORK))
    cwd = os.getcwd()
    os.chdir(FORK)
    import python_ai  # noqa: F401
    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    yield sit, CloneEnv
    os.chdir(cwd)


def act(k, x, y):
    return {"card_index": k, "target_x": np.array([x], np.float32), "target_y": np.array([y], np.float32),
            "activate_ability_slot1": 0, "activate_ability_slot2": 0}


def start(sit, CloneEnv, tpl, seed, **cfg):
    env = CloneEnv({"scenario_seed": 0, "decision_ticks": 1, **cfg})
    env.forced_template = tpl
    env.rng = np.random.default_rng(seed)
    env.scenario_rng = env.rng
    env.reset()
    return env


@pytest.mark.parametrize("name", ["dont_overcommit", "defend_counter", "safe_opening", "punish"])
def test_the_scripted_opponent_only_plays_cards_it_holds(mods, name):
    # Cole (2026-10-30): "how is this bot playing 3 Mini pekkas in a row??" The script used to put the card it wanted
    # into its hand; now it plays from its own cycle (the card or one that does the same job) or waits.
    sit, CloneEnv = mods
    plays = 0
    for seed in range(4):
        env = start(sit, CloneEnv, sit.BY_NAME[name], seed)
        played: set = set()
        while env.situation is not None:
            hand = list(env.game.get_hand_for_team(1))
            n = len(env.situation.opp_state.get("plays", []))
            situation = env.situation
            _, _, term, trunc, info = env.step(act(*sit.tempo_dump(env, played)))
            for p in situation.opp_state.get("plays", [])[n:]:
                assert p[1] in hand, (name, seed, p, hand)
                plays += 1
            if info.get("rl_situation_end") or term or trunc:
                break
    assert plays > 0  # it did play


def play_out(env, sit, policy="scripted"):
    """Steps to the episode's end: [(reward, info, terminated or truncated)] per step."""
    steps, played = [], set()
    while True:
        if policy == "scripted" and env.situation is not None:
            k, x, y = sit.scripted_defence(env)
        else:
            k, x, y = 4, 0.0, 0.0
        _, r, term, trunc, info = env.step(act(k, x, y))
        steps.append((float(r), {k2: float(v) for k2, v in info.items() if k2.startswith("rl_situation")}, term or trunc))
        if term or trunc:
            return steps


@pytest.mark.parametrize("name", ["chill", "press_calm", "desperate_even", "desperate", "spell_finish"])
def test_a_tempo_start_plays_to_the_end_of_the_game(mods, name):
    # Cole (2026-10-30): "These tempos should start out in these situatins, and last the rest of teh game". Every step
    # is a tempo step; the one scored end is the game's own; the rewards add up to the score, the result included.
    sit, CloneEnv = mods
    env = start(sit, CloneEnv, sit.BY_NAME[name], 3)
    assert env.situation.spec["opp"] is None and list(env.opp_deck) != list(sit.SCRIPT_DECK)  # the pro clone
    steps = play_out(env, sit)
    assert all(i["rl_situation"] == 1 for _, i, _ in steps)
    ends = [n for n, (_, i, _) in enumerate(steps) if i["rl_situation_end"]]
    assert ends == [len(steps) - 1] and steps[-1][1]["rl_situation_final"] == 1
    r = env.last_situation
    assert r["full_game"] and r["result"] in (-1.0, 0.0, 1.0) and r["parts"]["result"] == r["result"]
    assert sum(s[0] for s in steps) == pytest.approx(r["score"], abs=1e-4)
    assert sum(r["parts"].values()) == pytest.approx(r["score"], abs=1e-3)
    if name == "desperate":  # pass: reached overtime, or won
        assert r["passed"]["desperate"] == (r["clock"] > 180.0 or r["result"] > 0)
    if name == "spell_finish":
        assert "spell_finish" in r["passed"]


def test_a_defence_window_scores_then_the_game_plays_on(mods):
    # Cole's brother: the windows "make the bot think the scenario is over/game is over". The score is paid when the
    # window ends; the game goes on as an ordinary one (its shaping back on) and only its own end is final.
    sit, CloneEnv = mods
    env = start(sit, CloneEnv, sit.BY_NAME["hog_rider"], 5)
    steps = play_out(env, sit)
    ends = [n for n, (_, i, _) in enumerate(steps) if i["rl_situation_end"]]
    assert len(ends) == 1 and ends[0] < len(steps) - 1
    assert steps[ends[0]][1]["rl_situation_final"] == 0 and not steps[ends[0]][2]
    assert all(i["rl_situation"] == 0 for _, i, _ in steps[ends[0] + 1:])
    assert abs(steps[-1][0]) == 1.0 or steps[-1][0] == 0.0  # the game's own result


def test_without_full_games_the_window_ends_the_episode(mods):
    sit, CloneEnv = mods
    env = start(sit, CloneEnv, sit.BY_NAME["hog_rider"], 5, situation_full_game=False)
    steps = play_out(env, sit)
    assert steps[-1][1]["rl_situation_end"] == 1 and steps[-1][1]["rl_situation_final"] == 1
    assert env.game.get_current_tick() < 600  # well short of the game's end
