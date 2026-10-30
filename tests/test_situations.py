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
