"""Reinforcement learning against pro clones (option 3 step 2, 2026-10-27; AlphaStar's recipe: imitation first, then RL
against the imitation agents).

Team 1 is always the frozen imitation agent (scripts/train_bc.py: the fork's MicroRoyaleNet cloned from pro replays)
playing a pro's deck, drawn per episode from data/cache/rl/pro_decks.json (scripts/pro_decks.py) in proportion to how
many players used it. Everything else is the ClashRoyaleEnv fork's phase-2 environment (python_ai/envs/selfplay_env.py,
MIT): the reward, the scenario injection, team 1 deciding from its own mirrored observation. The fork's python_ai/ is
left untouched (its notes: a live run may depend on it); this subclasses it.

Paths come from the environment (set by scripts/rl_vs_clones.py before the fork is imported): RL_CLONE_CKPT (the
frozen clone), RL_PRO_DECKS (the deck pool).
"""
from __future__ import annotations

import json
import os

import numpy as np

import clash_royale_env
from python_ai.envs.selfplay_env import MicroRoyaleSelfPlayEnv


class CloneEnv(MicroRoyaleSelfPlayEnv):
    def __init__(self, env_config=None):
        env_config = dict(env_config or {})
        super().__init__(env_config)
        self.max_ticks = env_config.get("max_ticks", 3600)
        self.clone = os.environ["RL_CLONE_CKPT"]
        decks = json.loads(open(os.environ["RL_PRO_DECKS"]).read())
        self.pro_decks = [d["cards"] for d in decks]
        w = np.array([d["players"] for d in decks], dtype=float)
        self.deck_p = w / w.sum()
        self.set_historical_opponent(self.clone)
        self.opponent_kind = "neural"
        self.opp_deck: list[int] | None = None

    def _sample_pfsp_opponent(self):
        """Always the clone; a new pro deck each episode (the engine is rebuilt for it: decks are fixed at
        construction)."""
        self.opponent_kind = "neural"
        self.opponent_checkpoint_path = self.clone
        self.opp_deck = self.pro_decks[int(self.rng.choice(len(self.pro_decks), p=self.deck_p))]
        none = clash_royale_env.TowerTroopType.NONE
        self.game = clash_royale_env.ClashRoyaleEnv(self.deck, self.opp_deck, self.max_ticks, none, none)


def make_clone_env(seed=None, **env_config):
    """A picklable worker factory (AsyncVectorEnv spawns workers that import this module)."""
    config = dict(env_config)
    if seed is not None:
        config["scenario_seed"] = seed

    def _thunk():
        return CloneEnv(config)
    return _thunk
