"""Reinforcement learning against pro clones (option 3 step 2, 2026-10-27; AlphaStar's recipe: imitation first, then RL
against the imitation agents).

Team 1 is the frozen imitation agent (scripts/train_bc.py: the fork's MicroRoyaleNet cloned from pro replays)
playing a pro's deck, drawn per episode from data/cache/rl/pro_decks.json (scripts/pro_decks.py) in proportion to how
many players used it. Everything else is the ClashRoyaleEnv fork's phase-2 environment (python_ai/envs/selfplay_env.py,
MIT): the reward, the scenario injection, team 1 deciding from its own mirrored observation. The fork's python_ai/ is
left untouched (its notes: a live run may depend on it); this subclasses it.

Optionally a league (AlphaStar's answer to over-fitting one opponent: against the single clone the win rate climbed
to ~78% and flattened): with probability RL_LEAGUE_SHARE team 1 is instead one of the trainee's own past snapshots
(the .pth files in RL_SNAPSHOT_DIRS, os.pathsep-separated), drawn by the fork's PFSP weighting (the ones it beats
least most often), playing our deck, the one it was trained with.

Paths come from the environment (set by scripts/rl_vs_clones.py before the fork is imported): RL_CLONE_CKPT (the
frozen clone), RL_PRO_DECKS (the deck pool).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import clash_royale_env
from python_ai.envs.selfplay_env import PFSP_EXPONENT, PFSP_MIN_WEIGHT, MicroRoyaleSelfPlayEnv


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
        self.league_share = float(os.environ.get("RL_LEAGUE_SHARE", "0"))
        self.snapshot_dirs = [d for d in os.environ.get("RL_SNAPSHOT_DIRS", "").split(os.pathsep) if d]
        self.set_historical_opponent(self.clone)
        self.loaded = self.clone  # the weights in opponent_net (a snapshot load is skipped when it's already there)
        self.opponent_kind = "neural"
        self.opp_deck: list[int] | None = None

    def _snapshots(self) -> list[str]:
        return sorted(p for d in self.snapshot_dirs for p in glob.glob(os.path.join(d, "*.pth")))

    def _sample_pfsp_opponent(self):
        """The clone with a pro deck, or (league) a past snapshot with our deck; a new draw each episode (the engine
        is rebuilt for it: decks are fixed at construction)."""
        self.opponent_kind = "neural"
        snaps = self._snapshots() if self.league_share > 0 else []
        if snaps and self.rng.random() < self.league_share:
            w = np.array([max(PFSP_MIN_WEIGHT, (1.0 - self.pfsp_stats.get(p, 0.5)) ** PFSP_EXPONENT) for p in snaps])
            path = snaps[int(self.rng.choice(len(snaps), p=w / w.sum()))]
            self.opp_deck = list(self.deck)
        else:
            path = self.clone
            self.opp_deck = self.pro_decks[int(self.rng.choice(len(self.pro_decks), p=self.deck_p))]
        if path != self.loaded:
            self.set_historical_opponent(path)
            self.loaded = path
        self.opponent_checkpoint_path = path  # the key the fork's win-rate estimate (pfsp_stats) is kept under
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
