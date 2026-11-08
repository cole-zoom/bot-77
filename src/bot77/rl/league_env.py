"""Self-play against a league (2026-11-08, Cole: "make the bot play against itself soon"): whole games, our deck,
team 1 drawn per episode from a fixed pool instead of the pro clone alone.

The pool (RL_LEAGUE, a JSON list; scripts/selfplay_league.py builds it) holds three kinds of member, each with a weight:
  {"kind": "ckpt", "path": ".../epoch03.pth"}  a frozen checkpoint on our deck, deciding every 0.1 s (the school
                                               models were trained at 0.1 s; CloneEnv's league asked once a second,
                                               which handicapped them: the 2026-10-30 match_eval finding)
  {"kind": "deck", "key": "pekka_bs"}          a scripted deck bot (bot77.rl.decks) on its own deck, ticked every
                                               0.1 s and told each of our plays that lands, as deck_vs_model.py
  {"kind": "clone"}                            the pro clone on a pro deck, once a second (CloneEnv's default)
plus, with weight RL_SNAPSHOT_WEIGHT, the run's own snapshots (RL_SNAPSHOT_DIRS), as checkpoints at 0.1 s.

No situations, no scenarios. The reward is the game's result alone (engine +1 / -1 / 0) plus RL_CROWN_COEF per crown
of margin at the end; the trainer's shaping is switched off by the script. RL rounds v3-v6 all sank under the fork's
shaping; Cole's live record only counts wins.

info["league_opp"] is the member's index (snapshots: len(pool)) on the episode's last step, -1 before.
"""
from __future__ import annotations

import json
import os

import numpy as np

import clash_royale_env

from bot77.rl.clone_env import CloneEnv


class LeagueEnv(CloneEnv):
    def __init__(self, env_config=None):
        env_config = dict(env_config or {})
        env_config["scenarios_enabled"] = False  # whole games only (the fork injects its scenarios otherwise)
        env_config.setdefault("decision_ticks", 1)
        super().__init__(env_config)
        self.pool = json.loads(os.environ["RL_LEAGUE"])
        self.snapshot_weight = float(os.environ.get("RL_SNAPSHOT_WEIGHT", "0"))
        self.crown_coef = float(os.environ.get("RL_CROWN_COEF", "0.1"))
        self.bot = None
        self.member = -1
        import bot77.rl.decks  # noqa: F401  (registers the deck bots)
        from bot77.rl.deck_bots import DECKS
        self.DECKS = DECKS

    def _draw(self) -> tuple[int, dict]:
        snaps = self._snapshots() if self.snapshot_weight > 0 else []
        w = [float(m.get("w", 1.0)) for m in self.pool] + ([self.snapshot_weight] if snaps else [])
        i = int(self.rng.choice(len(w), p=np.array(w) / sum(w)))
        if i == len(self.pool):
            return i, {"kind": "ckpt", "path": snaps[int(self.rng.integers(len(snaps)))]}
        return i, self.pool[i]

    def _sample_pfsp_opponent(self):
        self.member, m = self._draw()
        self.opponent_kind = "neural"
        self.bot = None
        none = clash_royale_env.TowerTroopType.NONE
        if m["kind"] == "deck":
            seed = int(self.rng.integers(2 ** 31))
            self.bot = self.DECKS[m["key"]](seed)
            self.opp_deck = list(self.bot.deck_ids)
            self.opponent_checkpoint_path = None  # no PFSP bookkeeping for scripted members
        else:
            path = m["path"] if m["kind"] == "ckpt" else self.clone
            if m["kind"] == "ckpt":
                self.opp_deck = list(self.deck)
            else:
                self.opp_deck = self.pro_decks[int(self.rng.choice(len(self.pro_decks), p=self.deck_p))]
            if path != self.loaded:
                self.set_historical_opponent(path)
                self.loaded = path
            self.opponent_checkpoint_path = path
        # a checkpoint decides every 0.1 s, the clone once a second
        self.opp_every = 1 if m["kind"] == "ckpt" else max(1, 10 // self.decision_ticks)
        self.game = clash_royale_env.ClashRoyaleEnv(self.deck, self.opp_deck, self.max_ticks, none, none)
        self.game.seed(int(self.rng.integers(2 ** 31)))
        if self.bot is not None:
            self.bot.bind(self.game)

    def _opponent_action(self):
        if self.bot is not None:
            return (*(self.bot.tick() or (4, 0.0, 0.0)), False, False)
        return super()._opponent_action()

    def step(self, action, skip_frames=None):
        k = int(np.asarray(action["card_index"]).reshape(-1)[0])
        hand = list(self.game.get_hand())
        spent = float(self.game.get_elixir_spent(0))
        obs, reward, terminated, truncated, info = super().step(action, skip_frames)
        if self.bot is not None and k < 4 and k < len(hand) and float(self.game.get_elixir_spent(0)) > spent + 0.5:
            self.bot.note_play(hand[k], float(np.asarray(action["target_x"]).reshape(-1)[0]),
                               float(np.asarray(action["target_y"]).reshape(-1)[0]))
        info["league_opp"] = -1.0
        info["league_margin"] = 0.0
        if not (terminated or truncated):
            reward = 0.0
        else:
            g = self.game
            alive = [[g.get_tower_hp(t, s) > 0 for s in range(3)] for t in (0, 1)]
            margin = (3 - sum(alive[1])) - (3 - sum(alive[0]))  # crowns for - crowns against
            reward = float(np.sign(reward)) + self.crown_coef * margin
            info["league_opp"] = float(self.member)
            info["league_margin"] = float(margin)
        return obs, reward, terminated, truncated, info


def make_league_env(seed=None, **env_config):
    """A picklable worker factory (AsyncVectorEnv spawns workers that import this module)."""
    config = dict(env_config)
    if seed is not None:
        config["scenario_seed"] = seed

    def _thunk():
        return LeagueEnv(config)
    return _thunk
