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

Optionally defence situations (bot77.rl.situations, Cole's scenarios): with probability RL_SITUATION_SHARE an episode
starts mid-fight and ends when the threat resolves, scored on the outcome (info["rl_situation"] marks its steps,
"rl_situation_end" the scored one; the trainer turns the game's shaping off on those steps and treats the end as
terminal). When on, the fork's own scenarios are off.

Decision rate (2026-10-28, Cole: "0.1 decisions"): our side acts every `decision_ticks` engine ticks (env_config, or
RL_DECISION_TICKS; 10 = once a second, the fork's rate; 1 = every 0.1 s). The opponent keeps deciding once a second
whatever ours is: the clone and the snapshots so far were trained at that rate, and asked every tick they would play
ten times as often.

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

from bot77.rl import situations as sit


class CloneEnv(MicroRoyaleSelfPlayEnv):
    def __init__(self, env_config=None):
        env_config = dict(env_config or {})
        self.situation_share = float(os.environ.get("RL_SITUATION_SHARE", "0"))
        if self.situation_share > 0:
            env_config["scenarios_enabled"] = False  # ours replace the fork's
        super().__init__(env_config)
        self.decision_ticks = int(env_config.get("decision_ticks", os.environ.get("RL_DECISION_TICKS", "10")))
        self.opp_every = max(1, 10 // self.decision_ticks)  # our steps between two opponent decisions (1 s)
        self.opp_calls = 0
        self.situation: sit.Situation | None = None
        self.forced_template = None  # the exam sets one
        # Cole's reference starts (the play-page games where he beat the scripted defence): this share of situation
        # episodes replays one of those exact starts (template + seed), so the bot trains on the boards he won
        self.reference_share = float(os.environ.get("RL_REFERENCE_SHARE", "0"))
        ref = os.environ.get("RL_REFERENCE_FILE", "")
        self.references = [json.loads(l) for l in open(ref)] if ref and os.path.exists(ref) else []
        self.forced_seed = None  # (template, seed) to reproduce exactly; the exam's second yardstick sets it
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

    def _opponent_action(self):
        """The opponent decides once a second (every opp_every of our steps); in between it waits."""
        self.opp_calls += 1
        if (self.opp_calls - 1) % self.opp_every:
            return 4, 0.0, 0.0, False, False
        return super()._opponent_action()

    def reset(self, seed=None, options=None):
        fixed = self.forced_seed
        if fixed is None and self.references and self.situation_share > 0 and self.rng.random() < self.situation_share \
                and self.rng.random() < self.reference_share:
            r = self.references[int(self.rng.integers(len(self.references)))]
            fixed = (sit.BY_NAME[r["situation"]], int(r["seed"]))
        main_rng = None
        if fixed is not None:
            # the play page's start: the generator seeded before reset (opponent deck, situation and hand all follow)
            main_rng, main_srng = self.rng, getattr(self, "scenario_rng", None)
            self.rng = np.random.default_rng(fixed[1])
            self.scenario_rng = self.rng
        obs, info = super().reset(seed=seed, options=options)
        self.opp_calls = 0
        self.situation = None
        if fixed is not None or self.forced_template is not None or \
                (self.situation_share > 0 and self.rng.random() < self.situation_share):
            tpl = fixed[0] if fixed is not None else \
                (self.forced_template or sit.TEMPLATES[int(self.rng.integers(len(sit.TEMPLATES)))])
            self.situation = sit.Situation(self.game, tpl(self.rng), list(self.deck), self.rng,
                                           dt=self.decision_ticks / 10.0)
            obs = np.array(self.game.get_observation_for_team(0), dtype=np.float32)
        if main_rng is not None and self.forced_seed is None:
            self.rng, self.scenario_rng = main_rng, main_srng  # back to the env's own stream
        return obs, info

    def step(self, action, skip_frames=None):
        obs, reward, terminated, truncated, info = super().step(action, skip_frames or self.decision_ticks)
        info["rl_situation"] = 0.0
        info["rl_situation_end"] = 0.0
        info["rl_situation_score"] = 0.0
        if self.situation is not None:
            info["rl_situation"] = 1.0
            info["is_scenario"] = 1.0  # out of the headline W/L, as the fork does for its scenarios
            info["scenario_defensive"] = 1.0
            done, score, details = self.situation.step(terminated or truncated)
            if self.situation.just_dropped:  # the threat appeared after this step's observation was taken
                obs = np.array(self.game.get_observation_for_team(0), dtype=np.float32)
            if done:
                # the score is the whole reward; ended as a truncation so the trainer's draw penalty (terminated with
                # |reward| < 0.5) can't fire, and the trainer zeroes the bootstrap on rl_situation_end
                reward, terminated, truncated = score, False, True
                info["rl_situation_end"] = 1.0
                info["rl_situation_score"] = score
                self.last_situation = {"name": self.situation.spec["name"], "score": score, **details}
                self.situation = None
            else:
                reward, truncated = 0.0, False
                terminated = False
        return obs, reward, terminated, truncated, info


def make_clone_env(seed=None, **env_config):
    """A picklable worker factory (AsyncVectorEnv spawns workers that import this module)."""
    config = dict(env_config)
    if seed is not None:
        config["scenario_seed"] = seed

    def _thunk():
        return CloneEnv(config)
    return _thunk
