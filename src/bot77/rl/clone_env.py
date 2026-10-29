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

import copy
import glob
import json
import os

import numpy as np

import clash_royale_env
from python_ai.envs.selfplay_env import PFSP_EXPONENT, PFSP_MIN_WEIGHT, MicroRoyaleSelfPlayEnv

from bot77.rl import guards
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
        # every take of his on the same start is an answer key there (situation, seed) -> [[(t, card, x, y), ...], ...]
        self.keys: dict[tuple, list] = {}
        for r in self.references:
            take = [(p["t"], p["card_id"], p["x"], p["y"]) for p in r.get("plays", []) if p.get("card_id") is not None]
            if take:
                self.keys.setdefault((r["situation"], int(r["seed"])), []).append(take)
        self._start_tpl = None
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
        # the spell guard (bot77.rl.guards, plan P1, 2026-10-29): a Fireball / Arrows that would touch nothing of theirs
        # becomes a wait. On by default so training, the exam and live play agree; RL_SPELL_GUARD=0 (or env_config
        # spell_guard False) for the guard-off record. RL v1-v5 ran without it.
        self.spell_guard = bool(env_config.get("spell_guard", os.environ.get("RL_SPELL_GUARD", "1") != "0"))
        self.spell_vetoes = 0  # this episode's
        # full games (RL, plan P2): a per-step penalty on elixir lost at the cap, RL_LEAK_COEF per elixir. Default 0 so
        # v1-v5 stay reproducible; ~0.009 makes a whole regulation game of sitting at 10 (~84 elixir) cost about one
        # Princess Tower in the fork's reward (W_TOWER_DESTROYED 0.6 + W_BLDG 0.5 x 2534 / MAX_BUILDING_HP ~ 0.74)
        self.rl_leak_coef = float(os.environ.get("RL_LEAK_COEF", "0"))

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
        if (self._start_tpl or self.forced_template) in sit.SCRIPTED_OPP:
            self.opp_deck = list(sit.SCRIPT_DECK)  # Cole's 1 / 8 / 9 and the tempo situations: the scripted opponent's cards
        if path != self.loaded:
            self.set_historical_opponent(path)
            self.loaded = path
        self.opponent_checkpoint_path = path  # the key the fork's win-rate estimate (pfsp_stats) is kept under
        none = clash_royale_env.TowerTroopType.NONE
        self.game = clash_royale_env.ClashRoyaleEnv(self.deck, self.opp_deck, self.max_ticks, none, none)
        # the engine's own generator (hand cycle, deals) seeded from ours without drawing from it, so a start is exact
        # again and the situation draws are unchanged (it was OS entropy: Cole's replays diverged, 2026-10-28)
        self.game.seed(int(copy.deepcopy(self.rng).integers(2 ** 31)))

    def _opponent_action(self):
        """The opponent decides once a second (every opp_every of our steps); in between it waits."""
        self.opp_calls += 1
        if (self.opp_calls - 1) % self.opp_every:
            return 4, 0.0, 0.0, False, False
        if self.situation is not None and self.situation.spec.get("opp"):
            move = sit.opponent_script(self.situation, self)  # the review situations' scripted opponent
            return (*(move or (4, 0.0, 0.0)), False, False)
        return super()._opponent_action()

    def reset(self, seed=None, options=None):
        fixed = self.forced_seed
        if fixed is None and self.references and self.situation_share > 0 and self.rng.random() < self.situation_share \
                and self.rng.random() < self.reference_share:
            r = self.references[int(self.rng.integers(len(self.references)))]
            fixed = (sit.BY_NAME[r["situation"]], int(r["seed"]))
            key = self.keys.get((r["situation"], int(r["seed"])), [])
        else:
            key = []
        main_rng = None
        drawn = None
        if fixed is None and self.forced_template is None and self.situation_share > 0 \
                and self.rng.random() < self.situation_share:
            # drawn before the engine is built: Cole's 1 / 8 / 9 need the scripted opponent's deck (2026-10-28)
            drawn = sit.TEMPLATES[int(self.rng.integers(len(sit.TEMPLATES)))]
        self._start_tpl = fixed[0] if fixed is not None else drawn
        if fixed is not None:
            # the play page's start: the generator seeded before reset (opponent deck, situation and hand all follow)
            main_rng, main_srng = self.rng, getattr(self, "scenario_rng", None)
            self.rng = np.random.default_rng(fixed[1])
            self.scenario_rng = self.rng
        obs, info = super().reset(seed=seed, options=options)
        self.opp_calls = 0
        self.spell_vetoes = 0
        self.situation = None
        if fixed is not None or self.forced_template is not None or drawn is not None:
            tpl = fixed[0] if fixed is not None else (self.forced_template or drawn)
            self.situation = sit.Situation(self.game, tpl(self.rng), list(self.deck), self.rng,
                                           dt=self.decision_ticks / 10.0)
            self.situation.key = key
            obs = np.array(self.game.get_observation_for_team(0), dtype=np.float32)
        if main_rng is not None and self.forced_seed is None:
            self.rng, self.scenario_rng = main_rng, main_srng  # back to the env's own stream
        return obs, info

    def _guard(self, action):
        """The spell guard on our action: (the action to play, vetoed?, the whiffed spell's id if it goes through)."""
        k = int(np.asarray(action["card_index"]).reshape(-1)[0])
        if k >= 4:
            return action, False, None
        hand = list(self.game.get_hand())
        if k >= len(hand) or hand[k] not in guards.GUARDED or self.game.get_elixir() < sit.COST.get(hand[k], 99):
            return action, False, None
        x = float(np.asarray(action["target_x"]).reshape(-1)[0])
        y = float(np.asarray(action["target_y"]).reshape(-1)[0])
        if not guards.spell_vetoed(hand[k], x, y, self.game.get_units()):
            return action, False, None
        if not self.spell_guard:
            return action, False, hand[k]
        return {**action, "card_index": 4}, True, None

    def step(self, action, skip_frames=None):
        action, vetoed, whiff = self._guard(action)
        self.spell_vetoes += vetoed
        el0, spent0 = float(self.game.get_elixir()), float(self.game.get_elixir_spent(0))
        before = (list(self.game.get_hand()), float(self.game.get_elixir_spent(0))) if self.situation is not None else None
        obs, reward, terminated, truncated, info = super().step(action, skip_frames or self.decision_ticks)
        k = int(action["card_index"])
        if before is not None and k < 4 and float(self.game.get_elixir_spent(0)) > before[1] + 1e-6:
            # our play, timed as the play page logs Cole's (the time after this step)
            self.situation.my_plays.append((round(self.situation.seconds + self.situation.dt, 1), before[0][k],
                                            float(action["target_x"][0]), float(action["target_y"][0])))
        if whiff is not None and self.situation is not None and float(self.game.get_elixir_spent(0)) > spent0 + 1e-6:
            self.situation.wasted_spells += 1
            self.situation.wasted_spell_elixir += sit.COST.get(whiff, 0)
        if self.situation is None and self.rl_leak_coef > 0:
            reward -= self.rl_leak_coef * sit.leak_step(self.game, 0, el0, spent0, int(skip_frames or self.decision_ticks))
        info["spell_veto"] = float(vetoed)
        info["spell_vetoes"] = float(self.spell_vetoes)
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
                if details.get("key_match") is not None:
                    reward += sit.KEY_BONUS * details["key_match"]  # Cole's take as a positive signal
                info["rl_situation_end"] = 1.0
                info["rl_situation_score"] = score
                self.last_situation = {"name": self.situation.spec["name"], "score": score, **details,
                                       "spell_vetoes": self.spell_vetoes}
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
