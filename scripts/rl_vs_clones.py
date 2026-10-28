"""Option 3 step 2 (2026-10-27): PPO from the imitation agent, against frozen pro clones playing pros' decks.

  uv run --no-sync --with gymnasium==1.3.0 --with tensorboard python scripts/rl_vs_clones.py \
      [--clone data/models/bc_v2/frozen.pth] [--run data/models/rl_clones_v1] [--episodes 20000]

Run it again to resume (the fork's trainer resumes from its own checkpoint). The trainee starts from the imitation
agent's weights and plays Cole's current deck; team 1 is a frozen copy of the same imitation agent with a pro's deck
(bot77.rl.clone_env). The loop, reward, entropy schedule and logging are the fork's phase-2 trainer
(python_ai/trainers/train_selfplay.py) with a lower learning rate and a critic warm-up (below), minus its league: no
historical snapshots, scripted bots or exploiters join the pool, so the only opponent is the clone.

Under $RUN (gitignored): bootstrap.pth (the imitation weights the run starts from), bootstrap_selfplay.pth (the live
checkpoint), runs/rl_selfplay (TensorBoard), historical_checkpoints/ (snapshots every 2,000 episodes).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"
DECK = "2,81,5,43,9,95,7,3"  # Giant, Battle Ram, Mini P.E.K.K.A, Mega Minion, Bomber, Goblin Hut, Fireball, Arrows (2026-10-27:
# Bomber for Archers, Cole: "we were facing a lot of splash"; rl_clones_v1 trained with Archers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", default="data/models/bc_v2/frozen.pth")
    ap.add_argument("--run", default="data/models/rl_clones_v1")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--envs", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tuning, so a third of the fork's from-scratch 3e-4")
    ap.add_argument("--league-share", type=float, default=0.0,
                    help="share of episodes against the trainee's own past snapshots (bot77.rl.clone_env)")
    ap.add_argument("--snapshots", nargs="*", default=[],
                    help="more snapshot directories for the league (e.g. an earlier run's historical_checkpoints)")
    ap.add_argument("--situation-share", type=float, default=0.0,
                    help="share of episodes that start in a defence situation (bot77.rl.situations; replaces the "
                         "fork's scenarios)")
    ap.add_argument("--init", default=None, help="start the trainee from these weights instead of the clone")
    ap.add_argument("--critic-warmup", type=int, default=500, help="episodes during which only the value head learns")
    a = ap.parse_args()
    run = (ROOT / a.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    clone = (ROOT / a.clone).resolve()
    bootstrap = run / "bootstrap.pth"
    if not bootstrap.exists():  # the trainee starts as the clone (or --init); the opponent stays frozen at the clone
        shutil.copy((ROOT / a.init).resolve() if a.init else clone, bootstrap)
    # the fork reads these at import (python_ai/deck.py, train_selfplay.selfplay_paths, rl/checkpointing.py)
    os.environ.update({
        "CLASH_DECK": os.environ.get("CLASH_DECK", DECK),
        "CLASH_WEIGHTS": str(bootstrap),
        "CLASH_LOGDIR": str(run / "runs" / "rl"),
        "CLASH_RUN_ROOT": str(run),
        "CLASH_NUM_ENVS": str(a.envs),
        "RL_CLONE_CKPT": str(clone),
        "RL_PRO_DECKS": str((ROOT / "data/cache/rl/pro_decks.json").resolve()),
        "RL_LEAGUE_SHARE": str(a.league_share),
        "RL_SITUATION_SHARE": str(a.situation_share),
        "RL_SNAPSHOT_DIRS": os.pathsep.join([str(run / "historical_checkpoints")]
                                            + [str((ROOT / d).resolve()) for d in a.snapshots]),
    })
    os.environ.setdefault("OMP_NUM_THREADS", str(a.envs))
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import gymnasium as gym
    import python_ai  # noqa: F401  (before torch / numpy: the fork's import-order note)
    import torch
    from python_ai.rl.seeding import worker_seeds
    from python_ai.trainers import train_selfplay as ts

    from python_ai.rl.config import PPOConfig

    from bot77.rl.clone_env import make_clone_env

    import numpy as np
    from python_ai.rl import base_trainer as bt

    class EnvsWithInfos:
        """The vector env, keeping the last step's infos (situation steps switch the game's shaping off, below)."""

        def __init__(self, envs):
            self.envs, self.infos = envs, {}

        def step(self, action):
            out = self.envs.step(action)
            self.infos = out[4]
            return out

        def __getattr__(self, name):
            return getattr(self.envs, name)

    def situation_mask(trainer):
        v = trainer.envs.infos.get("rl_situation") if trainer.envs is not None else None
        return np.zeros(trainer.cfg.num_envs, np.float32) if v is None else np.asarray(v, np.float32)

    # On a situation's steps the reward is the situation's score alone: the game's shaping (tower potentials, troop
    # trades, the flawless-defence bonus) is zeroed there, or rushing the enemy tower would still pay. Patched in the
    # trainer's module namespace (the fork's python_ai/ is not edited).
    _shaping, _flawless = bt.compute_shaping, bt.flawless_defense_bonus
    current = {}
    bt.compute_shaping = lambda *x, **k: _shaping(*x, **k) * (1.0 - situation_mask(current["t"]))
    bt.flawless_defense_bonus = lambda *x, **k: _flawless(*x, **k) * (1.0 - situation_mask(current["t"]))

    class CloneTrainer(ts.Phase2Trainer):
        pipeline_name = "rl_clones"

        def __init__(self, *x, **k):
            super().__init__(*x, **k)
            current["t"] = self
            self.situation_scores: dict[str, list[float]] = {}

        def _truncation_bootstrap(self, dones, raw_rewards, next_obs, truncateds):
            # a situation's end is final: its score is the whole return (no bootstrap from the game that would
            # have gone on)
            boot = super()._truncation_bootstrap(dones, raw_rewards, next_obs, truncateds)
            end = self.envs.infos.get("rl_situation_end")
            if end is not None and np.any(end):
                e = torch.as_tensor(np.asarray(end, np.float32), device=self.device)
                boot["nonterminal"] = boot["nonterminal"] * (1 - e)
                boot["flag"] = boot["flag"] * (1 - e)
                boot["value"] = boot["value"] * (1 - e)
            return boot

        def on_episode_end(self, i, ctx):
            end = ctx.infos.get("rl_situation_end")
            if end is not None and end[i]:
                self.situation_scores.setdefault("all", []).append(float(ctx.infos["rl_situation_score"][i]))
                if len(self.situation_scores["all"]) % 100 == 0:
                    last = self.situation_scores["all"][-100:]
                    print(f">>> situations: last 100 scores mean {np.mean(last):+.3f} "
                          f"(n={len(self.situation_scores['all'])})", flush=True)
                    self.writer.add_scalar("Situations/Score100", float(np.mean(last)), self.episodes_completed)
            super().on_episode_end(i, ctx)

        def run_update(self):
            # Critic warm-up: imitation never trained the value head, so its early advantages are noise (the smoke
            # run's first update clipped 57% of samples). For the first --critic-warmup episodes only the value head
            # and the auxiliary heads learn; the policy stays the clone's. (Adam skips parameters without a grad.)
            warm = self.episodes_completed < a.critic_warmup
            if warm:
                for name, p in self.net.named_parameters():
                    p.requires_grad_(name.split(".")[0] in ("value_head", "aux_card_head", "cycle_id_head"))
            try:
                return super().run_update()
            finally:
                if warm:
                    for p in self.net.parameters():
                        p.requires_grad_(True)

        def build_envs(self):
            seeds = worker_seeds(self.cfg.seed, self.cfg.num_envs)
            return EnvsWithInfos(gym.vector.AsyncVectorEnv([make_clone_env(seed=s) for s in seeds]))

        def build_replay_env(self):
            return None

        def _refresh_pool(self, initial=False):  # no league: the clone is the only opponent
            self.historical_pool = []

        def should_stop(self):
            return self.episodes_completed >= a.episodes

    print(f"run {run}\nclone {clone}\ndeck {os.environ['CLASH_DECK']}", flush=True)
    CloneTrainer(PPOConfig(lr=a.lr)).run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
