"""Option 3 step 2 (2026-10-27): PPO from the imitation agent, against frozen pro clones playing pros' decks.

  uv run --no-sync --with gymnasium==1.3.0 --with tensorboard python scripts/rl_vs_clones.py \
      [--clone data/models/bc_v2/frozen.pth] [--run data/models/rl_clones_v1] [--episodes 20000]

Run it again to resume (the fork's trainer resumes from its own checkpoint). The trainee starts from the imitation
agent's weights and plays Cole's current deck; team 1 is a frozen copy of the same imitation agent with a pro's deck
(bot77.rl.clone_env). The loop, reward, entropy schedule and logging are the fork's phase-2 trainer
(python_ai/trainers/train_selfplay.py) with a lower learning rate and a critic warm-up (below), minus its league: no
historical snapshots, scripted bots or exploiters join the pool, so the only opponent is the clone.

--leash-demos (2026-10-28, after the defence school): a leash to the teachers. After every PPO update the trainee takes
--leash-steps imitation steps (scripts/train_bc.py's loss) on short windows of the defence demos (Cole's and the
scripted defence's, scripts/build_defence_demos.py) and of --leash-pro pro games, with its own Adam. Rounds v3 / v4
drifted from their start to what the clones reward (the bridge-rush opener, committing in the overcommit situation:
-0.22 -> -1.05) and added nothing on the exam; the leash keeps what the school learned while RL works on the rest. The
fork's PPO loss has no hook for an extra term (python_ai/ stays untouched), hence separate steps. Play weight 1 and
lead 3 (a play also labels the 0.3 s before it): the school was trained at weight 50 and played at a log-50 wait
calibration, which is the same optimum as weight 1, so the leash pulls toward the policy the school played, not away.

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
    ap.add_argument("--decision-ticks", type=int, default=10,
                    help="engine ticks between our decisions: 10 = once a second (the fork's rate), 1 = every 0.1 s")
    ap.add_argument("--card-entropy", type=float, default=None,
                    help="target card-head entropy fraction (the fork's 0.35 is for 1 s decisions; at 0.1 s ~97%% of "
                         "decisions are waits and 0.35 would force random plays)")
    ap.add_argument("--engine-collision", action="store_true",
                    help="the engine's old pairwise pushing (v1-v5 trained on it) instead of the client's, as the page")
    ap.add_argument("--reference-file", default="data/live/cole_reference.jsonl",
                    help="Cole's starts (cole_reference_v3.jsonl: his takes plus his full-game tempo games, 2026-10-30)")
    ap.add_argument("--reference-share", type=float, default=0.0,
                    help="share of situation episodes that replay one of Cole's winning play-page starts")
    ap.add_argument("--init", default=None, help="start the trainee from these weights instead of the clone")
    ap.add_argument("--critic-warmup", type=int, default=500, help="episodes during which only the value head learns")
    ap.add_argument("--leash-demos", default=None, help="defence-school demos to replay after every update (see above)")
    ap.add_argument("--leash-pro", type=int, default=200, help="pro games (train split) in the leash's mix")
    ap.add_argument("--leash-share", type=float, default=0.7, help="share of the leash's windows from the demos")
    ap.add_argument("--leash-steps", type=int, default=2, help="imitation steps after every PPO update")
    ap.add_argument("--leash-batch", type=int, default=8)
    ap.add_argument("--leash-window", type=int, default=64, help="decisions a window (6.4 s; burn-in 16 mid-episode)")
    ap.add_argument("--leash-lr", type=float, default=1e-4)
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
        "RL_DECISION_TICKS": str(a.decision_ticks),
        "RL_REFERENCE_SHARE": str(a.reference_share),
        "RL_REFERENCE_FILE": str((ROOT / a.reference_file).resolve()),
        "RL_CLIENT_COLLISION": "0" if a.engine_collision else "1",  # the play page's physics (bot77.rl.clone_env)
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
            self.envs, self.infos, self.scores = envs, {}, []

        def step(self, action):
            out = self.envs.step(action)
            self.infos = out[4]
            # situation scores as they come (a defence window scores mid-game when the game plays on)
            end = self.infos.get("rl_situation_end")
            if end is not None:
                for i in np.flatnonzero(np.asarray(end) > 0):
                    self.scores.append(float(self.infos["rl_situation_score"][i]))
            return out

        def __getattr__(self, name):
            return getattr(self.envs, name)

    def situation_mask(trainer):
        v = trainer.envs.infos.get("rl_situation") if trainer.envs is not None else None
        return np.zeros(trainer.cfg.num_envs, np.float32) if v is None else np.asarray(v, np.float32)

    # At k decisions a second (k = 10 / decision_ticks) the one per-step shaping term, the elixir-overflow penalty,
    # would be charged k times as often: scale it back (deltas and potentials sum the same over a second).
    from python_ai.rewards import weights as rw
    per_second = 10 // a.decision_ticks
    if per_second > 1:
        _raw_shaping = bt.compute_shaping

        def scaled_shaping(stats, prev, *x, **k):
            out = _raw_shaping(stats, prev, *x, **k)
            if prev is None:
                return out
            over = np.maximum(0.0, stats["team0_elixir_current"] - rw.ELIXIR_OVERFLOW_THRESHOLD) / (
                10.0 - rw.ELIXIR_OVERFLOW_THRESHOLD)
            return out + rw.W_ELIXIR_OVERFLOW * over * (1.0 - 1.0 / per_second)
        bt.compute_shaping = scaled_shaping

    # On a situation's steps the reward is the situation's score alone: the game's shaping (tower potentials, troop
    # trades, the flawless-defence bonus) is zeroed there, or rushing the enemy tower would still pay. Patched in the
    # trainer's module namespace (the fork's python_ai/ is not edited).
    _shaping, _flawless = bt.compute_shaping, bt.flawless_defense_bonus
    current = {}
    bt.compute_shaping = lambda *x, **k: _shaping(*x, **k) * (1.0 - situation_mask(current["t"]))
    bt.flawless_defense_bonus = lambda *x, **k: _flawless(*x, **k) * (1.0 - situation_mask(current["t"]))

    leash_demos, leash_pro = [], []
    if a.leash_demos:
        sys.path.insert(0, str(ROOT / "scripts"))
        import train_bc as tb
        tb.WINDOW, tb.BURN_IN, tb.PLAY_WEIGHT, tb.LEAD = a.leash_window, 16, 1.0, 3
        tb.SHARDS = (ROOT / "data/cache/bc/shards_t01").resolve()
        tb.store.DB = str((ROOT / "data/lancedb").resolve())  # the cwd is the fork from here
        leash_demos = tb.load_demos((ROOT / a.leash_demos).resolve())
        pro = [e for e in tb.load_meta() if e["split"] == "train"]
        leash_pro = [pro[i] for i in np.random.default_rng(77).choice(len(pro), min(a.leash_pro, len(pro)), replace=False)]
        print(f"leash: {len(leash_demos)} demo episodes, {len(leash_pro)} pro games, {a.leash_steps} steps of "
              f"{a.leash_batch} x {a.leash_window} after every update", flush=True)

    class CloneTrainer(ts.Phase2Trainer):
        pipeline_name = "rl_clones"

        def __init__(self, *x, **k):
            super().__init__(*x, **k)
            current["t"] = self
            self.situation_scores: dict[str, list[float]] = {}

        def _truncation_bootstrap(self, dones, raw_rewards, next_obs, truncateds):
            # a situation's end that ended the episode (RL_SITUATION_FULL_GAME=0) is final: its score is the whole
            # return (no bootstrap from the game that would have gone on). With full games the game's own end is.
            boot = super()._truncation_bootstrap(dones, raw_rewards, next_obs, truncateds)
            end = self.envs.infos.get("rl_situation_final")
            if end is not None and np.any(end):
                e = torch.as_tensor(np.asarray(end, np.float32), device=self.device)
                boot["nonterminal"] = boot["nonterminal"] * (1 - e)
                boot["flag"] = boot["flag"] * (1 - e)
                boot["value"] = boot["value"] * (1 - e)
            return boot

        def on_episode_end(self, i, ctx):
            scores = self.situation_scores.setdefault("all", [])
            for sc in self.envs.scores:
                scores.append(sc)
                if len(scores) % 100 == 0:
                    last = scores[-100:]
                    print(f">>> situations: last 100 scores mean {np.mean(last):+.3f} (n={len(scores)})", flush=True)
                    self.writer.add_scalar("Situations/Score100", float(np.mean(last)), self.episodes_completed)
            self.envs.scores.clear()
            super().on_episode_end(i, ctx)

        def leash(self):
            """--leash-steps imitation steps on the teachers' windows (scripts/train_bc.py's loss)."""
            if not hasattr(self, "_leash_opt"):
                self._leash_opt = torch.optim.Adam(self.net.parameters(), lr=a.leash_lr)
                self._leash_rng = np.random.default_rng(77)
            n_demo = int(round(a.leash_batch * a.leash_share))
            lc = lp = 0.0
            for _ in range(a.leash_steps):
                batch = ([leash_demos[i] for i in self._leash_rng.choice(len(leash_demos), n_demo, replace=False)]
                         + [leash_pro[i] for i in self._leash_rng.choice(len(leash_pro), a.leash_batch - n_demo,
                                                                         replace=False)])
                l_card, l_cell, _ = tb.batch_loss(self.net, batch, self._leash_rng, self.device)
                self._leash_opt.zero_grad()
                (l_card + l_cell).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self._leash_opt.step()
                lc += float(l_card.detach())
                lp += float(l_cell.detach())
            self.writer.add_scalar("Leash/CardLoss", lc / a.leash_steps, self.episodes_completed)
            self.writer.add_scalar("Leash/CellLoss", lp / a.leash_steps, self.episodes_completed)
            self._leash_log = getattr(self, "_leash_log", 0) + 1
            if self._leash_log % 20 == 1:
                print(f"  >> leash @ ep {self.episodes_completed}: card loss {lc / a.leash_steps:.3f}, "
                      f"cell loss {lp / a.leash_steps:.3f}", flush=True)

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
                elif leash_demos:
                    self.leash()

        def build_envs(self):
            seeds = worker_seeds(self.cfg.seed, self.cfg.num_envs)
            return EnvsWithInfos(gym.vector.AsyncVectorEnv([make_clone_env(seed=s) for s in seeds]))

        def build_replay_env(self):
            return None

        def _refresh_pool(self, initial=False):  # no league: the clone is the only opponent
            self.historical_pool = []

        def should_stop(self):
            return self.episodes_completed >= a.episodes

    # the step-length-dependent settings, scaled to keep the same horizons in seconds at 0.1 s: discount (0.999 a
    # second -> 0.9999 a tick), GAE lambda (~10 s of lookahead), the BPTT window (5 s -> 10 s) and the rollout
    cfg = dict(lr=a.lr)
    if per_second > 1:
        cfg.update(gamma=0.999 ** (1.0 / per_second), gae_lambda=0.99, bptt_chunk=100, update_timestep=1000)
    if a.card_entropy is not None:
        import dataclasses
        ts.PHASE2_ENTROPY = dataclasses.replace(ts.PHASE2_ENTROPY, target_card=a.card_entropy)
    print(f"run {run}\nclone {clone}\ndeck {os.environ['CLASH_DECK']}\ndecisions every {a.decision_ticks / 10:.1f} s"
          f"\nppo {cfg}", flush=True)
    CloneTrainer(PPOConfig(**cfg)).run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
