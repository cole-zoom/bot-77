"""Self-play round against a league, anchored to the imitation model (2026-11-08, Cole: "make the bot play against
itself soon").

  uv run --no-sync --with gymnasium==1.3.0 --with tensorboard python scripts/selfplay_league.py \
      --init data/models/bc_school_v7/epoch03.pth --run data/models/league_v1 [--episodes 3000] [--envs 8]

Run it again to resume (the fork's trainer resumes from its own checkpoint). What's different from rl_vs_clones.py,
whose four rounds (v3-v6, 2026-10-28/30) each made their start worse (decisions.md, 2026-10-30): what they had in
common was lr 1e-4 on an imitation policy with no KL to it, the fork's game shaping, and opponents deciding once a
second. Here:
  - **KL anchor to the start.** After every PPO update, imitation-style steps pull the policy back toward a frozen copy
    of --init on the rollout's own states: KL(start || now) of the card head on every decision, and of the placement
    head (given the card played) on every play, on windows unrolled from a zeroed LSTM (--anchor-burn-in steps of
    burn-in, as train_bc.py's windows), the same unroll for both nets. Steps repeat until the card KL is under
    --kl-target or --anchor-max-steps is reached. The fork's PPO loss has no hook for an extra term (python_ai/ stays
    untouched), hence separate steps, as the old leash.
  - **The result as the reward**: +1 / -1 / 0 plus --crown-coef per crown of margin; the trainer's shaping, flawless
    bonus and draw penalty are off (bot77.rl.league_env).
  - **A league** (bot77.rl.league_env): frozen school models at 0.1 s, the five deck bots, the pro clone, the run's
    own snapshots. Lower lr (3e-5).
Then scripts/league_gate.py decides whether the result replaces the start.

Under $RUN (gitignored): bootstrap.pth, league.json, runs/ (TensorBoard), historical_checkpoints/, the fork's own
checkpoint; league_log.jsonl gets one line per update (per-member win rates, KL before / after, steps).
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"
DECK = "2,81,5,43,9,95,7,3"  # our live deck (rl_vs_clones.DECK)
SCHOOL = ["data/models/bc_school_v5/epoch03.pth", "data/models/bc_school_v6/epoch03.pth",
          "data/models/bc_school_v7/epoch03.pth"]
DECK_BOTS = ["pekka_bs", "golem", "gdp", "log_bait", "hog26"]


def league(a) -> list[dict]:
    """The pool: --w-school split over the school models, --w-decks over the deck bots, --w-clone the clone."""
    pool = [{"kind": "ckpt", "path": str((ROOT / p).resolve()), "w": a.w_school / len(SCHOOL)} for p in SCHOOL]
    pool += [{"kind": "deck", "key": k, "w": a.w_decks / len(DECK_BOTS)} for k in DECK_BOTS]
    if a.w_clone > 0:
        pool.append({"kind": "clone", "w": a.w_clone})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="data/models/bc_school_v7/epoch03.pth", help="the start")
    ap.add_argument("--anchor", default=None, help="the KL anchor (default: --init)")
    ap.add_argument("--hours", type=float, default=0.0, help="stop after this much wall-clock (0: no limit)")
    ap.add_argument("--run", default="data/models/league_v1")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--w-school", type=float, default=0.35)
    ap.add_argument("--w-decks", type=float, default=0.35)
    ap.add_argument("--w-clone", type=float, default=0.1)
    ap.add_argument("--w-snapshots", type=float, default=0.2, help="the run's own snapshots, once there are any")
    ap.add_argument("--snapshot-every", type=int, default=500, help="episodes between the run's own snapshots")
    ap.add_argument("--crown-coef", type=float, default=0.1)
    ap.add_argument("--critic-warmup", type=int, default=300, help="episodes during which only the value head learns")
    ap.add_argument("--kl-target", type=float, default=0.02, help="card-head KL (nats a decision) to get back under")
    ap.add_argument("--anchor-min-steps", type=int, default=1)
    ap.add_argument("--anchor-max-steps", type=int, default=8)
    ap.add_argument("--anchor-batch", type=int, default=8, help="windows a step")
    ap.add_argument("--anchor-window", type=int, default=64)
    ap.add_argument("--anchor-burn-in", type=int, default=32)
    ap.add_argument("--anchor-lr", type=float, default=1e-4)
    ap.add_argument("--place-kl-weight", type=float, default=0.5)
    a = ap.parse_args()
    T0 = time.time()
    run = (ROOT / a.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    init = (ROOT / a.init).resolve()
    bootstrap = run / "bootstrap.pth"
    if not bootstrap.exists():
        shutil.copy(init, bootstrap)
    pool = league(a)
    (run / "league.json").write_text(json.dumps(pool, indent=1))
    os.environ.update({
        "CLASH_DECK": os.environ.get("CLASH_DECK", DECK),
        "CLASH_WEIGHTS": str(bootstrap),
        "CLASH_LOGDIR": str(run / "runs" / "league"),
        "CLASH_RUN_ROOT": str(run),
        "CLASH_NUM_ENVS": str(a.envs),
        "RL_CLONE_CKPT": str((ROOT / "data/models/bc_v2/frozen.pth").resolve()),
        "RL_PRO_DECKS": str((ROOT / "data/cache/rl/pro_decks.json").resolve()),
        "RL_SITUATION_SHARE": "0",
        "RL_DECISION_TICKS": "1",
        "RL_CLIENT_COLLISION": "1",  # the play page's physics
        "RL_LEAGUE": json.dumps(pool),
        "RL_SNAPSHOT_WEIGHT": str(a.w_snapshots),
        "RL_SNAPSHOT_DIRS": str(run / "historical_checkpoints"),
        "RL_CROWN_COEF": str(a.crown_coef),
    })
    os.environ.setdefault("OMP_NUM_THREADS", str(a.envs))
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import gymnasium as gym
    import python_ai  # noqa: F401  (before torch / numpy: the fork's import-order note)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.rl import base_trainer as bt
    from python_ai.rl.config import PPOConfig
    from python_ai.rl.seeding import worker_seeds
    from python_ai.trainers import train_selfplay as ts

    from bot77.rl.league_env import make_league_env

    # the result is the whole reward (bot77.rl.league_env): no shaping, no flawless bonus, no draw penalty
    bt.compute_shaping = lambda stats, *x, **k: np.zeros(len(stats["team0_elixir_current"]), np.float32)
    bt.flawless_defense_bonus = lambda dones, *x, **k: np.zeros(len(dones), np.float32)
    bt.DRAW_PENALTY = 0.0

    ref = MicroRoyaleNet()
    anchor = (ROOT / a.anchor).resolve() if a.anchor else init
    ref.load_state_dict(torch.load(anchor, map_location="cpu", weights_only=False)["model"], strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    names = [m.get("path", m.get("key", m["kind"])).split("/")[-2] if m["kind"] == "ckpt" else m.get("key", m["kind"])
             for m in pool] + ["own snapshots"]
    log_path = run / "league_log.jsonl"

    class EnvsWithInfos:
        """The vector env, tallying each finished game's member and result."""

        def __init__(self, envs):
            self.envs, self.tally = envs, defaultdict(lambda: [0, 0, 0])  # member -> [wins, games, crown margin]

        def step(self, action):
            out = self.envs.step(action)
            opp = out[4].get("league_opp")
            if opp is not None:
                for i in np.flatnonzero(np.asarray(opp) >= 0):
                    t = self.tally[names[int(opp[i])]]
                    t[0] += int(out[1][i] > 0.5)
                    t[1] += 1
                    t[2] += int(out[4]["league_margin"][i])
            return out

        def __getattr__(self, name):
            return getattr(self.envs, name)

    def unroll(net, obs, cards, keep):
        """Card and placement logits over (L, B) from a zeroed LSTM; placement given the stored card."""
        L, B = cards.shape
        flat = obs.reshape(L * B, -1)
        feats, embeds, spatial, hires = net.extract_features_hires(flat)
        mask = net.affordability_mask(flat).view(L, B, net.hand_size + 1)
        hx = torch.zeros(B, 256, device=obs.device)
        cx = torch.zeros(B, 256, device=obs.device)
        out = net.forward_sequence(feats.view(L, B, -1), embeds.view(L, B, net.hand_size + 1, -1),
                                   spatial.view(L, B, *spatial.shape[1:]), flat.view(L, B, -1), mask, cards, keep,
                                   (hx, cx), hires_seq=hires.view(L, B, *hires.shape[1:]))
        return out[0], out[1]

    def kl(ref_logits, cur_logits):
        """KL(ref || cur) per row, over the arms the reference allows."""
        lp_ref = F.log_softmax(ref_logits, -1)
        lp_cur = F.log_softmax(cur_logits, -1)
        p = lp_ref.exp()
        d = torch.where(p > 0, p * (lp_ref - lp_cur), torch.zeros_like(p))
        return torch.nan_to_num(d, nan=0.0, posinf=0.0).sum(-1)

    class LeagueTrainer(ts.Phase2Trainer):
        pipeline_name = "league"

        def build_envs(self):
            seeds = worker_seeds(self.cfg.seed, self.cfg.num_envs)
            return EnvsWithInfos(gym.vector.AsyncVectorEnv([make_league_env(seed=s) for s in seeds]))

        def build_replay_env(self):
            return None

        def _refresh_pool(self, initial=False):  # the league is LeagueEnv's, not the fork's
            self.historical_pool = []

        def should_stop(self):
            n = self.episodes_completed
            late = a.hours > 0 and time.time() - T0 > a.hours * 3600
            if (n >= a.episodes or late) and getattr(self, "_snap_at", -1) != n:
                self._snap_at = n  # the last weights, whatever the snapshot cadence
                d = run / "historical_checkpoints"
                d.mkdir(exist_ok=True)
                torch.save({"model": self.net.state_dict()}, d / f"ep{n:05d}.pth")
            return n >= a.episodes or late

        def on_episode_end(self, i, ctx):
            super().on_episode_end(i, ctx)
            n = self.episodes_completed
            if n and n % a.snapshot_every == 0 and getattr(self, "_snap_at", -1) != n:
                self._snap_at = n
                d = run / "historical_checkpoints"
                d.mkdir(exist_ok=True)
                torch.save({"model": self.net.state_dict()}, d / f"ep{n:05d}.pth")

        def anchor(self, batch):
            """KL steps toward the start on this rollout's states; returns (kl before, kl after, steps)."""
            if not hasattr(self, "_anchor_opt"):
                self._anchor_opt = torch.optim.Adam(self.net.parameters(), lr=a.anchor_lr)
                self._anchor_rng = np.random.default_rng(77)
                ref.to(self.device)  # the trainer's device: the GPU when there is one
            dev = self.device
            obs, cards, masks = batch["obs"], batch["card_actions"], batch["masks"]
            dec = batch["decision"] * batch["valid"]
            T, N = cards.shape[:2]
            W, burn = a.anchor_window + a.anchor_burn_in, a.anchor_burn_in
            first = last = None
            steps = 0
            while steps < a.anchor_max_steps:
                t0 = self._anchor_rng.integers(0, T - W + 1, a.anchor_batch)
                ev = self._anchor_rng.integers(0, N, a.anchor_batch)
                tt = (torch.as_tensor(t0).unsqueeze(0) + torch.arange(W).unsqueeze(1)).to(obs.device)
                ee = torch.as_tensor(ev).unsqueeze(0).expand(W, -1).to(obs.device)
                o, c = obs[tt, ee].to(dev), cards[tt, ee].long().to(dev)
                keep = masks[tt, ee].to(dev)  # 1 - done: forward_sequence zeroes the LSTM after a game's last step, as in PPO
                rows = dec[tt, ee].to(dev) > 0.5
                rows[:burn] = False
                with torch.no_grad():
                    rc, rp = unroll(ref, o, c, keep)
                cc, cp = unroll(self.net, o, c, keep)
                k_card = kl(rc, cc)[rows]
                played = rows & (c < self.net.hand_size)
                k_place = kl(rp, cp)[played] if played.any() else torch.zeros(1, device=dev)
                loss = k_card.mean() + a.place_kl_weight * k_place.mean()
                kc = float(k_card.mean().detach()) if len(k_card) else 0.0
                first = kc if first is None else first
                last = kc
                if steps >= a.anchor_min_steps and kc < a.kl_target:
                    break
                self._anchor_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self._anchor_opt.step()
                steps += 1
            return first, last, steps

        def run_update(self):
            # Critic warm-up: imitation never trained the value head, so its early advantages are noise (rl_vs_clones)
            warm = self.episodes_completed < a.critic_warmup
            drain = self.buffer.drain
            kept = {}

            def drain_and_keep():
                kept["b"] = drain()
                return kept["b"]
            self.buffer.drain = drain_and_keep
            if warm:
                for name, p in self.net.named_parameters():
                    p.requires_grad_(name.split(".")[0] in ("value_head", "aux_card_head", "cycle_id_head"))
            try:
                stats = super().run_update()
            finally:
                self.buffer.drain = drain
                if warm:
                    for p in self.net.parameters():
                        p.requires_grad_(True)
            row = {"ep": self.episodes_completed, "warm": warm}
            if not warm and "b" in kept:
                row["kl_before"], row["kl_after"], row["anchor_steps"] = self.anchor(kept["b"])
            kept.clear()
            tally = self.envs.tally if self.envs is not None else {}
            row["members"] = {k: {"won": v[0], "games": v[1], "margin": v[2]} for k, v in tally.items()}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if tally:
                print("  >> league @ ep {}: {}{}".format(
                    self.episodes_completed,
                    ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(tally.items())),
                    f" | KL {row['kl_before']:.3f} -> {row['kl_after']:.3f} in {row['anchor_steps']}"
                    if "kl_before" in row else ""), flush=True)
                tally.clear()
            return stats

    # the step-length-dependent settings at 0.1 s, as rl_vs_clones: discount 0.999 a second, ~10 s of GAE lookahead
    cfg = dict(lr=a.lr, gamma=0.999 ** 0.1, gae_lambda=0.99, bptt_chunk=100, update_timestep=1000)
    print(f"run {run}\ninit {init}\nanchor {anchor}\ndevice {'cuda' if torch.cuda.is_available() else 'cpu'}\nleague {[(n, round(m['w'], 3)) for n, m in zip(names, pool)]} + own snapshots "
          f"{a.w_snapshots}\nppo {cfg}", flush=True)
    LeagueTrainer(PPOConfig(**cfg)).run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
