"""Whole games, no situations (2026-10-30, for RL round v6): a checkpoint plays N full games on held-out seeds against the
pro clone (a pro deck each game) or against another checkpoint (our deck, as the league plays its snapshots, but at
its own 0.1 s pace: the trainer's league asks it once a second).

  uv run --no-sync --with gymnasium==1.3.0 python scripts/match_eval.py --ckpt A.pth [--vs clone | --vs B.pth] [--n 60]

Seeds 20_000+ (training and the situation exam use others). Our side decides every 0.1 s and samples as in training;
the opponent decides once a second (CloneEnv). Physics as the play page and training: the client's pushing, and the
engine's defaults (tower reach footprint, pathing, the pocket). The spell guard is on, as everywhere.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"
DECK = "2,81,5,43,9,95,7,3"
_S: dict = {}


def setup(ckpt):
    os.environ.update({"CLASH_DECK": DECK, "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "0",
                       "RL_CLIENT_COLLISION": "1"})
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import python_ai  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from python_ai.models.net import MicroRoyaleNet
    from bot77.rl.clone_env import CloneEnv
    net = MicroRoyaleNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    net.eval()
    _S.update(torch=torch, net=net, CloneEnv=CloneEnv)


def game(args):
    ckpt, vs, seed = args
    if not _S:
        setup(ckpt)
    torch, net = _S["torch"], _S["net"]
    from torch.distributions import Categorical
    from python_ai.models.policy_io import LSTM_HIDDEN
    env = _S["CloneEnv"]({"scenario_seed": 0, "decision_ticks": 1, "scenarios_enabled": False})  # whole games only (the fork injects its scenarios otherwise)
    if vs != "clone":
        env.league_share = 1.0
        env._snapshots = lambda: [vs]  # the league draws its one member
        # a checkpoint trained at 0.1 s decides at 0.1 s too: CloneEnv asks the opponent once a second (the clone's
        # pace), which handicaps a 0.1 s checkpoint; measured, the 8-2 model "beat" itself 3-0-1 that way
        env.opp_every = 1
    env.rng = np.random.default_rng(seed)
    env.scenario_rng = env.rng
    torch.manual_seed(seed)
    obs, _ = env.reset()
    h = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
    while True:
        o = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            m = net.affordability_mask(o)
            f, e, sp, hi = net.extract_features_hires(o)
            lg, _, _, _, h = net.step_lstm_and_card(f, h, m)
            k = Categorical(logits=lg).sample()
            x, y = net.cell_to_xy(Categorical(logits=net.placement_given_card(h[0], e, k, o, sp, hires_map=hi)).sample())
        obs, r, term, trunc, info = env.step({"card_index": int(k.item()), "target_x": np.array([float(x.item())], np.float32),
                                              "target_y": np.array([float(y.item())], np.float32),
                                              "activate_ability_slot1": 0, "activate_ability_slot2": 0})
        if term or trunc:
            g = env.game
            towers = [[max(0, g.get_tower_hp(t, s)) > 0 for s in range(3)] for t in (0, 1)]
            return {"seed": seed, "result": float(r), "clock": round(g.get_current_tick() * 0.1, 1),
                    "crowns_for": 3 - sum(towers[1]), "crowns_against": 3 - sum(towers[0]),
                    "opp_deck": list(env.opp_deck)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vs", default="clone", help="'clone' (a pro deck each game) or another checkpoint")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ckpt = str((ROOT / a.ckpt).resolve())
    vs = a.vs if a.vs == "clone" else str((ROOT / a.vs).resolve())
    with Pool(a.workers) as pool:
        rows = pool.map(game, [(ckpt, vs, 20_000 + i) for i in range(a.n)])
    res = np.array([r["result"] for r in rows])
    w, l = (res > 0.5).mean(), (res < -0.5).mean()
    se = np.sqrt(w * (1 - w) / len(rows))
    print(f"{Path(a.ckpt).name} vs {Path(a.vs).name if a.vs != 'clone' else 'the pro clone'}: {len(rows)} games, "
          f"win {w:.0%} (+-{1.96 * se:.0%}), loss {l:.0%}, draw {1 - w - l:.0%}; crowns for {np.mean([r['crowns_for'] for r in rows]):.2f}, "
          f"against {np.mean([r['crowns_against'] for r in rows]):.2f}; mean length {np.mean([r['clock'] for r in rows]):.0f} s")
    if a.json:
        Path(a.json).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
