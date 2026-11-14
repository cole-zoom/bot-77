"""How closely checkpoints follow Cole on his held-out shadow games (2026-11-14): the v8 test.

  uv run --no-sync python scripts/shadow_agreement.py --ckpt A.pth B.pth ... [--src data/cache/bc/shadow_t01_held]

Per checkpoint, over every held-out game run as one sequence (scripts/shadow_lance.py's guesses):
card_top1 (its top slot is his card, at his plays), p_his (the chance it gave his card, wait included),
auc (play vs wait: 1 - p_wait ranks his play moments above his waits), cell_tiles (median distance from its cell for
his card to his cell, in board tiles), cell_exact.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shadow_lance import guesses  # noqa: E402
from train_bc import BOARD_W  # noqa: E402
from python_ai.models.policy_io import load_net  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def score(ckpt: str, files: list[Path]) -> dict:
    net = load_net(ckpt, torch.device("cpu")).eval()
    top1, phis, dist, exact, at_play, at_wait = [], [], [], [], [], []
    for f in files:
        d = np.load(f)
        card, cell = d["card"], d["cell"]
        g = guesses(net, d["obs"], card)
        play = card < 4
        top1 += list(g["top"][play] == card[play])
        phis += list(g["p"][play, card[play]])
        at_play += list(1 - g["p"][play, 4]); at_wait += list(1 - g["p"][~play, 4])
        for m, c in zip(g["cell_his"][play], cell[play]):
            if c >= 0:
                dist.append(np.hypot(m % BOARD_W - c % BOARD_W, m // BOARD_W - c // BOARD_W)); exact.append(m == c)
    return {"plays": int(len(top1)), "card_top1": round(float(np.mean(top1)), 3), "p_his": round(float(np.mean(phis)), 3),
            "auc": round(auc(np.array(at_play), np.array(at_wait)), 3),
            "cell_tiles": round(float(np.median(dist)), 2), "cell_exact": round(float(np.mean(exact)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--src", default=str(ROOT / "data/cache/bc/shadow_t01_held"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    files = sorted(Path(a.src).glob("*.npz"))
    print(f"{len(files)} held-out games in {a.src}", flush=True)
    for c in a.ckpt:
        if not Path(c).exists():
            print(c, "missing", flush=True)
            continue
        print(c, score(c, files), flush=True)


if __name__ == "__main__":
    main()
