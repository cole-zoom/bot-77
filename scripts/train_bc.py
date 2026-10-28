"""Imitation pretraining of the self-play network on pro replays (option 3 step 1, 2026-10-24).

  uv run --no-sync python scripts/train_bc.py [--epochs 6] [--out data/models/bc_v1]

Data: data/cache/bc/shards (scripts/build_bc_dataset.py): one trajectory per player per human replay, a decision a
second. The network is the fork's MicroRoyaleNet (python_ai, MIT), the same one the reinforcement learning trains, so
the result can be its starting point. Training follows the fork's behaviour cloning (python_ai/trainers/bc_pretrain.py)
with: rows whose card or cell is unknown left out of that loss, the winner's rows weighted WIN_WEIGHT (Cole: weight
winners up; results from matches.result), play rows weighted PLAY_WEIGHT in the card loss, and batched truncated BPTT through `forward_sequence` (random windows of
WINDOW steps, the first BURN_IN only warming the LSTM). Held-back games are only scored.

--shards data/cache/bc/shards_t01 (2026-10-28): the 0.1 s data (scripts/build_bc_dataset.py --step 0.1). Its shards
are compressed (the observation is ~99% zeros), so each is unpacked once into a sparse matrix held in memory (~100
values a row) and windows are densified on the fly; use longer windows (--window 256 = 25.6 s) and a heavier play
weight (--play-weight; plays are ~1% of the rows, against ~10% at 1 s).

--demos data/cache/bc/defence_t01 (2026-10-28, the defence school; scripts/build_defence_demos.py): the scripted
defence's and Cole's situation episodes, --demo-share of every batch. A scripted episode is kept only if it scored at
least its template's median and above -2 (the better half; none of the emergency, which the script loses);
Cole's are all kept, weighted as built (3). Every window that starts at an episode's first step (all the demos, the
start of a pro game) is trained from that step: the LSTM is fresh there, so no burn-in is needed, and the demos'
first decision is often the answer itself.

Scores (held-back): card at plays (which of the 4 hand cards, given a play happened; A v3 got ~51%), play-vs-wait AUC,
placement (the cell the pro used, and the distance in tiles to it).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from bot77.sim import store

FORK = Path(__file__).resolve().parents[1] / "third_party" / "clash-royale-env"
sys.path.insert(0, str(FORK))
from python_ai.models.net import MicroRoyaleNet  # noqa: E402

SHARDS = Path("data/cache/bc/shards")
WIN_WEIGHT = 2.0
PLAY_WEIGHT = 5.0  # ~90% of decisions are waits: without this the card head barely learns WHICH card (v1: 32%)
WINDOW, BURN_IN, BATCH = 64, 16, 16
BOARD_W = 18


_CACHE: dict = {}


def obs_rows(f) -> "sp.csr_matrix":
    """A shard's observations as a sparse matrix, unpacked once and kept (dense float16 would be ~50 GB at 0.1 s)."""
    if f not in _CACHE:
        d = np.load(f)
        _CACHE[f] = (sp.csr_matrix(d["obs"].astype(np.float32)), d["card"], d["cell"])
    return _CACHE[f]


def load_meta():
    results = {m["match_id"]: m["result"] for m in store.read("matches", "source IN ('replay_tv', 'replay_bl')")}
    pairs = {m["match_id"]: m["pair_match_id"] for m in store.read("matches", "pov_side = 'bottom'")}
    eps = []
    for f in sorted(SHARDS.glob("*.npz")):
        d = np.load(f)
        mid, side = f.stem.rsplit("_", 1)
        res = results.get(mid) if side == "bottom" else results.get(pairs.get(mid))
        eps.append({"file": str(f), "n": len(d["card"]), "split": str(d["split"]),
                    "w": WIN_WEIGHT if res == "win" else 1.0, "result": res})
    return eps


def window(ep, rng, length=None):
    length = length or WINDOW
    obs, card, cell = obs_rows(ep["file"])
    n = len(card)
    s = 0 if n <= length else int(rng.integers(0, n - length + 1))
    sl = slice(s, s + length)
    return obs[sl].toarray(), card[sl], cell[sl], s


def batch_loss(net, eps, rng, device):
    """Card + placement loss on a batch of windows (padded to WINDOW; loss from BURN_IN on)."""
    obs = np.zeros((WINDOW, len(eps), net.obs_dim if hasattr(net, "obs_dim") else 13977), np.float32)
    card = np.full((WINDOW, len(eps)), -1, np.int64)
    cell = np.full((WINDOW, len(eps)), -1, np.int64)
    valid = np.zeros((WINDOW, len(eps)), bool)
    w = np.zeros((WINDOW, len(eps)), np.float32)
    burn = np.zeros(len(eps), np.int64)
    for b, ep in enumerate(eps):
        o, c, ce, start = window(ep, rng)
        L = len(c)
        obs[:L, b], card[:L, b], cell[:L, b], valid[:L, b], w[:L, b] = o, c, ce, True, ep["w"]
        burn[b] = 0 if start == 0 else BURN_IN  # a window from the episode's first step has a true fresh LSTM
    return sequence_loss(net, torch.tensor(obs, device=device), torch.tensor(card, device=device),
                         torch.tensor(cell, device=device), torch.tensor(valid, device=device),
                         torch.tensor(w, device=device), burn_in=torch.tensor(burn, device=device))


def sequence_loss(net, obs, card, cell, valid, w, burn_in=0):
    L, B = card.shape
    flat = obs.reshape(L * B, -1)
    feats, embeds, spatial, hires = net.extract_features_hires(flat)
    mask = net.affordability_mask(flat).view(L, B, net.hand_size + 1)
    card_in = card.clamp_min(0)  # the placement head is conditioned on the played card (wait / unknown: masked out)
    hx = torch.zeros(B, 256, device=obs.device)
    cx = torch.zeros(B, 256, device=obs.device)
    keep = torch.ones(L, B, device=obs.device)
    out = net.forward_sequence(feats.view(L, B, -1), embeds.view(L, B, net.hand_size + 1, -1),
                               spatial.view(L, B, *spatial.shape[1:]), flat.view(L, B, -1), mask, card_in, keep,
                               (hx, cx), hires_seq=hires.view(L, B, *hires.shape[1:]))
    cl, pl = out[0], out[1]
    if isinstance(burn_in, torch.Tensor):  # one burn-in per column
        t_ok = torch.arange(L, device=obs.device).unsqueeze(1) >= burn_in.unsqueeze(0)
    else:
        t_ok = torch.zeros(L, B, dtype=torch.bool, device=obs.device)
        t_ok[burn_in:] = True
    rows = valid & t_ok & (card >= 0)
    rows = rows & torch.isfinite(cl.gather(2, card_in.unsqueeze(2)).squeeze(2))  # a mask-illegal target: dropped
    ce_card = F.cross_entropy(cl[rows], card[rows], reduction="none")
    wc = w[rows] * torch.where(card[rows] < net.hand_size, PLAY_WEIGHT, 1.0)
    l_card = (ce_card * wc).sum() / wc.sum().clamp_min(1e-8)
    prow = rows & (card < net.hand_size) & (cell >= 0)
    cell_in = cell.clamp_min(0)
    prow = prow & torch.isfinite(pl.gather(2, cell_in.unsqueeze(2)).squeeze(2))
    if prow.any():
        ce_cell = F.cross_entropy(pl[prow], cell[prow], reduction="none")
        l_cell = (ce_cell * w[prow]).sum() / w[prow].sum().clamp_min(1e-8)
    else:
        l_cell = torch.zeros((), device=obs.device)
    return l_card, l_cell, (cl, pl, rows, prow)


@torch.no_grad()
def evaluate(net, eps, device, max_eps=0):
    """Whole held-back trajectories, from the start of each game (no burn-in)."""
    net.eval()
    play_hit = play_n = cell_hit = cell_n = 0
    dist, p_wait, is_wait = [], [], []
    for ep in eps[: max_eps or None]:
        o, c, ce = obs_rows(ep["file"])
        obs = torch.tensor(o.toarray(), device=device).unsqueeze(1)
        card = torch.tensor(c, device=device).unsqueeze(1)
        cell = torch.tensor(ce, device=device).unsqueeze(1)
        valid = torch.ones_like(card, dtype=torch.bool)
        w = torch.ones_like(card, dtype=torch.float32)
        _, _, (cl, pl, rows, prow) = sequence_loss(net, obs, card, cell, valid, w)
        prob = torch.softmax(cl, -1)
        p_wait += prob[..., net.hand_size][rows].tolist()
        is_wait += (card[rows] == net.hand_size).tolist()
        played = rows & (card < net.hand_size)
        if played.any():
            pick = cl[..., : net.hand_size].argmax(-1)
            play_hit += int((pick[played] == card[played]).sum())
            play_n += int(played.sum())
        if prow.any():
            pc = pl.argmax(-1)
            cell_hit += int((pc[prow] == cell[prow]).sum())
            cell_n += int(prow.sum())
            a, b = pc[prow], cell[prow]
            dist += torch.hypot((a % BOARD_W - b % BOARD_W).float(), (a // BOARD_W - b // BOARD_W).float()).tolist()
    net.train()
    return {"card_at_plays": play_hit / max(1, play_n), "plays": play_n,
            "play_vs_wait_auc": roc_auc_score(np.array(is_wait) == 0, 1 - np.array(p_wait)) if len(set(is_wait)) > 1 else None,
            "cell_exact": cell_hit / max(1, cell_n), "cell_dist_median_tiles": float(np.median(dist)) if dist else None,
            "cells": cell_n}


def load_demos(path: Path) -> list[dict]:
    """The defence-school episodes: Cole's all, the script's at or above its template's median score."""
    eps = []
    for f in sorted(path.glob("*.npz")):
        d = np.load(f)
        src, rest = f.stem.split("_", 1)
        eps.append({"file": str(f), "n": len(d["card"]), "split": "train", "w": float(d["weight"][0]),
                    "result": None, "src": "cole" if src.startswith("cole") else src,
                    "tpl": rest.rsplit("_", 1)[0], "score": float(d["result"])})
    med = {}
    for e in eps:
        if e["src"] == "script":
            med.setdefault(e["tpl"], []).append(e["score"])
    med = {k: float(np.median(v)) for k, v in med.items()}
    # and never below -2 (the script's better half of the emergency still loses about a tower: not worth copying)
    return [e for e in eps if e["src"] != "script" or (e["score"] >= med[e["tpl"]] and e["score"] > -2.0)]


def main():
    global SHARDS, WINDOW, BURN_IN, PLAY_WEIGHT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = one window per training episode")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="data/models/bc_v1")
    ap.add_argument("--eval-eps", type=int, default=0)
    ap.add_argument("--init", default=None, help="continue from this checkpoint (model_weights.pth)")
    ap.add_argument("--shards", default=str(SHARDS))
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--play-weight", type=float, default=PLAY_WEIGHT)
    ap.add_argument("--demos", default=None, help="defence-school episodes (scripts/build_defence_demos.py)")
    ap.add_argument("--demo-share", type=float, default=0.3, help="share of every batch drawn from --demos")
    a = ap.parse_args()
    SHARDS, WINDOW, BURN_IN, PLAY_WEIGHT = Path(a.shards), a.window, a.burn_in, a.play_weight
    torch.manual_seed(77)
    rng = np.random.default_rng(77)
    device = torch.device("cpu")
    eps = load_meta()
    train = [e for e in eps if e["split"] == "train"]
    held = [e for e in eps if e["split"] == "heldback"]
    print(f"episodes: {len(train)} train ({sum(e['n'] for e in train)} decisions), {len(held)} held-back "
          f"({sum(e['n'] for e in held)}); winners weighted {WIN_WEIGHT} "
          f"({sum(e['result'] == 'win' for e in train)} of {len(train)} train episodes are winners)", flush=True)
    demos = load_demos(Path(a.demos)) if a.demos else []
    n_demo = int(round(BATCH * a.demo_share)) if demos else 0
    if demos:
        print(f"demos: {len(demos)} episodes ({sum(e['n'] for e in demos)} decisions; "
              f"{sum(e['src'] == 'cole' for e in demos)} of Cole's), {n_demo} of every {BATCH} windows", flush=True)
    net = MicroRoyaleNet().to(device)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["model"])
        print(f"continuing from {a.init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("held-back before training:", evaluate(net, held, device, a.eval_eps), flush=True)
    history, best = [], -1e9
    for epoch in range(a.epochs):
        t0 = time.time()
        steps = a.steps_per_epoch or max(1, len(train) // BATCH)
        lc = lp = 0.0
        for s in range(steps):
            batch = [train[i] for i in rng.choice(len(train), BATCH - n_demo, replace=False)]
            batch += [demos[i] for i in rng.choice(len(demos), n_demo, replace=False)] if n_demo else []
            l_card, l_cell, _ = batch_loss(net, batch, rng, device)
            loss = l_card + l_cell
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            lc += float(l_card.detach())
            lp += float(l_cell.detach())
        ev = evaluate(net, held, device, a.eval_eps)
        history.append({"epoch": epoch, "card_loss": lc / steps, "cell_loss": lp / steps, **ev})
        print(f"epoch {epoch}: card loss {lc / steps:.3f}, cell loss {lp / steps:.3f}, held-back {ev} "
              f"({time.time() - t0:.0f} s)", flush=True)
        torch.save({"model": net.state_dict(), "bc_epoch": epoch, "bc_history": history}, out / "last.pth")
        if demos:  # the defence school is judged by the situation exam, so every epoch is kept for it
            torch.save({"model": net.state_dict(), "bc_epoch": epoch}, out / f"epoch{epoch:02d}.pth")
        # the best held-back checkpoint (card at plays + play/wait AUC - cell distance / 20): placement over-fits
        # after ~70 epochs (v3: train cell loss falling, held-back distance rising)
        score = ev["card_at_plays"] + (ev["play_vs_wait_auc"] or 0) - (ev["cell_dist_median_tiles"] or 20) / 20
        if epoch == 0 or score > best:
            best = score
            torch.save({"model": net.state_dict(), "bc_epoch": epoch, "bc_history": history}, out / "model_weights.pth")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {out / 'model_weights.pth'}")


if __name__ == "__main__":
    main()
