"""Defence-school demonstrations (2026-10-28, after RL round v4; Cole: "build that out"): imitation data on the defence
situations, in the imitation shards' format, so scripts/train_bc.py can mix them with the pro data.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/build_defence_demos.py [--seeds 300] [--workers 6]
  -> data/cache/bc/defence_t01/<source>_<template>_<seed>.npz   (gitignored)

Why: RL rounds v3 / v4 never found the cheap answers by trial (one Mini P.E.K.K.A low, then wait): v4 spends about
twice the scripted defence's elixir and scores -1.19 on the exam against its -0.61. Shown the answers, the network
starts RL knowing them.

Two teachers, every 0.1 s (decision_ticks 1, the pace the network plays at):
- the scripted defence (bot77.rl.situations.scripted_defence) on every training template, seeds 0..N (the exam uses
  10_000+, so its states stay held out); every episode is saved with its score (`result`), and the trainer keeps the
  better ones (the script loses the emergency and most supported pushes: a fixed bar would leave those with no demos);
- Cole's play-page games (data/live/situation_play.jsonl, all of them: "take all of my takes and use them as answer
  keys or positive signal"), replayed from their exact starts as scripts/rescore_cole.py does, weight --cole-weight.
Per decision the network's observation (float16), the hand slot played (4 = wait) and its board cell (the fork's
binning, -1 for a wait), the same fields as data/cache/bc/shards_t01.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"
OUT = ROOT / "data/cache/bc/defence_t01"
DECK = "2,81,5,43,9,95,7,3"
BOARD_W, BOARD_H = 18, 34
_ENV: dict = {}


def cell_from_xy(x: float, y: float) -> int:
    """The fork's binning, as scripts/build_bc_dataset.py."""
    return int(np.clip(int(y), 0, BOARD_H - 1)) * BOARD_W + int(np.clip(int(x), 0, BOARD_W - 1))


def setup():
    os.environ.update({"CLASH_DECK": DECK, "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import python_ai  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    from bot77.sim import cards as sc
    _ENV.update(torch=torch, sit=sit, CloneEnv=CloneEnv,
                ids={c["name"]: i for i, c in sc.engine_cards().items() if not c.get("evolution")})


def episode(tpl_name: str, seed: int, plays=None):
    """One situation from its exact start; the script plays unless `plays` (Cole's, by time) is given."""
    torch, sit = _ENV["torch"], _ENV["sit"]
    env = _ENV["CloneEnv"]({"scenario_seed": 0, "decision_ticks": 1})
    env.forced_template = {t.__name__: t for t in sit.TEMPLATES}[tpl_name]
    env.rng = np.random.default_rng(seed)
    env.scenario_rng = env.rng
    torch.manual_seed(seed)
    env.reset()
    at = {int(round(p["t"] * 10)): p for p in plays or [] if p.get("landed", True)}
    obs, card, cell = [], [], []
    i = 0
    while True:
        i += 1
        k, x, y = 4, 0.0, 0.0
        if plays is None:
            k, x, y = sit.scripted_defence(env)
        elif i in at:
            p = at[i]
            cid = _ENV["ids"].get(p["card"])
            hand = list(env.game.get_hand())
            if cid is not None and cid not in hand:  # the cycle didn't bring his card: put it in hand (as rescore_cole)
                env.game.set_hand_for_team(0, [cid] + [c for c in hand if c != cid][:3])
                hand = list(env.game.get_hand())
            if cid in hand:
                k, x, y = hand.index(cid), float(p["x"]), float(p["y"])
        # the observation the choice was made on, read after any hand change
        obs.append(np.asarray(env.game.get_observation_for_team(0), np.float16))
        card.append(k)
        cell.append(cell_from_xy(x, y) if k < 4 else -1)
        _, _, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                            "target_y": np.array([y], np.float32),
                                            "activate_ability_slot1": 0, "activate_ability_slot2": 0})
        if info.get("rl_situation_end") or term or trunc:
            return env.last_situation, np.stack(obs), np.array(card, np.int64), np.array(cell, np.int64)


def job(args):
    source, tpl_name, seed, plays, weight, min_score, out = args
    path = Path(out) / f"{source}_{tpl_name}_{seed}.npz"  # Cole's sources carry the game's index: one file a game
    if not _ENV:
        setup()
    res, obs, card, cell = episode(tpl_name, seed, plays)
    if res is None or (plays is None and res["score"] < min_score):
        return source, tpl_name, None if res is None else res["score"], 0
    np.savez_compressed(path, obs=obs, card=card, cell=cell, t=np.arange(len(card), dtype=np.float32) / 10,
                        weight=np.full(len(card), weight, np.float32), split=np.array("train"),
                        side=np.array(source), result=np.array(f"{res['score']:.3f}"))
    return source, tpl_name, res["score"], len(card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300, help="scripted episodes per template (seeds 0..N-1)")
    ap.add_argument("--min-score", type=float, default=-99.0,
                    help="keep a scripted episode only at or above this (default all: each file records its score in "
                         "`result`, and scripts/train_bc.py filters at load)")
    ap.add_argument("--cole-weight", type=float, default=3.0)
    ap.add_argument("--no-cole", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    setup()
    sit = _ENV["sit"]
    jobs = [("script", t.__name__, s, None, 1.0, a.min_score, a.out) for t in sit.TEMPLATES for s in range(a.seeds)]
    if not a.no_cole:
        for n, g in enumerate(json.loads(l) for l in open(ROOT / "data/live/situation_play.jsonl")):
            name = sit.BY_NAME[g["situation"]].__name__
            # one file per game (the same start can have several of his takes)
            jobs.append((f"cole{n:03d}", name, int(g["seed"]), g["plays"], a.cole_weight, a.min_score, a.out))
    t0, kept, rows, by = time.time(), 0, 0, {}
    with Pool(a.workers) as pool:
        for n, (source, name, score, r) in enumerate(pool.imap_unordered(job, jobs), 1):
            src = "cole" if source.startswith("cole") else "script"
            b = by.setdefault((src, name), [0, 0, []])
            b[0] += 1
            b[1] += r > 0
            if score is not None:
                b[2].append(score)
            kept += r > 0
            rows += r
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} episodes, {kept} kept, {rows} rows, {time.time() - t0:.0f} s", flush=True)
    print(f"done: {kept} of {len(jobs)} episodes kept, {rows} rows, {time.time() - t0:.0f} s -> {a.out}")
    for (src, name), (n, k, sc) in sorted(by.items()):
        print(f"  {src:6s} {name:16s} {k:4d} / {n:4d} kept   mean score {np.mean(sc) if sc else float('nan'):+.2f}")


if __name__ == "__main__":
    main()
