"""Replay Cole's play-page games (data/live/situation_play.jsonl) from their exact starts with his plays at their exact
ticks, and score them (and the scripted defence) under the current situation scoring (2026-10-28).

  uv run --no-sync --with gymnasium==1.3.0 python scripts/rescore_cole.py [--since N] [--out data/live/cole_rescored.jsonl]

The play page starts a game with the template's generator and torch both seeded with the game's seed and steps every
0.1 s, a play logged at time t having gone in on step round(10 t); the replay does the same, so on an unchanged
engine it would reproduce his recorded game; the "repro" line says how often it does. Few games played before
2026-10-28 ~10:00 do: the engine's hand cycle was unseeded then (now seeded from the start), so his later cards came in a
different order (they're put in hand here when the cycle hasn't brought them) and the opponent's hand differed.
Scores: v1 = the formula he played under, v2 = the board-aware one (Cole's six "good on the board" games).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=0, help="skip the first N games of the log")
    ap.add_argument("--out", default=None, help="write each game with its new scores here (jsonl)")
    ap.add_argument("--no-scripted", action="store_true")
    a = ap.parse_args()
    os.environ.update({"CLASH_DECK": "2,81,5,43,9,95,7,3", "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    games = [json.loads(l) for l in open(ROOT / "data/live/situation_play.jsonl")][a.since:]
    sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
    os.chdir(ROOT / "third_party" / "clash-royale-env")
    import python_ai  # noqa: F401
    import torch

    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    from bot77.sim import cards as sc
    torch.set_num_threads(2)  # as the play page (sampling follows the float results)
    ids = {c["name"]: i for i, c in sc.engine_cards().items() if not c.get("evolution")}

    def run(tpl, seed, plays=None, ticks=1):
        env = CloneEnv({"scenario_seed": 0, "decision_ticks": ticks})
        env.forced_template = tpl
        env.rng = np.random.default_rng(seed)
        env.scenario_rng = env.rng
        torch.manual_seed(seed)
        env.reset()
        at = {int(round(p["t"] * 10)): p for p in plays or [] if p.get("landed", True)}
        i = 0
        while True:
            i += 1
            k, x, y = 4, 0.0, 0.0
            if plays is None:
                k, x, y = sit.scripted_defence(env)
            elif i in at:
                p = at[i]
                hand = list(env.game.get_hand())
                cid = ids.get(p["card"])
                if cid is not None and cid not in hand:
                    env.game.set_hand_for_team(0, [cid] + [c for c in hand if c != cid][:3])
                    hand = list(env.game.get_hand())
                if cid in hand:
                    k, x, y = hand.index(cid), float(p["x"]), float(p["y"])
            _, _, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                "target_y": np.array([y], np.float32),
                                                "activate_ability_slot1": 0, "activate_ability_slot2": 0})
            if info.get("rl_situation_end") or term or trunc:
                return env.last_situation

    out = open(ROOT / a.out, "w") if a.out else None
    rep = 0
    print(f"{'situation':22s} {'seed':>5s}  {'you then':>8s} {'you v1':>7s} {'you v2':>7s}  {'scr v1':>7s} {'scr v2':>7s}")
    for g in games:
        tpl = sit.BY_NAME[g["situation"]]
        you = run(tpl, g["seed"], g["plays"])
        scr = None if a.no_scripted else run(tpl, g["seed"], None, ticks=10)
        then = g["result"]["you"]["score"]
        rep += abs(you["tower_hp_lost"] - g["result"]["you"]["tower_hp_lost"]) < 1
        nan = float("nan")
        print(f"{g['situation']:22s} {g['seed']:5d}  {then:+8.2f} {you['score_v1']:+7.2f} {you['score']:+7.2f}  "
              f"{scr['score_v1'] if scr else nan:+7.2f} {scr['score'] if scr else nan:+7.2f}")
        if out:
            out.write(json.dumps({**g, "rescored": {"you": you, "scripted": scr}}, default=float) + "\n")
    print(f"repro (same tower HP lost as recorded): {rep}/{len(games)}")


if __name__ == "__main__":
    main()
