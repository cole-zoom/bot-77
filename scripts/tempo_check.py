"""The tempo templates' reference check (docs/plan-elixir-spells.md P3, 2026-10-29): each pair is useless if the
scoring can't tell its halves apart. Five scripted players on every tempo template, the same starts for each:

  never   never plays
  dump    every affordable troop at the bridge as soon as it can (bot77.rl.situations.tempo_dump)
  chill   the tempo policy forced to chill (hold to 9.5, then one cheap / back play; defends on a threat)
  press   the tempo policy forced to press (win condition at the bridge, support behind; defends on a threat)
  tempo   the tempo policy by the lead (+3 even, 0 desperate: Cole) - the one the demos come from

Orderings it must produce (the plan's): chill > dump and chill > never on the chill halves; press > chill and
press > never on the press halves; tempo wins both halves of a pair (>= every other player, within 0.01).
spell_finish (a single): tempo (which spells the tower) > chill and > never.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/tempo_check.py [--n 40] [--seed0 0] [--workers 10]

Seeds 0.. are training seeds (tune here, if anywhere); --seed0 10000 reports on the exam's held-out seeds.
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
PLAYERS = ["never", "dump", "chill", "press", "tempo"]
_S: dict = {}


def setup():
    os.environ.update({"CLASH_DECK": "2,81,5,43,9,95,7,3", "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import python_ai  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    _S.update(torch=torch, sit=sit, CloneEnv=CloneEnv)


def run(args):
    tpl_name, seed, player = args
    if not _S:
        setup()
    sit, torch = _S["sit"], _S["torch"]
    env = _S["CloneEnv"]({"scenario_seed": 0, "decision_ticks": 1})
    env.forced_template = sit.BY_NAME[tpl_name]
    env.rng = np.random.default_rng(seed)
    env.scenario_rng = env.rng
    torch.manual_seed(seed)
    env.reset()
    played: set = set()
    plays = []
    while True:
        if player == "never":
            k, x, y = 4, 0.0, 0.0
        elif player == "dump":
            k, x, y = sit.tempo_dump(env, played)
        else:
            k, x, y = sit.tempo_policy(env, played, None if player == "tempo" else player)
        if k < 4:
            plays.append((round(env.situation.seconds, 1), int(env.game.get_hand()[k]), round(x, 1), round(y, 1),
                          round(float(env.game.get_elixir()), 1)))
        _, _, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                            "target_y": np.array([y], np.float32),
                                            "activate_ability_slot1": 0, "activate_ability_slot2": 0})
        if info.get("rl_situation_end") or term or trunc:
            r = env.last_situation
            return {"template": tpl_name, "seed": seed, "player": player, "score": r["score"],
                    "lost": r["tower_hp_lost"], "dealt": r["tower_hp_dealt"], "spent": r["elixir_spent"],
                    "leaked": r["elixir_leaked"], "plays": plays, "opp_plays": r.get("opp_plays", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None, help="every episode (jsonl)")
    a = ap.parse_args()
    if a.out:
        a.out = str(Path(a.out).resolve())  # before setup() changes directory
    setup()
    sit = _S["sit"]
    tpls = [t.__name__ for t in sit.TEMPO_TEMPLATES + sit.HELD_OUT_TEMPLATES]  # the held-out pair still reported
    jobs = [(t, a.seed0 + i, p) for t in tpls for i in range(a.n) for p in PLAYERS]
    with Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=4)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
    m = {(t, p): np.mean([r["score"] for r in res if r["template"] == t and r["player"] == p]) for t in tpls for p in PLAYERS}
    stat = {(t, p): [np.mean([r[k] for r in res if r["template"] == t and r["player"] == p]) for k in ("lost", "dealt", "spent", "leaked")]
            for t in tpls for p in PLAYERS}
    print(f"seeds {a.seed0}..{a.seed0 + a.n - 1}; score (lost / dealt tower HP, spent / leaked elixir), means")
    print(f"{'template':20s}" + "".join(f"{p:>30s}" for p in PLAYERS))
    for t in tpls:
        print(f"{t:20s}" + "".join(f"{m[t, p]:+7.2f} ({stat[t, p][0]:4.0f}/{stat[t, p][1]:4.0f}, {stat[t, p][2]:4.1f}/{stat[t, p][3]:4.1f})"
                                   for p in PLAYERS))
    ok_all = True
    print("\norderings:")
    chill_halves = [c.__name__ for c, _ in sit.TEMPO_PAIRS]
    press_halves = [p.__name__ for _, p in sit.TEMPO_PAIRS]
    for t in tpls:
        checks = []
        if t in chill_halves:
            checks = [("chill > dump", m[t, "chill"] > m[t, "dump"]), ("chill > never", m[t, "chill"] > m[t, "never"])]
        elif t in press_halves:
            checks = [("press > chill", m[t, "press"] > m[t, "chill"]), ("press > never", m[t, "press"] > m[t, "never"])]
        else:
            checks = [("tempo > chill", m[t, "tempo"] > m[t, "chill"]), ("tempo > never", m[t, "tempo"] > m[t, "never"])]
        best = max(m[t, p] for p in PLAYERS)
        if t in chill_halves + press_halves:  # a pair: the same scripted policy wins both halves
            checks.append(("tempo wins", m[t, "tempo"] >= best - 0.01))
        else:  # a single (spell_finish): the plan's ordering only; tempo vs the best player reported, not gated
            checks.append((f"(tempo - best {m[t, 'tempo'] - best:+.2f})", True))
        ok = all(c[1] for c in checks)
        ok_all &= ok
        print(f"  {t:20s} {'PASS' if ok else 'FAIL'}  " + ", ".join(f"{n} {'ok' if v else 'NO'}" for n, v in checks))
    print(f"\nall orderings: {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
