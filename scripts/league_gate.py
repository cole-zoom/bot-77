"""The eval gate for a self-play round (scripts/selfplay_league.py): does the candidate replace the start?

  uv run --no-sync --with gymnasium==1.3.0 python scripts/league_gate.py --cand data/models/league_v1/historical_checkpoints/ep03000.pth \
      [--base data/models/bc_school_v7/epoch03.pth] [--n 200] [--workers 3]

Two checks, both on held-out seeds:
  1. Head to head, paired by seed. match_eval puts the checkpoint on team 0, which wins well over half against an
     identical opponent (2026-10-30: 67% for the 8-2 model vs itself), so the candidate's win rate against the base
     means nothing alone. The base plays the same seeds against itself; the candidate passes when it wins more of
     them (McNemar on the seeds one won and the other lost, p < --p).
  2. The five deck bots (deck_vs_model.py, 30 games each): the candidate's total may not fall more than --deck-slack
     wins below the base's.
Both results land in $OUT (default: next to the candidate) as gate_<name>.json.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ["uv", "run", "--no-sync", "--with", "gymnasium==1.3.0", "python"]


def h2h(ckpt, vs, n, workers, out):
    if not out.exists():
        subprocess.run([*RUN, "scripts/match_eval.py", "--ckpt", ckpt, "--vs", vs, "--n", str(n), "--workers",
                        str(workers), "--json", str(out)], cwd=ROOT, check=True)
    return {r["seed"]: r["result"] for r in map(json.loads, out.read_text().splitlines())}


def decks(ckpt, workers, out):
    if not out.exists():
        subprocess.run([*RUN, "scripts/deck_vs_model.py", "--ckpt", ckpt, "--n", "30", "--workers", str(workers),
                        "--json", str(out)], cwd=ROOT, check=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    per = {}
    for r in rows:
        per[r["deck"]] = per.get(r["deck"], 0) + int(r["result"] > 0.5)
    return per


def mcnemar(b, c):
    """Two-sided exact McNemar p on b (only the candidate won) vs c (only the base won)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", default="data/models/bc_school_v7/epoch03.pth")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--deck-slack", type=int, default=6, help="about one standard error of a 150-game total")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cand, base = Path(a.cand), Path(a.base)
    out = Path(a.out) if a.out else cand.parent
    out.mkdir(parents=True, exist_ok=True)
    tag = lambda p: f"{p.parent.name}_{p.stem}"  # noqa: E731
    mirror = h2h(str(base), str(base), a.n, a.workers, base.parent / f"h2h_mirror_n{a.n}.jsonl")
    vs = h2h(str(cand), str(base), a.n, a.workers, out / f"h2h_{tag(cand)}_vs_{tag(base)}.jsonl")
    seeds = sorted(set(mirror) & set(vs))
    b = sum(vs[s] > 0.5 and not mirror[s] > 0.5 for s in seeds)
    c = sum(mirror[s] > 0.5 and not vs[s] > 0.5 for s in seeds)
    p = mcnemar(b, c)
    w_cand = sum(vs[s] > 0.5 for s in seeds) / len(seeds)
    w_base = sum(mirror[s] > 0.5 for s in seeds) / len(seeds)
    d_base = decks(str(base), a.workers, base.parent / "gate_decks.jsonl")
    d_cand = decks(str(cand), a.workers, out / f"gate_decks_{tag(cand)}.jsonl")
    passed_h2h = b > c and p < a.p
    passed_decks = sum(d_cand.values()) >= sum(d_base.values()) - a.deck_slack
    res = {"cand": str(cand), "base": str(base), "seeds": len(seeds), "cand_vs_base": w_cand, "base_vs_base": w_base,
           "only_cand_won": b, "only_base_won": c, "mcnemar_p": p, "decks_cand": d_cand, "decks_base": d_base,
           "pass_h2h": passed_h2h, "pass_decks": passed_decks, "pass": passed_h2h and passed_decks}
    (out / f"gate_{tag(cand)}.json").write_text(json.dumps(res, indent=1))
    print(f"head to head (team 0 vs the base, {len(seeds)} seeds): candidate {w_cand:.0%}, base itself {w_base:.0%}; "
          f"only the candidate won {b}, only the base {c}, McNemar p {p:.3f} -> {'pass' if passed_h2h else 'fail'}")
    print(f"deck bots (wins of 30): candidate {d_cand} = {sum(d_cand.values())}, base {d_base} = "
          f"{sum(d_base.values())} -> {'pass' if passed_decks else 'fail'}")
    print("GATE", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()
