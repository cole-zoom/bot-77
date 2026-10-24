"""S2: replay windows of the human replays in the simulator and compare with the detections (docs/simulator.md).
-> LanceDB `sim_validation` (one row per window x metric x card x time bin) and a summary on stdout.

  uv run python scripts/sim_validate.py --run s2_base [--splits train,heldback] [--games 50]

Calibrate on train / dev, report on held-back; test and held-out are never run.
"""
import argparse
import collections
import time

import numpy as np

from bot77.sim import store
from bot77.sim.validate import DT_BINS, load_game, metrics, run_window, windows

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--splits", default="train,dev,heldback")
ap.add_argument("--games", type=int, default=0, help="at most this many games per split (0 = all)")
ap.add_argument("--towers", default="liquipedia_l11", help="bot77.sim.engine.TOWER_PRESETS")
ap.add_argument("--sight", default="strict", choices=["strict", "edge"], help="engine sight rule (calibration)")
ap.add_argument("--load-times", default="on", choices=["on", "off"], help="first-hit windup (calibration)")
ap.add_argument("--kings", default="wake", choices=["wake", "asleep"], help="wake a King whose HP bar was seen before")
args = ap.parse_args()
import clash_royale_env  # noqa: E402  (the calibration switch; process-wide)
clash_royale_env.set_sight_edge_to_edge(args.sight == "edge")
clash_royale_env.set_load_times(args.load_times == "on")
splits = args.splits.split(",")
assert not {"test", "heldout"} & set(splits), "never on test / held-out"

games = store.read("matches", "source IN ('replay_tv', 'replay_bl') AND pov_side = 'bottom'")
games = sorted((g for g in games if g["split"] in splits), key=lambda g: g["match_id"])
if args.games:
    per = collections.defaultdict(list)
    for g in games:
        per[g["split"]].append(g)
    games = [g for v in per.values() for g in v[: args.games]]

rows, n_win, t0 = [], 0, time.time()
for i, m in enumerate(games):
    g = load_game(m["match_id"], m["split"], m["source"])
    for w in windows(g):
        r = run_window(g, w, args.towers, args.kings == "wake")
        rows += metrics(g, w, r)
        n_win += 1
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(games)} games, {n_win} windows, {time.time() - t0:.0f} s", flush=True)
commit = store.engine_commit()
for r in rows:
    r["run"], r["engine"], r["towers"], r["sight"], r["kings"], r["load_times"] = (
        args.run, commit, args.towers, args.sight, args.kings, args.load_times)
store.write("sim_validation", rows, where=f"run = '{args.run}'")
print(f"{len(games)} games, {n_win} windows, {len(rows)} rows -> sim_validation (run {args.run}, engine {commit})")


def wmedian(vals, weights):
    o = np.argsort(vals)
    v, w = np.asarray(vals)[o], np.asarray(weights)[o]
    return float(v[np.searchsorted(np.cumsum(w), w.sum() / 2)])


groups = [(f"{split}, {kind}", split, kind) for split in splits for kind in ("empty start", "1-2 units at start")]
for label, split, kind in groups:
    R = [r for r in rows if r["split"] == split and (r["start_units"] == 0) == (kind == "empty start")]
    if not R:
        continue
    wins = {(r["match_id"], r["t0"]) for r in R}
    clean = {(r["match_id"], r["t0"]) for r in R if r["metric"] == "plays_refused" and r["value"] == 0}
    print(f"\n== {label}: {len(wins)} windows ({len(clean)} with every play accepted)")
    for b in range(len(DT_BINS) - 1):
        e = [r for r in R if r["metric"] == "pos_err" and r["dt_bin"] == b and (r["match_id"], r["t0"]) in clean]
        lane = [r for r in R if r["metric"] == "same_lane" and r["dt_bin"] == b and (r["match_id"], r["t0"]) in clean]
        cnt = [r for r in R if r["metric"] == "count_err" and r["dt_bin"] == b and (r["match_id"], r["t0"]) in clean]
        if e:
            print(f"  +{DT_BINS[b]:>4}-{DT_BINS[b + 1]:<4} s  position error median {wmedian([r['value'] for r in e], [r['n'] for r in e]):.2f}"
                  f" (pairs {sum(r['n'] for r in e)})  same lane {np.average([r['value'] for r in lane], weights=[r['n'] for r in lane]):.1%}"
                  f"  count error {np.average([r['value'] for r in cnt], weights=[r['n'] for r in cnt]):.2f}")
    for metric in ("winner_agree", "winner_agree_onesided"):
        v = [r["value"] for r in R if r["metric"] == metric and (r["match_id"], r["t0"]) in clean]
        if v:
            print(f"  {metric}: {np.mean(v):.1%} of {len(v)}")
    life = [r["value"] for r in R if r["metric"] == "life_diff" and (r["match_id"], r["t0"]) in clean]
    if life:
        print(f"  life time sim - replay: median {np.median(life):+.2f} s, |diff| median {np.median(np.abs(life)):.2f} s, n {len(life)}")
    for side in ("own", "opp"):
        real = {(r["match_id"], r["t0"]): r["value"] for r in R if r["metric"] == f"tower_dmg_{side}_real"}
        sim = {(r["match_id"], r["t0"]): r["value"] for r in R if r["metric"] == f"tower_dmg_{side}_sim"}
        keys = [k for k in real if k in sim and k in clean]
        if not keys:
            continue
        a, b = np.array([real[k] for k in keys]), np.array([sim[k] for k in keys])
        hit = (a > 0.02) | (b > 0.02)
        big = a > 0.10
        print(f"  tower damage to {side} (princess-tower units, {len(keys)} windows with tower HP): real total "
              f"{a.sum():.2f}, sim {b.sum():.2f}; damaged in either: {hit.sum()}, both agree it was damaged "
              f"{np.mean((a > 0.02) == (b > 0.02)):.0%}; |sim - real| median {np.median(np.abs(b - a)[hit]) if hit.any() else 0:.3f}"
              + (f"; where real > 0.10: sim/real median {np.median(b[big] / a[big]):.2f} (n {big.sum()})" if big.any() else ""))
    ref = [r for r in R if r["metric"] == "plays_refused"]
    print(f"  plays refused: {sum(r['value'] for r in ref):.0f} of {sum(r['n'] for r in ref)}")
