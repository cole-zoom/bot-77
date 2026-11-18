"""Does the live path see what the simulator sees? (2026-11-18; idea from Hasty-CR's tests/test_rl_policy.py, MIT,
github.com/hastylmao/Hasty-CR: "get it wrong and nothing throws".)

Sim demos and self-play train on the engine's own observation (get_observation_for_team on a running match). Live
play and the pro shards build it differently: the screen read is mirrored into a fresh engine every decision
(bot77.sim.mirror), which knows only what a reader can see. This plays engine matches with random legal plays for
both sides and, every --every seconds, rebuilds the moment from a perfect reader (true units, positions, HP, towers,
hand, both elixirs, both decks, the opponent's plays with their times) and compares the two observations section by
section (bot77.live.rebuild.section_of). What still differs is the mirror itself, not perception.

  uv run --no-sync python scripts/obs_parity.py [--matches 6] [--every 2.0] [--unit-hp] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bot77.features.frames import RIVER_Y  # noqa: E402
from bot77.live.rebuild import CYC0, HAND0, NCARD, SCAL0, SPATIAL, H, W, section_of  # noqa: E402
from bot77.sim import cards as sc  # noqa: E402
from bot77.sim.engine import Sim  # noqa: E402
from bot77.sim.mirror import mirror, observation  # noqa: E402
from bot77.sim.scenario import _body_max_hp  # noqa: E402

OWN = ["Hog Rider", "Mini P.E.K.K.A", "Musketeer", "Cannon", "Fireball", "The Log", "Skeletons", "Ice Spirit"]
OPPS = [["Giant", "Mini P.E.K.K.A", "Musketeer", "Goblin Hut", "Fireball", "Arrows", "Skeletons", "Minions"],
        ["Knight", "Archers", "Valkyrie", "Baby Dragon", "Zap", "Fireball", "Bomb Tower", "Goblin Gang"],
        ["P.E.K.K.A", "Battle Ram", "Electro Wizard", "Bandit", "Poison", "Zap", "Minions", "Royal Ghost"]]


def ids(names: list[str]) -> list[int]:
    out = [sc.engine_id(n) for n in names]
    missing = [n for n, e in zip(names, out) if e is None]
    if missing:
        sys.exit(f"not in the engine: {missing}")
    return out


def perfect_read(sim: Sim, unit_hp: bool) -> tuple[list[dict], dict, int]:
    """What an error-free screen reader would hand the live path: every unit on the board (feet, our frame), spells
    and carried bodies left out as the live path leaves them, towers as HP fractions. -> (units, towers, spawned
    bodies left out: a spawned body has no card id, so the mirror can't name it)."""
    units, spawned = [], 0
    for u in sim.units():
        if u["spell"] or u["attached_to"] >= 0:
            continue
        if u["card_id"] < 0:
            spawned += 1
            continue
        d = {"card": u["card"], "cls": "", "team": u["team"], "x": u["x"], "y": u["y"]}
        if unit_hp:
            d["hp_frac"] = float(np.clip(u["hp"] / max(1.0, _body_max_hp(u["card_id"], None)), 0, 1))
        units.append(d)
    towers = {k: v["frac"] for k, v in sim.towers().items()}
    return units, towers, spawned


def play_random(sim: Sim, team: str, rng: np.random.Generator) -> int | None:
    """One random legal play for `team` from its hand, if affordable. -> the card played or None."""
    hand = [c for c in sim.hand(team) if sc.cost(c) is not None and sc.cost(c) <= sim.elixir(team)]
    if not hand:
        return None
    c = int(rng.choice(hand))
    for _ in range(10):
        x = float(rng.uniform(1, 17))
        y = float(rng.uniform(RIVER_Y + 2, 30)) if team == "own" else float(rng.uniform(3, RIVER_Y - 2))
        if sim.play(c, x, y, team, force=False).played:
            return c
    return None


def compare(native: np.ndarray, rebuilt: np.ndarray, tol: float = 1e-3) -> dict[str, dict]:
    """Per section: cells that differ, cells in the section, mean |difference| over the differing cells."""
    d = np.abs(native - rebuilt)
    out: dict[str, dict] = defaultdict(lambda: {"diff": 0, "cells": 0, "abs": 0.0})
    for c in range(len(native)):
        s = out[section_of(c)]
        s["cells"] += 1
        if d[c] > tol:
            s["diff"] += 1
            s["abs"] += float(d[c])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between compared moments")
    ap.add_argument("--play-every", type=float, default=2.5, help="mean seconds between a side's tries to play")
    ap.add_argument("--unit-hp", action="store_true", help="give the mirror troop HP (BOT77_LIVE_HP, as live --hp)")
    ap.add_argument("--json")
    a = ap.parse_args()
    own = ids(OWN)
    tot: dict[str, dict] = defaultdict(lambda: {"diff": 0, "cells": 0, "abs": 0.0, "moments_with_diff": 0})
    moments, failed, spawned_total, units_total = 0, 0, 0, 0
    examples: dict[str, list] = defaultdict(list)
    for m in range(a.matches):
        rng = np.random.default_rng(m)
        opp = ids(OPPS[m % len(OPPS)])
        sim = Sim.new(own, opp)
        sim.reset()
        opp_played: list[tuple[int, float]] = []  # (engine id, match time)
        next_cmp = a.every
        while not sim.over and sim.t < 175:
            for team in ("own", "opp"):
                if rng.random() < 0.1 / a.play_every:
                    c = play_random(sim, team, rng)
                    if c is not None and team == "opp":
                        opp_played.append((c, sim.t))
            if sim.step(1):
                break
            if sim.t + 1e-6 < next_cmp:
                continue
            next_cmp += a.every
            native = observation(sim)
            units, towers, spawned = perfect_read(sim, a.unit_hp)
            spawned_total += spawned
            units_total += len(units)
            reb = mirror(own, opp, sim.t, sim.hand("own"), sim.elixir("own"), sim.elixir("opp"), units, towers,
                         [(e, sim.t - t) for e, t in opp_played], unit_hp=a.unit_hp)
            if reb is None:
                failed += 1
                continue
            rebuilt = observation(reb)
            moments += 1
            for s, v in compare(native, rebuilt).items():
                t = tot[s]
                t["diff"] += v["diff"]; t["cells"] += v["cells"]; t["abs"] += v["abs"]
                t["moments_with_diff"] += v["diff"] > 0
                if v["diff"] and len(examples[s]) < 3:
                    examples[s].append({"match": m, "t": round(sim.t, 1), "cells": v["diff"]})
        print(f"match {m}: t={sim.t:.0f}s, {moments} moments so far", flush=True)
    print(f"\n{moments} moments compared ({failed} the mirror couldn't build); units read {units_total}, "
          f"spawned bodies left out {spawned_total}; troop HP {'on' if a.unit_hp else 'off'}\n")
    print(f"{'section':24} {'moments differing':>18} {'cells differing':>16} {'mean |diff|':>12}")
    rows = {}
    for s in sorted(tot):
        t = tot[s]
        frac_m = t["moments_with_diff"] / max(1, moments)
        frac_c = t["diff"] / max(1, t["cells"])
        mean = t["abs"] / max(1, t["diff"])
        rows[s] = {"moments_differing": round(frac_m, 3), "cells_differing": round(frac_c, 5), "mean_abs": round(mean, 4),
                   "examples": examples[s]}
        print(f"{s:24} {frac_m:>17.0%} {frac_c:>16.4%} {mean:>12.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps({"moments": moments, "failed": failed, "unit_hp": a.unit_hp,
                                            "spawned_left_out": spawned_total, "sections": rows}, indent=1))


if __name__ == "__main__":
    main()
