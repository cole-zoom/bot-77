"""Measure, from the pro replays, when a defender's troop pulls (kites) a troop-targeting attacker off its walk to the
tower, and fit the attacker's acquisition distance (its sight) to it (2026-10-28; Cole: a Prince in the right lane is
kited to the left lane by a Mini P.E.K.K.A dropped at engine tile (8, 12.5); in the engine it walks to its tower).

  uv run --no-sync python scripts/measure_kites.py [--workers 4] [--examples 8]

Why: CombatEntity::findTarget makes a unit a candidate when (centre distance - its targeting radius) <= sight (5.5
for the Prince, Mini P.E.K.K.A, Knight and most melee) and the nearest by that rank wins, towers competing at 1.5 (fork
602a950). At Cole's tile the kiting troop is ~6.6 tiles from the Prince as it crosses.

What it does, per human replay (sources replay_tv / replay_bl, detections run v5l_e15, ~4 fps; the coordinate
conversion and track linking of scripts/measure_pulls.py, attackers mirrored to walk toward low y):
- attacker tracks of ground melee troop-targeters (ATTACKERS); defender tracks of ground troops (the cards table's
  type 'troop', flyers left out: the attackers can't hit air);
- an event: a defender troop track appears while an attacker is on or crossing the river (y < 19) and still >=
  LOCK_DIST from its lane tower; clean ones only: no other defender ground troop within CLEAN tiles of the attacker
  and no defender building up within BUILDING_CLEAR tiles at that moment;
- outcome from the tracks afterwards (until the attacker reaches its tower or WINDOW s): PULLED if the attacker's x
  moves >= DEVIATE toward the defender and it comes within MEET of it; NOT PULLED if it reaches its tower (REACH_TOWER)
  having moved < 0.5 toward the defender while the defender was still alive; anything else is dropped;
- the defender placed in the attacker's own path (|dx| < OFF_LANE at the start) can't tell a pull from a walk-in and
  is left out of the fit (reported);
- the fit, the engine's rule at the informative moment: for a pull, the distance when the attacker's turn began (its
  sideways move toward the defender passed TURN and kept growing); for a non-pull, the closest the two came. Pulled
  iff that distance - R_TROOP <= S (and, for a pull, nearer than the tower at 1.5); S swept per attacker card and
  pooled; accuracy for the engine's sight vs the best S, and for edge-to-edge sight (S + both radii), on train + dev,
  then held-back. (Judging over the whole path is uninformative: a pull ends with the two meeting, so every S fits.)
  Never on test / held-out.
"""
import argparse
import bisect
import collections
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_pulls import BUILDINGS, LOCK_DIST, REACH_TOWER, RUN, TOWERS, _engine, _sites, _tracks  # noqa: E402

from bot77.sim import store  # noqa: E402

# engine sight (get_card_stats 'sight_range'), ground-only melee / short-range troop-targeters
ATTACKERS = {"Prince": 5.5, "Dark Prince": 5.5, "Mini P.E.K.K.A": 5.5, "Knight": 5.5, "Valkyrie": 5.5,
             "P.E.K.K.A": 5.0, "Mega Knight": 5.5, "Lumberjack": 5.5, "Bandit": 6.0, "Barbarians": 5.5,
             "Elite Barbarians": 5.5, "Bowler": 4.0}
FLYERS = {"Minions", "Minion Horde", "Bats", "Mega Minion", "Baby Dragon", "Inferno Dragon", "Electro Dragon",
          "Skeleton Dragons", "Phoenix", "Flying Machine", "Balloon", "Lava Hound", "Lava Pups", "Skeleton Barrel"}
R_TROOP = 0.0  # a troop's targeting radius in the engine (Entity::getCollisionRadius is 0 for troops: rank = centre distance)
R_SELF = 0.4  # Entity::IMPLICIT_TROOP_RADIUS: what edge-to-edge sight adds per unit
CLEAN = 6.0
BUILDING_CLEAR = 8.0
WINDOW = 10.0
DEVIATE = 1.0
MEET = 2.6
OFF_LANE = 2.0


def _at(track, t):
    """The track's position at t (nearest detection within 0.6 s), or None."""
    ts = [p[0] for p in track]
    i = bisect.bisect_left(ts, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(track)), key=lambda j: abs(ts[j] - t), default=None)
    if best is None or abs(ts[best] - t) > 0.6:
        return None
    return track[best][1:]


def match_events(args):
    mid, split, troop_cards = args
    rows = store.read("detections", f"run_id = '{RUN}' AND match_id = '{mid}'")
    out = []
    for atk in ("enemy", "friendly"):
        dfd = "friendly" if atk == "enemy" else "enemy"
        conv = lambda r: {**r, **dict(zip(("x", "y"), _engine(r["x"], r["y"], atk)))}
        dfd_rows = [conv(r) for r in rows if r["team"] == dfd]
        sites = _sites([r for r in dfd_rows if r["card"] in BUILDINGS])
        dtracks = []
        for card in {r["card"] for r in dfd_rows if r["card"] in troop_cards and r["card"] not in FLYERS}:
            dtracks += [(card, tr) for tr in _tracks([r for r in dfd_rows if r["card"] == card])]
        if not dtracks:
            continue
        for acard in ATTACKERS:
            for tr in _tracks([conv(r) for r in rows if r["team"] == atk and r["card"] == acard]):
                tower = TOWERS[0] if np.median([p[1] for p in tr]) < 8.5 else TOWERS[1]
                for dcard, dtr in dtracks:
                    t0 = dtr[0][0]
                    a0 = _at(tr, t0)
                    if a0 is None or a0[1] >= 19.0 or math.dist(a0, tower) < LOCK_DIST:
                        continue
                    d0p = dtr[0][1:]
                    others = [1 for c2, t2 in dtracks if t2 is not dtr and (p := _at(t2, t0)) and math.dist(p, a0) < CLEAN]
                    blds = [s for s in sites if s[3] <= t0 <= s[4] and math.dist((s[1], s[2]), a0) < BUILDING_CLEAR]
                    if others or blds:
                        continue
                    # walk forward along the attacker's track
                    toward = math.copysign(1.0, d0p[0] - a0[0]) if abs(d0p[0] - a0[0]) > 1e-6 else 0.0
                    rule_hits = []  # (d, d_tower, lateral move toward the defender) along the way, for the fit
                    outcome, deviate_max, met = None, 0.0, False
                    for t, x, y in tr:
                        if t < t0 or t > t0 + WINDOW:
                            continue
                        dp = _at(dtr, t)
                        d_tow = math.dist((x, y), tower)
                        lat = (x - a0[0]) * toward
                        deviate_max = max(deviate_max, lat)
                        if dp is not None:
                            d = math.dist((x, y), dp)
                            rule_hits.append((d, d_tow, lat))
                            if d <= MEET and deviate_max >= DEVIATE:
                                outcome = True
                                break
                        if d_tow <= REACH_TOWER:
                            alive = dtr[-1][0] >= t - 0.5
                            if deviate_max < 0.5 and alive:
                                outcome = False
                            break
                    if outcome is None or not rule_hits:
                        continue
                    out.append({"match_id": mid, "split": split, "t": t0, "card": acard, "defender": dcard,
                                "d0": math.dist(a0, d0p), "dx0": abs(d0p[0] - a0[0]), "d_tower0": math.dist(a0, tower),
                                "pulled": outcome, "path": rule_hits,
                                "attacker": (round(a0[0], 1), round(a0[1], 1)), "def": (round(d0p[0], 1), round(d0p[1], 1))})
    return out


TURN = 0.4  # lateral move toward the defender (tiles) taken as the start of a turn (track jitter is ~0.2-0.3)


def acquisition(e):
    """Pulled: the distance when the turn began (the last point before the attacker's sideways move toward the
    defender passed TURN and kept growing). Not pulled: the closest the two came without a turn. Either is the
    distance the attacker's sight has to be judged against."""
    path = e["path"]
    if not e["pulled"]:
        return min(d for d, _, _ in path)
    for i, (d, dt, lat) in enumerate(path):
        if lat >= TURN and all(l2 >= TURN - 0.15 for _, _, l2 in path[i:]):
            return path[max(0, i - 1)][0]
    return path[-1][0]


def rule(e, S, tower_r=1.5):
    """The engine's rule at sight S: pulled iff the defender came within S (by its footprint) while nearer than the
    tower; for a pull, judged at the moment the turn began."""
    d = acquisition(e)
    if e["pulled"]:
        dt = next(dt for dd, dt, _ in e["path"] if dd == d)
        return d - R_TROOP <= S and d - R_TROOP < dt - tower_r
    return d - R_TROOP <= S


def acc(ev, S_of):
    ok = [rule(e, S_of(e)) == e["pulled"] for e in ev]
    return sum(ok) / max(1, len(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args()
    cards = store.read("cards", "kind = 'deck_card'")
    troop_cards = {c["name"] for c in cards if c.get("type") == "troop"} | {"Barbarians", "Spear Goblins", "Goblins",
                                                                           "Skeletons", "Golemite", "Elixir Blob"}
    ms = store.read("matches", "source IN ('replay_tv', 'replay_bl') AND pov_side = 'bottom'")
    jobs = [(m["match_id"], m["split"], troop_cards) for m in ms if m["split"] in ("train", "dev", "heldback")]
    with Pool(a.workers) as p:
        ev = [e for chunk in p.imap_unordered(match_events, jobs) for e in chunk]
    in_path = [e for e in ev if e["dx0"] < OFF_LANE]
    ev = [e for e in ev if e["dx0"] >= OFF_LANE]
    fit = [e for e in ev if e["split"] != "heldback"]
    held = [e for e in ev if e["split"] == "heldback"]
    print(f"{len(jobs)} games; off-lane events: {len(fit)} (train + dev), {len(held)} held-back; "
          f"{len(in_path)} in the attacker's own path left out")
    print(f"  pulled: {sum(e['pulled'] for e in fit)} of {len(fit)} (fit set)")
    engine = lambda e: ATTACKERS[e["card"]]
    Ss = np.round(np.arange(3.0, 10.01, 0.25), 2)
    pooled = [acc(fit, lambda e, S=S: S) for S in Ss]
    best = float(Ss[int(np.argmax(pooled))])
    band = [float(S) for S, x in zip(Ss, pooled) if x >= max(pooled) - 1e-9]
    print(f"  engine sights: {acc(fit, engine):.1%} | edge-to-edge (sight + {2 * R_SELF}): "
          f"{acc(fit, lambda e: engine(e) + 2 * R_SELF):.1%} | best pooled S {best} ({max(pooled):.1%}; "
          f"equally good {band[0]}-{band[-1]})")
    print(f"  held-back ({len(held)}): engine {acc(held, engine):.1%}, S {best} {acc(held, lambda e: best):.1%}")
    for S in (5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        print(f"    S {S}: {acc(fit, lambda e, S=S: S):.1%}")
    turns = sorted(acquisition(e) for e in fit if e["pulled"])
    misses = sorted(acquisition(e) for e in fit if not e["pulled"])
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")
    print(f"  distance when a pull began: median {q(turns, .5):.1f}, 75% {q(turns, .75):.1f}, 90% {q(turns, .9):.1f}, max "
          f"{turns[-1] if turns else float('nan'):.1f} | closest without a pull: min {misses[0] if misses else float('nan'):.1f}, "
          f"10% {q(misses, .1):.1f}, median {q(misses, .5):.1f}")
    print("  per attacker (fit set): card n pulled | engine sight acc | best S (acc) | turns began at (median / max)")
    for card, n in collections.Counter(e["card"] for e in fit).most_common():
        sub = [e for e in fit if e["card"] == card]
        accs = [acc(sub, lambda e, S=S: S) for S in Ss]
        bS = float(Ss[int(np.argmax(accs))])
        tv = sorted(acquisition(e) for e in sub if e["pulled"])
        print(f"    {card:16s} {n:4d} {sum(e['pulled'] for e in sub):4d} | {acc(sub, engine):.0%} | {bS} ({max(accs):.0%}) "
              f"| {q(tv, .5):.1f} / {tv[-1] if tv else float('nan'):.1f}")
    bins = collections.defaultdict(lambda: [0, 0])
    for e in fit:
        k = int(e["d0"])
        bins[k][0] += e["pulled"]
        bins[k][1] += 1
    print("  pull rate by distance when the defender appeared (tiles): "
          + ", ".join(f"{k}-{k + 1}: {p}/{n}" for k, (p, n) in sorted(bins.items())))
    pr = [e for e in fit + held if e["card"] == "Prince"]
    print(f"  Prince events ({len(pr)}):")
    for e in sorted(pr, key=lambda e: -acquisition(e))[: a.examples]:
        print(f"    {e['match_id']} t={e['t']:.1f}s vs {e['defender']}: prince {e['attacker']} def {e['def']}, d0 {e['d0']:.1f} "
              f"{'turn began at' if e['pulled'] else 'closest'} {acquisition(e):.1f} -> {'PULLED' if e['pulled'] else 'not pulled'} ({e['split']})")
    wrong = [e for e in fit if rule(e, engine(e)) != e["pulled"]]
    print(f"  examples the engine's sight gets wrong ({len(wrong)}):")
    for e in sorted(wrong, key=lambda e: -acquisition(e))[: a.examples]:
        print(f"    {e['match_id']} t={e['t']:.1f}s {e['card']} vs {e['defender']}: {e['attacker']} -> {e['def']}, d0 {e['d0']:.1f} "
              f"{'turn began at' if e['pulled'] else 'closest'} {acquisition(e):.1f} -> {'PULLED' if e['pulled'] else 'not pulled'}")


if __name__ == "__main__":
    main()
