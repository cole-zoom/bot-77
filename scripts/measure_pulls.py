"""Measure, from the pro replays, when a defender's building pulls a building-targeting troop, and fit the engine's
crown-tower targeting radius to it (2026-10-28; Cole: "that hut should pull a giant ... but it doesn't").

  uv run --no-sync python scripts/measure_pulls.py [--workers 4] [--examples 6]

Why: the engine ranks a building-targeter's candidates by distance to their footprint edge (BuildingTargeter::
findTarget: rank = centre distance - getTargetingRadius(), the nearest in sight wins), and Tower::getTargetingRadius
is 3.3 for a princess tower (4.4 King), fitted to one real-play example (a Hog passing a Cannon 7-8 tiles off-lane).
In the engine a Goblin Hut pulls a lane Giant only from within ~4.5 tiles of it.

What it does, per human replay (sources replay_tv / replay_bl, detections run v5l_e15, ~4 fps):
- detections -> engine coordinates (bot77.sim.geometry: box centre + FEET_DY, then ours_to_engine), every event
  mirrored so the attacker walks toward low y (defender's princess towers at (3, 6) / (14, 6));
- building-targeter tracks (nearest-neighbour links per card and team, gaps <= 1 s, >= 4 detections) and defender
  building sites (detections of one building card clustered by position; alive from first to last sighting);
- an event: a troop on the defender's side (y < 15.5) while a defender building is up, taken at the first frame both
  hold and the troop is still >= LOCK_DIST from its lane princess tower's centre (closer, it may already be locked on in
  range: excluded); the outcome from the track afterwards: PULLED if it comes within REACH of the building before
  REACH_TOWER of the tower, NOT PULLED if the reverse; neither (killed, lost, spelled) is dropped;
- the fit: pulled iff d_building - r < d_tower - R and d_building - r <= sight (the engine's rule), r the building's
  footprint (1.0), R swept; accuracy for the engine's 3.3 against the best R, on train + dev, then held-back.

Ground troops and flyers (Balloon, Lava Hound) are reported apart. Never on test / held-out.
"""
import argparse
import collections
import math
from multiprocessing import Pool

import numpy as np

from bot77.features.frames import FEET_DY
from bot77.sim import store
from bot77.sim.geometry import ours_to_engine

RUN = "v5l_e15"
TARGETERS = {"Hog Rider": 9.5, "Giant": 7.5, "Battle Ram": 6.5, "Golem": 7.0, "Royal Giant": 7.5, "Balloon": 7.7,
             "Ram Rider": 6.5, "Goblin Giant": 7.5, "Electro Giant": 7.5, "Lava Hound": 7.5, "Royal Hogs": 9.5,
             "Wall Breakers": 7.0, "Elixir Golem": 7.0}  # engine sight (get_card_stats), the fit's sight gate
FLYERS = {"Balloon", "Lava Hound"}
BUILDINGS = {"Cannon", "Tesla", "Inferno Tower", "Bomb Tower", "Goblin Hut", "Tombstone", "Goblin Cage", "Furnace",
             "Barbarian Hut", "Elixir Collector", "Mortar", "X-Bow"}
TOWERS = ((3.0, 6.0), (14.0, 6.0))
R_BUILDING = 1.0  # Building::COLLISION_RADIUS
LOCK_DIST = 4.0  # closer to the tower's centre than this, the troop may already be attacking it
REACH = 2.6  # building centre: its footprint + a melee reach
REACH_TOWER = 3.3  # tower centre: the same for the tower's 1.5 footprint
MAX_JUMP = 2.0  # engine tiles between linked detections (~0.25 s apart; Hog ~0.33 tiles)


def _engine(x, y, team):
    ex, ey = ours_to_engine(x, y + FEET_DY)
    return (17.0 - ex, 33.0 - ey) if team == "friendly" else (ex, ey)  # attacker = the other side, walking to low y


def _tracks(rows):
    """[(t, x, y)] tracks, greedy nearest-neighbour per frame."""
    by_t = collections.defaultdict(list)
    for r in rows:
        by_t[r["t"]].append((r["x"], r["y"]))
    tracks, open_ = [], []
    for t in sorted(by_t):
        pts = list(by_t[t])
        nxt = []
        for tr in open_:
            if t - tr[-1][0] > 1.0:
                tracks.append(tr)
                continue
            if pts:
                j = min(range(len(pts)), key=lambda i: math.dist(pts[i], tr[-1][1:]))
                if math.dist(pts[j], tr[-1][1:]) <= MAX_JUMP * max(1.0, (t - tr[-1][0]) / 0.25):
                    tr.append((t, *pts.pop(j)))
            nxt.append(tr)
        for p in pts:
            nxt.append([(t, *p)])
        open_ = nxt
    return [tr for tr in tracks + open_ if len(tr) >= 4]


def _sites(rows):
    """Building sites: [(card, x, y, t_first, t_last)], detections within 1.5 tiles merged."""
    sites = []
    for r in sorted(rows, key=lambda r: r["t"]):
        for s in sites:
            if s[0] == r["card"] and math.dist((s[1], s[2]), (r["x"], r["y"])) < 1.5 and r["t"] - s[4] < 3.0:
                s[4] = r["t"]
                break
        else:
            sites.append([r["card"], r["x"], r["y"], r["t"], r["t"]])
    return [s for s in sites if s[4] - s[3] >= 1.0]


def match_events(args):
    mid, split = args
    rows = store.read("detections", f"run_id = '{RUN}' AND match_id = '{mid}'")
    out = []
    for atk in ("enemy", "friendly"):
        dfd = "friendly" if atk == "enemy" else "enemy"
        conv = lambda r, team: {**r, **dict(zip(("x", "y"), _engine(r["x"], r["y"], team)))}
        sites = _sites([conv(r, atk) for r in rows if r["team"] == dfd and r["card"] in BUILDINGS])
        if not sites:
            continue
        for card in TARGETERS:
            trs = _tracks([conv(r, atk) for r in rows if r["team"] == atk and r["card"] == card])
            for tr in trs:
                tower = TOWERS[0] if np.median([p[1] for p in tr]) < 8.5 else TOWERS[1]
                for i, (t, x, y) in enumerate(tr):
                    if y >= 15.5:
                        continue
                    d_t = math.dist((x, y), tower)
                    if d_t < LOCK_DIST:
                        break
                    up = [s for s in sites if s[3] <= t <= s[4] and s[2] < 15.5]
                    if not up:
                        continue
                    s = min(up, key=lambda s: math.dist((x, y), (s[1], s[2])))
                    d_b = math.dist((x, y), (s[1], s[2]))
                    outcome = None
                    for t2, x2, y2 in tr[i + 1:]:
                        if t2 > t + 15:
                            break
                        if math.dist((x2, y2), (s[1], s[2])) <= REACH and t2 <= s[4] + 0.5:
                            outcome = True
                            break
                        if math.dist((x2, y2), tower) <= REACH_TOWER:
                            outcome = False
                            break
                    if outcome is not None:
                        out.append({"match_id": mid, "split": split, "t": t, "card": card, "building": s[0],
                                    "d_building": d_b, "d_tower": d_t, "pulled": outcome,
                                    "troop": (round(x, 1), round(y, 1)), "site": (round(s[1], 1), round(s[2], 1))})
                    break  # one decision per track: the first moment it applies
    return out


def accuracy(ev, R, r=R_BUILDING):
    ok = [((e["d_building"] - r < e["d_tower"] - R) and (e["d_building"] - r <= TARGETERS[e["card"]])) == e["pulled"]
          for e in ev]
    return sum(ok) / max(1, len(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    ms = store.read("matches", "source IN ('replay_tv', 'replay_bl') AND pov_side = 'bottom'")
    jobs = [(m["match_id"], m["split"]) for m in ms if m["split"] in ("train", "dev", "heldback")]
    with Pool(a.workers) as p:
        ev = [e for chunk in p.imap_unordered(match_events, jobs) for e in chunk]
    fit = [e for e in ev if e["split"] != "heldback" and e["card"] not in FLYERS]
    held = [e for e in ev if e["split"] == "heldback" and e["card"] not in FLYERS]
    fly = [e for e in ev if e["card"] in FLYERS]
    print(f"{len(jobs)} games; events: {len(fit)} ground (train + dev), {len(held)} ground held-back, {len(fly)} flyers")
    print(f"  pulled: {sum(e['pulled'] for e in fit)} of {len(fit)} (fit set)")
    Rs = np.round(np.arange(0.0, 6.01, 0.05), 2)
    accs = [accuracy(fit, R) for R in Rs]
    best = Rs[int(np.argmax(accs))]
    band = [float(R) for R, acc in zip(Rs, accs) if acc >= max(accs) - 1e-9]
    print(f"  tower targeting radius 3.3 (engine): {accuracy(fit, 3.3):.1%} | best R {best} ({max(accs):.1%}; "
          f"equally good {band[0]}-{band[-1]}) | collision radius 1.5: {accuracy(fit, 1.5):.1%}")
    print(f"  held-back ({len(held)}): 3.3 {accuracy(held, 3.3):.1%}, R {best} {accuracy(held, best):.1%}")
    for R in (1.5, 2.0, 2.5, 3.0, 3.3):
        print(f"    R {R}: {accuracy(fit, R):.1%}")
    print("  per troop (fit set): card n pulled | acc 3.3 | acc best R")
    for card, n in collections.Counter(e["card"] for e in fit).most_common():
        sub = [e for e in fit if e["card"] == card]
        print(f"    {card:14s} {n:4d} {sum(e['pulled'] for e in sub):4d} | {accuracy(sub, 3.3):.0%} | {accuracy(sub, best):.0%}")
    # the sight gate: pulled events whose building was beyond the engine's sight (the gate would forbid them)
    beyond = [e for e in fit if e["pulled"] and e["d_building"] - R_BUILDING > TARGETERS[e["card"]]]
    print(f"  pulled from beyond the engine's sight (d_building - 1 > sight): {len(beyond)}"
          + (f" e.g. {[(e['card'], round(e['d_building'], 1)) for e in beyond[:5]]}" if beyond else ""))
    if fly:
        print(f"  flyers: {len(fly)}, pulled {sum(e['pulled'] for e in fly)}; 3.3 {accuracy(fly, 3.3):.0%}, R {best} {accuracy(fly, best):.0%}")
    wrong = [e for e in fit if ((e["d_building"] - R_BUILDING < e["d_tower"] - 3.3)) != e["pulled"]]
    print(f"  examples the engine's 3.3 gets wrong ({len(wrong)}):")
    for e in sorted(wrong, key=lambda e: -abs(e["d_building"] - e["d_tower"]))[: a.examples]:
        print(f"    {e['match_id']} t={e['t']:.1f}s {e['card']} vs {e['building']}: troop {e['troop']} building {e['site']}, "
              f"d_building {e['d_building']:.1f} d_tower {e['d_tower']:.1f} -> {'PULLED' if e['pulled'] else 'not pulled'}")
    bins = collections.defaultdict(lambda: [0, 0])
    for e in fit:
        k = round(e["d_tower"] - e["d_building"])
        bins[k][0] += e["pulled"]
        bins[k][1] += 1
    print("  pull rate by d_tower - d_building (tiles): " + ", ".join(f"{k:+d}: {p}/{n}" for k, (p, n) in sorted(bins.items())))


if __name__ == "__main__":
    main()
