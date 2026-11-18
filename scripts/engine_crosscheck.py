"""Our engine against Hasty-CR's published predictions (2026-11-18; github.com/hastylmao/Hasty-CR, MIT: its README,
sim/check.py and "Known limitations"). Neither engine is the truth: where they disagree, the replays (sim_validate)
or Cole decide. His numbers are from his README; his engine wasn't run here.

  uv run --no-sync python scripts/engine_crosscheck.py

Setups (our frame: own = bottom; troop towers zeroed where a duel must be alone):
 1. A lone Hog Rider at the bridge on an undefended princess tower: hits and damage before it dies (his: 6, 1902).
 2. A Musketeer walking into a Cannon from 10 tiles, nothing else: who wins, when, Musketeer HP lost (his: Musketeer,
    7.1 s, untouched).
 3. A Prince charging a lone Musketeer from 6 tiles: who wins, when.
 4. Body-block: a Hog's arrival at a princess tower with one Skeleton in its path vs none (his: ~0.5 s, uncalibrated;
    it used to stop the Hog for good).
 5. A Battle Ram meeting a Knight: does it get through (his: held indefinitely, a known bug).
 6. After our left princess tower falls, an enemy Hog in that lane: its first building target (his bug: the far
    lane's tower instead of the King).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bot77.features.frames import RIVER_Y  # noqa: E402
from bot77.sim import cards as sc  # noqa: E402
from bot77.sim.engine import TOWER_PRESETS, Sim  # noqa: E402
from bot77.sim.geometry import engine_to_ours  # noqa: E402

DECK = [sc.engine_id(n) for n in ["Hog Rider", "Musketeer", "Cannon", "Prince", "Skeletons", "Battle Ram", "Knight",
                                  "Ice Golem"]]
HOG, MUSK, CANNON, PRINCE, SKEL, RAM, KNIGHT = DECK[:7]


def new(harmless_towers: bool = False) -> Sim:
    s = Sim.new(DECK, DECK)
    s.reset()
    if harmless_towers:  # keep the towers (the arena needs them), take away their shots
        (khp, _, kcd), (php, _, pcd) = TOWER_PRESETS[s.towers_preset]
        for tm in (0, 1):
            s.env.set_tower_stats(tm, 0, khp, 0, kcd)
            for slot in (1, 2):
                s.env.set_tower_stats(tm, slot, php, 0, pcd)
    return s


def alive(s: Sim, card: int, team: str) -> list[dict]:
    return [u for u in s.units() if u["card_id"] == card and u["team"] == team]


def hog_on_undefended_tower():
    s = new()
    before = s.towers()["opp_l"]["hp"]
    s.inject(HOG, 3.45, RIVER_Y + 1.0, "own")
    s.step(1)  # commit the injection
    first_hit, hits, last = None, 0, before
    while alive(s, HOG, "own") and s.t < 60:
        s.step(1)
        hp = s.towers()["opp_l"]["hp"]
        if hp < last:
            hits += 1
            first_hit = first_hit or (last - hp)
        last = hp
    return {"hits": hits, "damage": round(before - last), "damage_per_hit": round(first_hit or 0)}


def duel(attacker: int, defender: int, gap: float):
    """attacker (opp) already on our side, `gap` engine tiles from the defender (own), both on the centre line, towers
    harmless (shots zeroed) and our princess towers gone (buildings in sight would pull the attacker). Engine tiles,
    not our frame (our y is screen-scaled). Seconds from the start, and from the first damage either way."""
    s = new(harmless_towers=True)
    for slot in ("l", "r"):
        s.set_tower("own", slot, 0)
    ey_def = 7.5  # engine y: our King ~2.5, our princess towers 6, the river 16.5
    s.inject(defender, *engine_to_ours(8.5, ey_def), "own")
    s.inject(attacker, *engine_to_ours(8.5, ey_def + gap), "opp")
    s.step(1)  # commit the injections
    t0 = s.t
    a0, d0 = alive(s, attacker, "opp")[0]["hp"], alive(s, defender, "own")[0]["hp"]
    first, decay = None, 0.0
    while alive(s, attacker, "opp") and alive(s, defender, "own") and s.t < 90:
        s.step(1)
        a, d = alive(s, attacker, "opp"), alive(s, defender, "own")
        if first is None and ((a and a[0]["hp"] < a0) or not a or not d or (d and d0 - d[0]["hp"] > 30)):
            first = s.t
    a, d = alive(s, attacker, "opp"), alive(s, defender, "own")
    win = "attacker" if a and not d else "defender" if d and not a else "neither"
    return {"winner": sc.our_name(attacker if win == "attacker" else defender) if win != "neither" else win,
            "seconds": round(s.t - t0, 1), "from_first_damage_s": round(s.t - first, 1) if first else None,
            "attacker_hp_lost": round(a0 - a[0]["hp"]) if a else None}


def hog_arrival(skeleton: bool) -> float | None:
    """Enemy Hog from their bridge (left lane) to our left princess tower's first hit; one Skeleton body in its path."""
    s = new()
    s.inject(HOG, 3.45, RIVER_Y - 1.5, "opp")
    if skeleton:
        s.inject(SKEL, 3.45, RIVER_Y + 2.5, "own", bodies=1)
    hp0 = s.towers()["own_l"]["hp"]
    while s.t < 20:
        s.step(1)
        if s.towers()["own_l"]["hp"] < hp0:
            return round(s.t, 1)
    return None


def ram_vs_knight():
    s = new(harmless_towers=True)
    s.inject(RAM, 3.45, RIVER_Y - 1.5, "opp")
    s.inject(KNIGHT, 3.45, RIVER_Y + 3.0, "own")
    hp0 = s.towers()["own_l"]["hp"]
    while s.t < 30:
        s.step(1)
        if s.towers()["own_l"]["hp"] < hp0:
            return {"reached_tower_s": round(s.t, 1)}
        if not alive(s, RAM, "opp"):  # rams turn into Barbarians when they hit or die
            return {"ram_gone_s": round(s.t, 1), "tower_hit": s.towers()["own_l"]["hp"] < hp0}
    return {"stuck_after_s": 30}


def retarget_after_princess_falls():
    s = new()
    s.set_tower("own", "l", 0)
    s.inject(HOG, 3.45, RIVER_Y - 1.5, "opp")
    towers = {u["id"]: u for u in s.units(towers=True) if u["tower"]}
    first = None
    for _ in range(150):
        s.step(1)
        h = alive(s, HOG, "opp")
        if not h:
            break
        tid = h[0]["target"]
        if tid in towers and first is None:
            t = towers[tid]
            first = {"target": "king" if abs(t["x"] - 9) < 1 else ("left princess" if t["x"] < 9 else "right princess"),
                     "team": t["team"], "at_s": round(s.t, 1)}
    hits = {k: round(v["max"] - v["hp"]) for k, v in s.towers().items() if v["hp"] < v["max"] and k.startswith("own")}
    return {"first_building_target": first, "own_damage_taken": hits}


def main():
    rows = [
        ("Lone Hog on an undefended princess tower", "6 hits, 1902 dmg", hog_on_undefended_tower()),
        ("Musketeer walks into a Cannon (his: 10 tiles; ours: 9, on the line to our King)", "Musketeer wins, 7.1 s, untouched",
         duel(MUSK, CANNON, 9)),
        ("Prince charges a lone Musketeer from 6 tiles", "(not published)", duel(PRINCE, MUSK, 6)),
        ("Hog to tower, no Skeleton / one Skeleton in path", "~0.5 s delay",
         {"no_skeleton_s": hog_arrival(False), "one_skeleton_s": hog_arrival(True)}),
        ("Battle Ram meets a Knight", "held indefinitely (his bug)", ram_vs_knight()),
        ("Our left princess down, enemy Hog in that lane", "King (his bug: far princess)",
         retarget_after_princess_falls()),
    ]
    for q, his, ours in rows:
        print(f"- {q}\n    Hasty-CR: {his}\n    ours:     {ours}")


if __name__ == "__main__":
    main()
