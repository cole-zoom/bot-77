"""S0 speed check for the simulator engine (docs/simulator.md): seeded random-play matches, like the fork's own
measurement (README "Engine speed": 20 seeded random-play matches), and the cost of one `snapshot()` fork mid-match.

  uv run python scripts/sim_bench.py [--matches 20]

Both sides play the session 2 deck. Each tick a side plays with a small chance: a random hand card at a random legal
spot (the engine's own `is_valid_placement`). Times are wall clock on one core, the engine only (no observations).
"""
import argparse
import random
import statistics
import time

import clash_royale_env as cre

DECK = ["Giant", "Mini PEKKA", "Musketeer", "Goblin Hut", "Fireball", "Arrows", "Skeletons", "Minions"]
PLAY_CHANCE = 0.04  # per side per tick: about one play every 2.5 s, roughly a real game's pace


def card_ids(names):
    by_name = {cre.get_card_info(i)["name"]: i for i in cre.get_all_card_ids()}
    return [by_name[n] for n in names]


def random_action(env, team, rng):
    """(hand index, x, y in the team's own frame) or (-1, 0, 0) for no play."""
    if rng.random() > PLAY_CHANCE:
        return -1, 0.0, 0.0
    hand = env.get_hand_for_team(team)
    i = rng.randrange(len(hand))
    if env.get_elixir_for_team(team) < cre.get_card_info(hand[i])["cost"]:
        return -1, 0.0, 0.0
    for _ in range(20):
        x, y = rng.uniform(0, cre.ARENA_WIDTH - 1), rng.uniform(0, cre.ARENA_HEIGHT - 1)
        real_y = y if team == 0 else cre.ARENA_HEIGHT - 1 - y  # step_self_play mirrors team 1's y
        if env.is_valid_placement(hand[i], x, real_y, team):
            return i, x, y
    return -1, 0.0, 0.0


def play_match(deck, seed, snapshot_at=None):
    rng = random.Random(seed)
    env = cre.ClashRoyaleEnv(deck, deck)
    env.seed(seed)
    env.reset()
    ticks, snap_us = 0, None
    t0 = time.perf_counter()
    while True:
        a0, a1 = random_action(env, 0, rng), random_action(env, 1, rng)
        r = env.step_self_play_fast(*a0, *a1, skip_frames=1)
        ticks += 1
        if snapshot_at is not None and ticks == snapshot_at:
            s0 = time.perf_counter()
            env.snapshot()
            snap_us = (time.perf_counter() - s0) * 1e6
        if r.done:
            break
    return time.perf_counter() - t0, ticks, r.reward0, snap_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--snapshots", type=int, default=200)
    args = ap.parse_args()
    deck = card_ids(DECK)

    secs, ticks, results = [], [], []
    for seed in range(args.matches):
        s, n, r, _ = play_match(deck, seed)
        secs.append(s)
        ticks.append(n)
        results.append(r)
    per_tick = [s / n * 1e6 for s, n in zip(secs, ticks)]
    print(f"matches: {args.matches}, ticks per match median {statistics.median(ticks):.0f} "
          f"(min {min(ticks)}, max {max(ticks)}), results {results.count(1)} W / {results.count(-1)} L / "
          f"{results.count(0)} D")
    print(f"match wall time: median {statistics.median(secs) * 1e3:.1f} ms, max {max(secs) * 1e3:.1f} ms "
          f"(includes the Python driving loop)")
    print(f"per tick: median {statistics.median(per_tick):.1f} us")

    # snapshot() at mid-match boards (tick 900 = 1:30 in), one per seed
    snaps = [play_match(deck, 1000 + i, snapshot_at=900)[3] for i in range(args.snapshots // 10)]
    env = cre.ClashRoyaleEnv(deck, deck)
    env.seed(0)
    env.reset()
    rng = random.Random(0)
    for _ in range(900):
        env.step_self_play_fast(*random_action(env, 0, rng), *random_action(env, 1, rng), skip_frames=1)
    for _ in range(args.snapshots):
        s0 = time.perf_counter()
        env.snapshot()
        snaps.append((time.perf_counter() - s0) * 1e6)
    snaps = [s for s in snaps if s is not None]
    print(f"snapshot() at tick 900: median {statistics.median(snaps):.1f} us, max {max(snaps):.1f} us "
          f"({len(snaps)} forks)")

    # the engine alone, no Python per tick: one 10 s rollout from a fork, as search will use it
    fork = env.snapshot()
    s0 = time.perf_counter()
    fork.step_self_play_fast(-1, 0, 0, -1, 0, 0, skip_frames=100)
    print(f"10 s rollout from a fork (100 ticks in one call): {(time.perf_counter() - s0) * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
