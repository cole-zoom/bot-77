"""Step 5 of the input rebuild (2026-11-16): the new input block for the simulator demos (school_v3), by re-simulating
each episode and feeding the engine's units to OppInputs as if the live detector had read them.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/rebuild_sim_block.py --check 20 [--keep 0.5]
  uv run --no-sync --with gymnasium==1.3.0 python scripts/rebuild_sim_block.py [--workers 8] [--keep P]

Per episode (data/cache/bc/school_v3/<source>_<template>_<seed>.npz): build_defence_demos.episode's loop again (same
template, seed, physics and, for Cole's, his plays), and every 0.1 s step the engine's units go to OppInputs as reads
(card = the unit's card, team, the feet -> box centre in our frame, conf 0.9), each kept with probability --keep (the
live detector misses units: PlayTracker's recall on the replays is 65%); our own plays (the script's / Cole's) go to
own_play. Spells aren't engine units, so their spell slots stay empty (live: Fireball in a slot 67%).
--check N: N episodes, reports whether the re-simulated observations equal the saved ones (the episode replays
exactly) and the tracker's recall of the opponent's real plays (new opponent units by unit id) at --keep.
Out: data/cache/bc/school_v4/ same rows and labels, obs 13,977 + 437. Where an episode doesn't replay exactly, its
old rows with a zero block (block-present 0).

Result 2026-11-16 (--check 12): 1 of 12 episodes replays exactly on today's engine; the rest part from the saved rows
3.5-21 s in (median ~4.5 s), first in one board-HP cell (the engine's stats moved since the demos were built on
10-30), then the scripted defence's choices. A block from a different trajectory would describe a board the rows don't
show, so school_v3 (and games_v9's 19 sim games, built the same way) keep a zero block: train_bc --extra-obs pads old
rows with zeros (extra_obs.fit_obs), which is block-present 0; no files are written.
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
sys.path.insert(0, str(ROOT / "scripts"))
import build_defence_demos as bdd  # noqa: E402

SRC, OUT = ROOT / "data/cache/bc/school_v3", ROOT / "data/cache/bc/school_v4"
DECK_NAMES = "Giant,Battle Ram,Mini P.E.K.K.A,Mega Minion,Bomber,Goblin Hut,Fireball,Arrows"


def _cole_games() -> list[dict]:
    return [json.loads(line) for line in open(ROOT / "data/live/situation_play.jsonl")]


def resim(path: str, keep: float, seed_noise: int = 0) -> dict:
    if not bdd._ENV:
        bdd.setup()  # (chdir to the fork, as the demos were built)
    import clash_royale_env as cre
    from bot77.features.frames import FEET_DY
    from bot77.live.extra_obs import EXTRA_DIM, OppInputs
    from bot77.live.rebuild import cards as load_cards
    from bot77.sim import cards as sc
    from bot77.sim.geometry import engine_to_ours
    E = bdd._ENV
    f = Path(path)
    z = dict(np.load(f, allow_pickle=True))
    source, rest = f.stem.split("_", 1)
    tpl_name, seed = rest.rsplit("_", 1)
    seed = int(seed)
    today = {"client_collision": True, "obstacle_slide": True, "bridge_own_line": True, "tower_reach_footprint": True,
             "pocket_deploy": True}
    plays, switches = None, today
    if source.startswith("cole"):
        g = _cole_games()[int(source[4:])]
        plays, switches = g["plays"], bdd.physics(g)
    for k, v in switches.items():
        getattr(cre, "set_" + k)(v)
    if "cards" not in E:
        E["cards"] = load_cards()
    rng = np.random.default_rng(seed * 7919 + seed_noise)
    torch, sit = E["torch"], E["sit"]
    env = E["CloneEnv"]({"scenario_seed": 0, "decision_ticks": 1})
    env.forced_template = {t.__name__: t for t in sit.TEMPLATES}[tpl_name]
    env.rng = np.random.default_rng(seed)
    env.scenario_rng = env.rng
    torch.manual_seed(seed)
    env.reset()
    at = {int(round(p["t"] * 10)): p for p in plays or [] if p.get("landed", True)}
    oi = OppInputs(E["cards"], DECK_NAMES.split(","))
    names = {e: c["name"] for e, c in sc.engine_cards().items()}
    old = z["obs"]
    from bot77.live.rebuild import section_of
    same, blocks, i, first_diff = True, [], 0, None
    seen_ids, true_plays = set(), []
    while True:
        i += 1
        k, x, y = 4, 0.0, 0.0
        if plays is None:
            k, x, y = sit.scripted_defence(env)
        elif i in at:
            p = at[i]
            cid = E["ids"].get(p["card"])
            hand = list(env.game.get_hand())
            if cid is not None and cid not in hand:
                env.game.set_hand_for_team(0, [cid] + [c for c in hand if c != cid][:3])
                hand = list(env.game.get_hand())
            if cid in hand:
                k, x, y = hand.index(cid), float(p["x"]), float(p["y"])
        o = np.asarray(env.game.get_observation_for_team(0), np.float16)
        r = i - 1
        if r >= len(old) or not np.array_equal(o, old[r]):
            same = False
            first_diff = (r, int((o != old[r]).sum()) if r < len(old) else -1,
                          sorted({section_of(int(c)) for c in np.nonzero(o != old[r])[0]}) if r < len(old) else [])
            break
        t = r / 10.0
        units = []
        for u in env.game.get_units():
            if u.tower or u.hp <= 0:
                continue
            if u.team == 1 and u.id not in seen_ids:
                seen_ids.add(u.id)
                true_plays.append((t, names.get(u.card_id)))
            if rng.random() > keep:
                continue  # the detector missed it this read
            ox, oy = engine_to_ours(u.x, u.y)
            units.append((names.get(u.card_id), "friendly" if u.team == 0 else "enemy", ox, oy - FEET_DY, 0.9))
        oi.update_units(t, units, t)
        blocks.append(oi.block(t))
        if k < 4:
            hand = list(env.game.get_hand())
            if 0 <= k < len(hand) and hand[k] in names:
                ox, oy = engine_to_ours(x, y)
                oi.own_play(names[hand[k]], t, ox, oy)
        _, _, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                            "target_y": np.array([y], np.float32),
                                            "activate_ability_slot1": 0, "activate_ability_slot2": 0})
        if info.get("rl_situation_end") or term or trunc:
            break
    same = same and len(blocks) == len(old)
    blk = np.stack(blocks).astype(np.float16) if same else np.zeros((len(old), EXTRA_DIM), np.float16)
    # recall: a true opponent play (a new unit id of a card, grouped per card within 1 s) matched by a counted play
    tp = []
    for t, c in true_plays:
        if c and not any(c == c2 and abs(t - t2) < 1.0 for t2, c2 in tp):
            tp.append((t, c))
    tp = [(t, c) for t, c in tp if t > 0.05]  # units on the board at the start aren't plays
    hit = sum(any(q["card"] == c and -0.5 <= q["t"] - t <= 3.0 for q in oi.plays) for t, c in tp)
    return {"file": f.name, "rows": len(old), "replays_exactly": same, "true_opp_plays": len(tp), "counted": len(oi.plays),
            "recalled": hit, "first_diff": first_diff, "block": blk, "z": z}


def job(args) -> dict:
    path, keep, out = args
    try:
        r = resim(path, keep)
    except Exception as e:  # noqa: BLE001
        z = dict(np.load(path, allow_pickle=True))
        from bot77.live.extra_obs import EXTRA_DIM
        r = {"file": Path(path).name, "rows": len(z["obs"]), "replays_exactly": False, "error": repr(e),
             "block": np.zeros((len(z["obs"]), EXTRA_DIM), np.float16), "z": z}
    if out:
        z = r.pop("z")
        o = {k: v for k, v in z.items() if k != "obs"}
        o["obs"] = np.concatenate([z["obs"], r["block"]], 1)
        o["rebuilt"] = np.array(f"re-simulated, sim units -> OppInputs (keep {keep}); exact replay {r['replays_exactly']}")
        np.savez_compressed(Path(out) / r["file"], **o)
    else:
        r.pop("z")
    b = r.pop("block")
    r["block_nonzero_per_row"] = round(float((b != 0).sum(1).mean()), 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = sorted(str(f) for f in SRC.glob("*.npz"))
    if a.check:
        rng = np.random.default_rng(0)
        files = list(rng.choice(files, a.check, replace=False))
    else:
        OUT.mkdir(parents=True, exist_ok=True)
    t0, res = time.time(), []
    with Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(job, [(f, a.keep, None if a.check else str(OUT)) for f in files]), 1):
            res.append(r)
            if a.check or n % 200 == 0:
                print(n, json.dumps(r), f"{time.time() - t0:.0f} s", flush=True)
    ok = [r for r in res if r["replays_exactly"]]
    tp = sum(r.get("true_opp_plays", 0) for r in ok)
    rec = sum(r.get("recalled", 0) for r in ok)
    cnt = sum(r.get("counted", 0) for r in ok)
    summ = {"episodes": len(res), "replay_exactly": len(ok), "keep": a.keep, "true_opp_plays": tp,
            "recall": round(rec / max(1, tp), 3), "counted": cnt, "s": round(time.time() - t0)}
    print(json.dumps(summ))
    if not a.check:
        (OUT / "rebuild_summary.json").write_text(json.dumps({"summary": summ, "episodes": res}, indent=1))


if __name__ == "__main__":
    main()
