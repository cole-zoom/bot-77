"""Training clips for the spell stage (decisions.md, 2026-10-14): short arena clips around every spell play in the
non-test replays (both players: the top player's from replay truth, the bottom player's from our own hand log), every
champion / hero ability use ("<card> ability", from both hands' logs, 2026-10-15), plus
"no spell" clips at troop plays and at random quiet moments. The test, benchmark and v5 check games are never used.

Each clip is CLIP_FRAMES frames at FPS of the arena crop, starting PRE_S before the card left the hand, saved as one
JPEG strip (frames stacked vertically). The trainer random-crops 16 frames from it.

  uv run python scripts/spell_clips.py [--procs 8]   -> data/upload/spell_clips/{clips/*.jpg, index.json}
"""
import argparse, hashlib, json, glob, random, subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

FPS, W, H = 10, 144, 220
PRE_S, CLIP_FRAMES = 0.6, 28  # [t - 0.6, t + 2.2]
QUIET_S = 3.0  # random negatives stay this far from any play of either player
NEG_RANDOM, NEG_TROOP = 14, 8  # per game
VAL_SHARE = 0.1
OUT = Path("data/upload/spell_clips")
SKIP_SPELLS = {"Mirror", "Clone", "Spirit Empress"}  # copy another card / a troop in the game data


def arena_stream(path: str, arena, t0: float, t1: float) -> np.ndarray:
    x0, y0, x1, y1 = arena
    vf = f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},fps={FPS},scale={W}:{H}:flags=area"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", path, "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def game(job):
    m, path, arena, t_start, t_end, items, split = job
    try:
        frames = arena_stream(path, arena, t_start, t_end)
    except Exception as e:  # noqa: BLE001
        return [], f"{m}: {e}"
    rng = random.Random(m)
    plays = sorted(items, key=lambda i: i["t"])
    clips = [dict(i, kind="spell") for i in plays if i["spell"]]
    troops = [i for i in plays if not i["spell"]]
    near_spell = lambda t: any(abs(t - s["t"]) < 2.5 for s in plays if s["spell"])
    clips += [dict(i, label="__none", kind="troop") for i in rng.sample(troops, min(NEG_TROOP, len(troops)))
              if not near_spell(i["t"])]
    tries = 0
    while sum(c["kind"] == "quiet" for c in clips) < NEG_RANDOM and tries < 500:
        tries += 1
        t = rng.uniform(t_start + 2, t_end - 4)
        if all(abs(t - p["t"]) >= QUIET_S for p in plays):
            clips.append({"t": t, "label": "__none", "kind": "quiet", "side": None})
    out = []
    for k, c in enumerate(clips):
        i0 = int(round((c["t"] - PRE_S - t_start) * FPS))
        if i0 < 0 or i0 + CLIP_FRAMES > len(frames):
            continue
        name = f"{m}_{k:03d}"
        cv2.imwrite(str(OUT / "clips" / f"{name}.jpg"), np.concatenate(frames[i0:i0 + CLIP_FRAMES]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append({"name": name, "match_id": m, "t": round(c["t"], 2), "label": c["label"], "kind": c["kind"],
                    "side": c.get("side"), "split": split})
    return out, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); a = ap.parse_args()
    import lancedb
    from bot77.layout import Layout
    db = lancedb.connect("data/lancedb")
    ctype = {c["name"]: c["type"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000)
             .to_arrow().to_pylist()}
    is_spell = lambda c: ctype.get(c) == "spell" and c not in SKIP_SPELLS
    hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
    hold |= set(json.loads(Path("config/v5_check_games.json").read_text())["matches"])
    hold |= set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
    jobs = []
    for f in sorted(glob.glob("data/ground_truth/opponent_truth_replay_*.json")):
        tr = json.loads(Path(f).read_text()); m = tr["matches"][0]
        if m in hold:
            continue
        v = db.open_table("videos").search().where(f"video_id = '{tr['video_id']}'").to_arrow().to_pylist()[0]
        mt = db.open_table("matches").search().where(f"match_id = '{m}'").to_arrow().to_pylist()[0]
        arena = Layout.load(v["layout_id"]).katacr_arena
        items = [{"t": i["t_video"], "label": i["label"], "spell": is_spell(i["label"]), "side": "top"}
                 for i in tr["items"] if i["label"] != "__ability"]
        own = db.open_table("events").search().where(f"match_id = '{m}' AND kind = 'play'").limit(10**4) \
            .select(["t_video", "card"]).to_arrow().to_pylist()
        items += [{"t": e["t_video"], "label": e["card"] or "__unknown", "spell": is_spell(e["card"]), "side": "bottom"}
                  for e in own]
        # champion / hero ability uses from both hands (2026-10-15), as "<card> ability" positives
        for side, mid in (("top", tr.get("paired_with")), ("bottom", m)):
            ab = db.open_table("events").search().where(f"match_id = '{mid}' AND kind = 'ability'").limit(10**4) \
                .select(["t_video", "card"]).to_arrow().to_pylist() if mid else []
            items += [{"t": e["t_video"], "label": f"{e['card']} ability" if e["card"] else "__unknown", "spell": True,
                       "side": side} for e in ab]
        # unknown/other cards still count as plays (kept away from negatives) but never as positives
        items = [dict(i, spell=i["spell"] and i["label"] not in ("__unknown",)) for i in items]
        split = "val" if int(hashlib.sha1(m.encode()).hexdigest(), 16) % 1000 < VAL_SHARE * 1000 else "train"
        jobs.append((m, v["path"], arena, mt["t_start"], mt["t_end"], items, split))
    # POV recordings (2026-10-15): the classifier only saw replays and over-called Void on Cole's recordings. Use the
    # recordings outside dev / benchmark: our own spells (hand log) as positives, troop plays and quiet moments as
    # negatives. The opponent's plays aren't known there, so a quiet clip may rarely hold an opponent spell.
    dev = set()
    for f in glob.glob("data/ground_truth/opponent_truth_auto_*.json"):
        dev |= set(json.loads(Path(f).read_text())["matches"])
    pov_v = {v["video_id"]: v for v in db.open_table("videos").search().limit(5000).to_arrow().to_pylist()
             if v["video_id"].startswith(("cole_", "friend_"))}
    for mt in db.open_table("matches").search().limit(10**5).to_arrow().to_pylist():
        m = mt["match_id"]
        if mt["video_id"] not in pov_v or m in hold | dev:
            continue
        v = pov_v[mt["video_id"]]
        arena = Layout.load(v["layout_id"]).katacr_arena
        ev = db.open_table("events").search().where(f"match_id = '{m}' AND kind IN ('play', 'ability')").limit(10**4) \
            .select(["t_video", "card", "kind"]).to_arrow().to_pylist()
        items = [{"t": e["t_video"], "spell": e["kind"] == "ability" or is_spell(e["card"]), "side": "bottom",
                  "label": (f"{e['card']} ability" if e["kind"] == "ability" else e["card"]) if e["card"] else "__unknown"}
                 for e in ev]
        items = [dict(i, spell=i["spell"] and i["label"] != "__unknown") for i in items]
        split = "val" if int(hashlib.sha1(m.encode()).hexdigest(), 16) % 1000 < VAL_SHARE * 1000 else "train"
        jobs.append((m, v["path"], arena, mt["t_start"], mt["t_end"], items, split))
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    index, errs = [], []
    with Pool(a.procs) as pool:
        for k, (rows, err) in enumerate(pool.imap_unordered(game, jobs)):
            index += rows
            if err:
                errs.append(err)
            if k % 20 == 0:
                print(f"{k + 1}/{len(jobs)} games, {len(index)} clips", flush=True)
    (OUT / "index.json").write_text(json.dumps({"fps": FPS, "w": W, "h": H, "pre_s": PRE_S, "frames": CLIP_FRAMES,
                                                "clips": index}, indent=0))
    from collections import Counter
    print("errors", errs[:5], len(errs))
    print(Counter((c["split"], c["label"]) for c in index if c["label"] != "__none").most_common())
    print(Counter((c["split"], c["kind"]) for c in index))


if __name__ == "__main__":
    main()
