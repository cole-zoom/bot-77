"""Whole-game arena streams for spell-stage inference on the pod (decisions.md, 2026-10-14): the same crop, size and
frame rate as scripts/spell_clips.py, saved as JPEG strips of CHUNK frames. Sets: the replay test games, the replay
val split of spell_clips (threshold tuning), the dev POV games and the benchmark.

  uv run python scripts/spell_streams.py [--procs 8]   -> data/upload/spell_streams/<match>/{000.jpg..., meta.json}
"""
import argparse, glob, hashlib, json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spell_clips import FPS, VAL_SHARE, arena_stream  # noqa: E402

CHUNK = 100
OUT = Path("data/upload/spell_streams")


def one(job):
    m, path, arena, t0, t1, setname = job
    d = OUT / m
    if (d / "meta.json").exists():
        return m, None
    try:
        fr = arena_stream(path, arena, t0, t1)
    except Exception as e:  # noqa: BLE001
        return m, str(e)
    d.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(fr), CHUNK):
        cv2.imwrite(str(d / f"{k // CHUNK:03d}.jpg"), np.concatenate(fr[k:k + CHUNK]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (d / "meta.json").write_text(json.dumps({"match_id": m, "t_start": t0, "fps": FPS, "n": len(fr), "set": setname}))
    return m, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); a = ap.parse_args()
    import lancedb
    from bot77.layout import Layout
    db = lancedb.connect("data/lancedb")
    test = set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
    bench = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
    v5c = set(json.loads(Path("config/v5_check_games.json").read_text())["matches"])
    want = {}
    for f in glob.glob("data/ground_truth/opponent_truth_replay_*.json"):
        m = json.loads(Path(f).read_text())["matches"][0]
        if m in test:
            want[m] = "test"
        elif m not in bench | v5c and int(hashlib.sha1(m.encode()).hexdigest(), 16) % 1000 < VAL_SHARE * 1000:
            want[m] = "val"
    for f in glob.glob("data/ground_truth/opponent_truth_auto_*.json"):
        m = json.loads(Path(f).read_text())["matches"][0]
        want[m] = "bench" if m in bench else "dev"
    for m in bench:
        want.setdefault(m, "bench")
    for f in glob.glob("data/ground_truth/opponent_truth_live_*.json"):  # Cole's live games keyed by their replays
        want[json.loads(Path(f).read_text())["matches"][0]] = "live"
    jobs = []
    for m, s in sorted(want.items()):
        mt = db.open_table("matches").search().where(f"match_id = '{m}'").to_arrow().to_pylist()
        if not mt:
            print("no match row", m); continue
        mt = mt[0]
        v = db.open_table("videos").search().where(f"video_id = '{mt['video_id']}'").to_arrow().to_pylist()[0]
        arena = Layout.load(v["layout_id"]).katacr_arena
        jobs.append((m, v["path"], arena, mt["t_start"], mt["t_end"], s))
    print({s: sum(j[5] == s for j in jobs) for s in ("test", "val", "dev", "bench", "live")}, flush=True)
    with Pool(a.procs) as pool:
        for k, (m, err) in enumerate(pool.imap_unordered(one, jobs)):
            if err:
                print("ERR", m, err[:200], flush=True)
            if k % 10 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    print("STREAMS_DONE")


if __name__ == "__main__":
    main()
