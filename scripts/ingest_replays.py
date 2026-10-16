"""Turn recorded replays into opponent truth (decisions.md, 2026-10-12).

Each replay recording shows both hands. It is processed twice: as `<id>` with layout replay_bottom (the bottom
player's hand, plus V4 unit detection into unit_detections_v5l_e15) and as `<id>_top` with layout replay_top (the top
player's hand only). The top player's plays become the bottom player's opponent truth:
data/ground_truth/opponent_truth_replay_<id>_m01.json (kept apart from the paired dev set's opponent_truth_auto_*).

IDs: bl_<date>_rNN (Battle Log, data/raw/replays/) and tv_<date>_rNN (TV Royale, data/raw/tv_royale/; the league tab
is in that folder's runner_log.jsonl). Links named by ID go in data/raw/replay_links/. Already-processed IDs are
skipped, so this can be re-run after every recording session.

  uv run python scripts/ingest_replays.py            # process every new replay, then write truth
  uv run python scripts/ingest_replays.py --truth    # only (re)write truth for processed replays
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import lancedb

ROOT = Path(__file__).resolve().parents[1]
LINKS = ROOT / "data/raw/replay_links"
V4_WEIGHTS, V4_TABLE = "data/models/bot77_detector/v5l/epoch15.pt", "unit_detections_v5l_e15"  # V4 = run v5l e15
SOURCES = {"bl": ROOT / "data/raw/replays", "tv": ROOT / "data/raw/tv_royale"}


def replays() -> dict[str, Path]:
    out = {}
    for prefix, root in SOURCES.items():
        for mp4 in sorted(root.glob("*/replay_*.mp4")):
            if time.time() - mp4.stat().st_mtime < 120:  # the recorder may still be remuxing it
                continue
            out[f"{prefix}_{mp4.parent.name}_r{mp4.stem.split('_')[1]}"] = mp4
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", action="store_true")
    a = ap.parse_args()
    db = lancedb.connect(ROOT / "data/lancedb")
    done = {r["video_id"] for r in db.open_table("videos").search().limit(10**6).select(["video_id"]).to_arrow().to_pylist()}
    reps = replays()
    if not a.truth:
        LINKS.mkdir(parents=True, exist_ok=True)
        jobs = []
        for rid, mp4 in reps.items():
            for vid, layout, gpu in ((rid, "replay_bottom", True), (f"{rid}_top", "replay_top", False)):
                link = LINKS / f"{vid}.mp4"  # distinct names, never a case-only variant of another file
                if not link.exists():
                    link.symlink_to(Path("..") / mp4.relative_to(ROOT / "data/raw"))
                if vid not in done:
                    jobs.append({"id": vid, "video": str(link.relative_to(ROOT)), "layout": layout,
                                 "creator": "replay", "gpu": gpu, "weights": V4_WEIGHTS, "table": V4_TABLE})
        print(f"{len(reps)} replays, {len(jobs)} videos to process")
        if jobs:
            jf = ROOT / "data/logs/run_all_jobs_replays.json"
            jf.write_text(json.dumps(jobs, indent=1))
            subprocess.run([sys.executable, str(ROOT / "scripts/run_all.py"), str(jf)], cwd=ROOT, check=True)
        done = {r["video_id"] for r in db.open_table("videos").search().limit(10**6).select(["video_id"]).to_arrow().to_pylist()}
    from bot77.pairing import write_replay_truth
    n = 0
    for rid in reps:
        if rid in done and f"{rid}_top" in done:
            n += len(write_replay_truth(rid, f"{rid}_top"))
    print(f"truth written for {n} matches")


if __name__ == "__main__":
    main()
