"""Ingest Cole's live games + the in-game replays he records after each one (2026-10-16), and write answer keys for
the live recordings: the replay shows both hands, its top hand is the opponent's, and bot77.pairing.write_live_truth
moves those plays onto the live video's time line (pairing checked by Cole's own plays agreeing).

Files: a folder of iPhone Mirroring recordings named <name>_live.mov / <name>_replay.mov (e.g. g01_live.mov,
g01_replay.mov). Each is cropped to the phone (same crop as the earlier recordings) and processed: the live one with
layout cole_iphone_mirroring (+ V5 unit detection), the replay twice (bottom hand + V4 detection; top hand only).

  uv run python scripts/ingest_live_replays.py /Volumes/LaCie/bot-77-data/live_2026-10-16 [--crop W:H:X:Y]
Writes data/ground_truth/opponent_truth_live_<live match>.json (score with eval_dev.py --set live).
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ("data/models/bot77_detector/v5l/epoch15.pt", "unit_detections_v5l_e15")
V5 = ("data/models/bot77_detector/v6l/best.pt", "unit_detections_v6l_best")

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--crop", default="876:1900:1074:64", help="iPhone Mirroring window crop (w:h:x:y), as for 09-30 .. 10-07")
ap.add_argument("--replay-bottom", default="cole_replay_bottom", help="layout of a replay seen through mirroring")
ap.add_argument("--replay-top", default="cole_replay_top")
ap.add_argument("--truth", action="store_true", help="only (re)write the answer keys")
a = ap.parse_args()

folder = Path(a.folder)
tag = folder.name.replace(" ", "_")
games = sorted({p.stem.rsplit("_", 1)[0] for p in folder.glob("*_live.*")} & {p.stem.rsplit("_", 1)[0] for p in folder.glob("*_replay.*")})
print(len(games), "games with both a live recording and a replay:", games)
jobs, ids = [], []
for g in games:
    for kind in ("live", "replay"):
        src = next(folder.glob(f"{g}_{kind}.*"))
        vid = f"cl_{tag}_{g}" if kind == "live" else f"cr_{tag}_{g}"
        out = ROOT / "data/raw/live" / f"{vid}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind == "live":
            jobs.append({"id": vid, "video": str(out.relative_to(ROOT)), "layout": "cole_iphone_mirroring", "creator": "Cole",
                         "convert": [str(src), a.crop], "gpu": True, "weights": V5[0], "table": V5[1]})
        else:
            jobs.append({"id": vid, "video": str(out.relative_to(ROOT)), "layout": a.replay_bottom, "creator": "replay",
                         "convert": [str(src), a.crop], "gpu": True, "weights": V4[0], "table": V4[1]})
            top = ROOT / "data/raw/live" / f"{vid}_top.mp4"
            if not top.exists():
                top.symlink_to(out.name)
            jobs.append({"id": f"{vid}_top", "video": str(top.relative_to(ROOT)), "layout": a.replay_top,
                         "creator": "replay", "gpu": False})
    ids.append((f"cl_{tag}_{g}", f"cr_{tag}_{g}"))
if not a.truth and jobs:
    # replay top-hand jobs need the converted file from the bottom job: run bottoms first
    first = [j for j in jobs if not j["id"].endswith("_top")]
    rest = [j for j in jobs if j["id"].endswith("_top")]
    for batch, name in ((first, "a"), (rest, "b")):
        jf = ROOT / f"data/logs/run_all_jobs_live_{tag}_{name}.json"
        jf.write_text(json.dumps(batch, indent=1))
        subprocess.run([sys.executable, str(ROOT / "scripts/run_all.py"), str(jf)], cwd=ROOT, check=True)
from bot77.pairing import write_live_truth  # noqa: E402
n = 0
for live, rep in ids:
    n += len(write_live_truth(live, rep, f"{rep}_top"))
print(f"answer keys written for {n} live matches")
