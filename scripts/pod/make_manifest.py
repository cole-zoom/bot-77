"""The pro-replay read job's inputs (step 5, 2026-11-16): on the Mac, from LanceDB, everything the GPU box needs to run
scripts/read_replay_boards.py without the database.

  uv run --no-sync python scripts/pod/make_manifest.py --out /Volumes/LaCie/bot-77-data/pod_job [--link]
      [--transcode --crf 18] [--matches N]

Writes <out>/manifest.json (per match of shards_t01's 484: match_id, video file name, layout, t_start / t_end, the
bottom HUD's clock and elixir every 0.1 s), <out>/cards.json (the deck cards: id, name, type, elixir), and
<out>/videos.txt (each source video's local path and size). --link: symlinks <out>/videos/<video_id>.mp4 to the
originals (for rsync -L); --transcode: re-encodes each video at 10 fps instead (smaller upload; see
decisions.md for the read agreement measured against the originals).
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", default=str(ROOT / "data/cache/bc/shards_t01"), help="the matches to cover")
    ap.add_argument("--link", action="store_true")
    ap.add_argument("--transcode", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--matches", type=int, default=0)
    a = ap.parse_args()
    import lancedb
    from bot77.sim import store
    db = lancedb.connect(str(ROOT / "data/lancedb"))
    out = Path(a.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    want = sorted({f.stem.rsplit("_", 1)[0] for f in Path(a.shards).glob("*.npz")})
    if a.matches:
        want = want[: a.matches]
    ms = {m["match_id"]: m for m in db.open_table("matches").search().where(
        "source IN ('replay_tv', 'replay_bl')").limit(10**5).to_arrow().to_pylist()}
    vids = {v["video_id"]: v for v in db.open_table("videos").search().limit(10**5).to_arrow().to_pylist()}
    cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
    (out / "cards.json").write_text(json.dumps([{k: c[k] for k in ("id", "name", "type", "elixir")} for c in cards]))
    man, lines, total = [], [], 0
    for mid in want:
        m = ms[mid]
        v = vids[m["video_id"]]
        src = (ROOT / v["path"]).resolve()
        hud = sorted(store.read("hud_states", f"match_id = '{mid}'"), key=lambda r: r["t_video"])
        name = f"{m['video_id']}.mp4"
        t_start, t_end = float(m["t_start"]), float(m["t_end"] or hud[-1]["t_video"])
        entry = {"match_id": mid, "video": name, "layout_id": v["layout_id"], "t_start": t_start, "t_end": t_end,
                 "hud_t": [round(r["t_video"], 2) for r in hud], "hud_match": [r["t_match"] for r in hud],
                 "hud_elixir": [None if r["elixir"] is None else round(r["elixir"], 2) for r in hud]}
        dst = out / "videos" / name
        if a.transcode and not dst.exists():
            # the whole video (one match each, so the timeline and t_start / hud_t stay as they are) at 10 fps,
            # what the reader samples
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "fps=10", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", str(a.crf), "-an", str(dst)], check=True)
        elif a.link and not dst.exists():
            dst.symlink_to(src)
        size = dst.stat().st_size if a.transcode else src.stat().st_size
        total += size
        lines.append(f"{src}\t{size}")
        man.append(entry)
    (out / "manifest.json").write_text(json.dumps({"matches": man}))
    (out / "videos.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(man)} matches, {total / 1e9:.1f} GB of video -> {out}")


if __name__ == "__main__":
    main()
