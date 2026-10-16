"""V4 data engine, step 1 (decisions.md, 2026-10-10): extract the detector's arena crops around every trusted play so
the GPU pod can re-detect them with the newest detector (uploading the 69 GB of source video would take ~9 h).

For each auto-label item (bot77.autolabel.collect_items: POV plays of new cards and evo/hero forms, paired opponent
plays; benchmark matches excluded) the window [t-0.5, t+12] s is sampled at the detector's 4 fps, windows are merged
per match, and each frame's arena crop is saved at the detector input size (576x896) with the transform back to frame
pixels.
  uv run python scripts/mine_v4_extract.py [--jobs 8]
Writes data/upload/mine_v4/frames/<match_id>/<t>.jpg and data/upload/mine_v4/manifest.jsonl (gitignored).
"""
import argparse, collections, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

OUT = Path("data/upload/mine_v4")
FPS, BEFORE, AFTER = 4, 0.5, 12.0


def work(args):
    video, path, layout_id, windows = args
    from bot77.detect.units import INPUT_WH, UnitDetector
    from bot77.layout import Layout
    from bot77.video import iter_frames
    lay = Layout.load(layout_id)
    rows = []
    for match_id, a, b in windows:
        d = OUT / "frames" / match_id
        d.mkdir(parents=True, exist_ok=True)
        for t, f in iter_frames(Path(path), FPS, start=a, duration=b - a):
            crop, (ox, oy), sx, sy = UnitDetector.arena(None, f, lay)
            img = cv2.resize(crop, INPUT_WH, interpolation=cv2.INTER_AREA)
            name = f"{t:.2f}"
            if not (d / f"{name}.jpg").exists():  # resumable
                cv2.imwrite(str(d / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            rows.append({"match_id": match_id, "video_id": video, "t_video": round(t, 3), "file": f"frames/{match_id}/{name}.jpg",
                         "ox": ox, "oy": oy, "sx": sx, "sy": sy})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    import lancedb
    from bot77.autolabel import collect_items
    db = lancedb.connect("data/lancedb")
    hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
    vids = {v["video_id"]: v for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
    items = [i for i in collect_items() if i["match_id"] not in hold]
    per = collections.defaultdict(list)
    for i in items:
        per[(i["video_id"], i["match_id"])].append(i["t_video"])
    jobs = collections.defaultdict(list)
    for (v, m), ts in per.items():
        ws = []
        for t in sorted(ts):
            s, e = round((t - BEFORE) * FPS) / FPS, t + AFTER
            if ws and s <= ws[-1][1] + 0.5:
                ws[-1][1] = max(ws[-1][1], e)
            else:
                ws.append([s, e])
        jobs[v] += [(m, s, e) for s, e in ws]
    total = sum(e - s for w in jobs.values() for _, s, e in w)
    print(f"{len(items)} items, {len(per)} matches, {len(jobs)} videos, ~{int(total * FPS)} frames", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "manifest.jsonl", "w") as fh, ProcessPoolExecutor(a.jobs) as pool:
        for rows in pool.map(work, [(v, vids[v]["path"], vids[v]["layout_id"], w) for v, w in jobs.items()]):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            n += len(rows)
            print(f"{n} frames", flush=True)
    print("done", n)


if __name__ == "__main__":
    main()
