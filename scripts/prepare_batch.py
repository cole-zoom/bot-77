"""Build run_all jobs (argv[1]: output jobs file) for every downloaded video not yet processed. Each video gets the layout that
reads its clock best: its creator's known layout, or a per-video auto layout (decisions.md: calibrate per
video, not per creator). Videos where nothing reads the clock are reported and skipped."""
import json
from pathlib import Path

import lancedb

from bot77.autolayout import derive_layout
from bot77.calibrate import read_frame
from bot77.layout import Layout
from bot77.readers.clock import read_clock
from bot77.video import probe

import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/logs/run_all_jobs3.json"
# a creator's known layouts, including the hand-fitted ones from 2026-10-01 (same capture setup per creator)
KNOWN = {"OYASSUU": ["oyassuu_portrait"], "Ryley - Clash Royale": ["ryley_portrait", "auto_0m7iCgw8sYc"],
         "Ian77 - Clash Royale": ["ian_split"], "Rolex - Clash Royale": ["auto_Pn15Nh1xWoM"],
         "Mohamed Light": ["auto_xONsnIlyu1E"], "Mohamed Light Extra": ["auto_xONsnIlyu1E"],
         "Adrian P - Clash Royale": ["auto_ZXPr1gwF9ys"], "Viiper 1": ["auto_KjQYOTotuYE"]}
import time
while Path("data/raw/youtube/queue_viiper.txt").exists() and "DLV_DONE" not in Path("data/logs/run_all.log").read_text():
    time.sleep(60)  # Viiper is top priority (evo Valkyrie): wait for his downloads to join this batch
PRIO = [l.strip() for l in Path("data/raw/youtube/priority.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
PRIORITY_ONLY = "jobs5" in OUT  # batch 5 = Viiper + Tier-1 only (Cole, 2026-10-01); everything else -> batch 6


def finished(vid):
    return any(p.suffix in (".webm", ".mp4", ".mkv") and not p.name.split(".")[-2].startswith("f")
               for p in Path("data/raw").glob(f"{vid}.*"))


t0 = time.time()
while PRIORITY_ONLY and not all(finished(v) for v in PRIO) and time.time() - t0 < 4 * 3600:
    time.sleep(60)  # wait for the Tier-1 videos from queue5 too (they download first in that queue)
db = lancedb.connect("data/lancedb")
done = {v["video_id"] for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
jobs, report = [], []
seen = set()
for line in Path("data/raw/youtube/downloaded.txt").read_text().splitlines():
    vid, ch = line.split("|")[0].strip(), line.split("|")[1].strip()
    if vid in done or vid in seen or (PRIORITY_ONLY and vid not in PRIO):
        continue
    seen.add(vid)
    paths = [p for p in Path("data/raw").glob(f"{vid}.*") if p.suffix in (".webm", ".mp4", ".mkv") and ".part" not in p.name and not p.name.split(".")[-2].startswith("f")]
    if not paths:
        report.append(f"{vid} [{ch}]: no finished video file"); continue
    path = paths[0]
    du = probe(path)["duration"]
    probes = [du * q for q in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)]
    cands = list(KNOWN.get(ch, []))
    try:
        derive_layout(path, f"auto_{vid}", [du * q for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)], f"{ch} — per-video auto layout")
        cands.append(f"auto_{vid}")
    except Exception as e:  # noqa: BLE001
        report.append(f"{vid} [{ch}]: auto layout failed ({e})")
    best, best_n = None, -1
    for lid in cands:
        lay = Layout.load(lid)
        if tuple(lay.frame_size) != (probe(path)["width"], probe(path)["height"]) and not lid.startswith("auto_"):
            continue
        try:
            n = sum(read_clock(read_frame(path, t), lay).elapsed_s is not None for t in probes)
        except Exception as e:  # noqa: BLE001 - a broken file must not stop the batch
            report.append(f"{vid} [{ch}]: {lid} unreadable ({e.__class__.__name__})"); continue
        if n > best_n:
            best, best_n = lid, n
    report.append(f"{vid} [{ch}]: {best} reads the clock on {best_n}/6 frames")
    if best and best_n >= 3:
        jobs.append({"id": vid, "video": str(path), "layout": best, "creator": ch})
jobs.sort(key=lambda j: PRIO.index(j["id"]) if j["id"] in PRIO else len(PRIO))  # Viiper + Tier-1 first
Path(OUT).write_text(json.dumps(jobs, indent=1))
Path(OUT.replace(".json", "_prepare.log")).write_text("\n".join(report))
print("\n".join(report)); print(len(jobs), "jobs")
