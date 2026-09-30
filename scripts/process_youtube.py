"""Process every downloaded YouTube video end to end: layout (known creator, or auto-derived and
checked by eye afterwards via its calibration sheet), matches + POV plays, unit detection,
DINOv2 crops for mining. Skips videos already in the LanceDB `videos` table."""
import re
import subprocess
import sys
from pathlib import Path

import lancedb

from bot77.autolayout import derive_layout
from bot77.detect.run import detect_video
from bot77.layout import LAYOUT_DIR
from bot77.mine import build_crops
from bot77.video import probe

DB = Path("data/lancedb")
KNOWN = {"OYASSUU": "oyassuu_portrait"}
listing = Path("data/raw/youtube/downloaded.txt")

db = lancedb.connect(DB)
done = {v["video_id"] for v in db.open_table("videos").search().limit(10000).to_arrow().to_pylist()}
for line in listing.read_text().splitlines():
    vid, channel, title = [x.strip() for x in line.split("|")[:3]]
    if vid in done:
        continue
    video = next(Path("data/raw").glob(f"{vid}.*"))
    layout = KNOWN.get(channel)
    if layout is None:
        layout = re.sub(r"[^a-z0-9]+", "_", channel.lower()).strip("_") + "_auto"
        if not (LAYOUT_DIR / f"{layout}.json").exists():
            du = probe(video)["duration"]
            derive_layout(video, layout, [du * q for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)],
                          f"{channel} — auto-derived on {vid}")
        KNOWN[channel] = layout
    subprocess.run(["uv", "run", "bot77", "calibrate", "--layout", layout, "--video", str(video), "-n", "10"], check=False)
    print(f"== {vid} [{channel}] {title} -> {layout}", flush=True)
    subprocess.run(["uv", "run", "bot77", "process", "--video", str(video), "--layout", layout, "--creator", channel], check=True)
    detect_video(vid, DB)
    print(vid, build_crops(vid), "crops", flush=True)
