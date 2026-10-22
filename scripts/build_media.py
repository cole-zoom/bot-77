"""The `media` table (training-architecture phase 0, 2026-10-22): one row per video / recording file the project uses,
with where it really lives. data/raw keeps symlinks to the LaCie for cold files, so a path can look local and only work
while the drive is plugged in; this table says which is which.

Columns: path (as the code uses it, repo-relative), real_path, kind (pov / youtube / replay / tv_royale / bot_game /
lacie_only), tier (laptop / lacie), bytes, available (readable now), sha1_1mb (first MB, to spot moved / duplicate
files), video_id (when the `videos` table knows it).

  uv run python scripts/build_media.py
"""
import hashlib, os
from pathlib import Path

import lancedb
import pyarrow as pa

EXT = {".mp4", ".mkv", ".webm", ".mov"}
LACIE = "/Volumes/LaCie"
db = lancedb.connect("data/lancedb")
videos = db.open_table("videos").search().select(["video_id", "path"]).limit(10**6).to_arrow().to_pylist()
vid_of = {os.path.realpath(v["path"]): v["video_id"] for v in videos}  # most go through data/raw/replay_links


def kind(p: str) -> str:
    return ("bot_game" if "/live_games/" in p else "replay" if "/replays/" in p else "tv_royale" if "/tv_royale/" in p
            else "pov" if Path(p).name.startswith("cole_") else "youtube" if p.startswith("data/raw/") else "other")


def sha(p: Path) -> str | None:
    try:
        with open(p, "rb") as f:
            return hashlib.sha1(f.read(1 << 20)).hexdigest()
    except OSError:
        return None


rows, seen = [], set()
for p in sorted(Path("data/raw").rglob("*")):
    if p.suffix.lower() not in EXT or "replay_links" in p.parts:
        continue
    real = Path(os.path.realpath(p))
    ok = real.exists()
    rows.append({"path": str(p), "real_path": str(real), "kind": kind(str(p)),
                 "tier": "lacie" if str(real).startswith(LACIE) else "laptop", "bytes": real.stat().st_size if ok else None,
                 "available": ok, "sha1_1mb": sha(real) if ok else None, "video_id": vid_of.get(str(real))})
    seen.add(str(real))
lacie = Path(LACIE) / "bot-77-data"
if lacie.exists():  # files on the drive nothing in the repo points to (e.g. Cole's original .mov recordings)
    for p in sorted(lacie.rglob("*")):
        if p.suffix.lower() in EXT and str(p) not in seen and not p.name.startswith("._"):
            real = Path(os.path.realpath(p))
            if str(real) in seen:
                continue
            ok = real.exists()  # two links on the drive point at recordings that aren't there (2026-10-22)
            rows.append({"path": None, "real_path": str(p), "kind": "lacie_only", "tier": "lacie",
                         "bytes": real.stat().st_size if ok else None, "available": ok, "sha1_1mb": sha(real) if ok else None,
                         "video_id": None})
db.create_table("media", pa.Table.from_pylist(rows), mode="overwrite")
from collections import Counter
c = Counter((r["kind"], r["tier"]) for r in rows)
print(f"media: {len(rows)} files, {sum(r['bytes'] or 0 for r in rows) / 1e9:.0f} GB;", dict(c))
missing = [v for v in videos if os.path.realpath(v["path"]) not in {r["real_path"] for r in rows}]
print("unavailable:", sum(not r["available"] for r in rows), " videos rows without a media file:", len(missing),
      [v["path"] for v in missing[:5]])
