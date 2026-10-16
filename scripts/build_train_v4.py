"""Assemble the V4 upload (decisions.md, 2026-10-11): the real_v4 frames mined by the data engine, split by video
exactly like train_v2 (val videos never train). The pod already holds run 2/3's real_v3 split, KataCR val and
synth_v3, so only real_v4 goes up.

  uv run python scripts/build_train_v4.py      -> data/upload/real_v4.tgz (real4_train/, real4_val/)
"""
import json, os, shutil, subprocess
from pathlib import Path

SRCS, OUT = [Path("data/external/real_v4"), Path("data/external/real_v4s")], Path("data/upload/real_v4_bundle")
# real_v4s: opponent spells from paired matches (autolabel_paired_spells.py)
val_vids = set(json.loads(Path("data/external/train_v2/split.json").read_text())["val_videos"])
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
man = [{**x, "src": S} for S in SRCS for x in json.loads((S / "manifest.json").read_text())]
assert not any(x["match_id"] in hold for x in man), "benchmark leaked"
if OUT.exists():
    shutil.rmtree(OUT)
n = {"real4_train": 0, "real4_val": 0}
for x in man:
    d = "real4_val" if x["video_id"] in val_vids else "real4_train"
    (OUT / d / "images").mkdir(parents=True, exist_ok=True); (OUT / d / "labels").mkdir(parents=True, exist_ok=True)
    os.link(x["src"] / "images" / f"{x['name']}.jpg", OUT / d / "images" / f"{x['name']}.jpg")
    shutil.copy(x["src"] / "labels" / f"{x['name']}.txt", OUT / d / "labels" / f"{x['name']}.txt")
    n[d] += 1
subprocess.run(["tar", "--no-xattrs", "-czf", "data/upload/real_v4.tgz", "-C", str(OUT), "real4_train", "real4_val"],
               check=True, env={**os.environ, "COPYFILE_DISABLE": "1"})
print(n, "-> data/upload/real_v4.tgz")
