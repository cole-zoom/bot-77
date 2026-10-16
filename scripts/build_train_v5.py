"""Assemble the V5 replay upload (decisions.md, 2026-10-14): the replay clock-spot labels (real_v5r, plus real_v5r_new
for plays the 10-14 truth fix named), split by match so val matches never train. Goes up next to V4's data
(real_v3 split, real_v4.tgz, synth_v3 regenerated on the pod).

  uv run python scripts/build_train_v5.py      -> data/upload/real_v5r.tgz (real5_train/, real5_val/)
"""
import hashlib, json, os, shutil, subprocess
from pathlib import Path

import cv2

SRCS, OUT = [Path("data/external/real_v5r"), Path("data/external/real_v5r_new")], Path("data/upload/real_v5r_bundle")
VAL_SHARE = 0.1  # of replay matches, by a hash of the match id (stable across rebuilds)
MAX_SIDE = 896  # training resolution: smaller upload, no full-size decode every epoch
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
hold |= set(json.loads(Path("config/v5_check_games.json").read_text())["matches"])
hold |= set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
man = [{**x, "src": S} for S in SRCS if (S / "manifest.json").exists() for x in json.loads((S / "manifest.json").read_text())]
assert not any(x["match_id"] in hold for x in man), "a held-out game leaked"
is_val = lambda m: int(hashlib.sha1(m.encode()).hexdigest(), 16) % 1000 < VAL_SHARE * 1000
if OUT.exists():
    shutil.rmtree(OUT)
n, seen = {"real5_train": 0, "real5_val": 0}, set()
for x in man:
    if x["name"] in seen:
        continue
    seen.add(x["name"])
    d = "real5_val" if is_val(x["match_id"]) else "real5_train"
    (OUT / d / "images").mkdir(parents=True, exist_ok=True); (OUT / d / "labels").mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(x["src"] / "images" / f"{x['name']}.jpg"))
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:  # YOLO labels are normalised, so they don't change
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / d / "images" / f"{x['name']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.copy(x["src"] / "labels" / f"{x['name']}.txt", OUT / d / "labels" / f"{x['name']}.txt")
    n[d] += 1
val_matches = sorted({x["match_id"] for x in man if is_val(x["match_id"])})
(OUT / "split.json").write_text(json.dumps({"val_matches": val_matches}, indent=1))
subprocess.run(["tar", "--no-xattrs", "-czf", "data/upload/real_v5r.tgz", "-C", str(OUT), "real5_train", "real5_val", "split.json"],
               check=True, env={**os.environ, "COPYFILE_DISABLE": "1"})
print(n, f"{len(val_matches)} val matches -> data/upload/real_v5r.tgz")
