"""Upload bundle of the failure-mining labels (decisions.md, 2026-10-16): data/external/real_mined (+ _r2), all train
(held-out games were never mined). Fine-tuned in next to V5's data by scripts/katacr/pod_v6.sh.

  uv run python scripts/build_train_mined.py      -> data/upload/real_mined.tgz (mined_train/)
"""
import json, os, shutil, subprocess
from pathlib import Path

import cv2

SRCS, OUT = [Path("data/external/real_mined"), Path("data/external/real_mined_r2")], Path("data/upload/real_mined_bundle")
MAX_SIDE = 896
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
for f in ("config/v5_check_games.json", "config/replay_test_games.json", "config/heldback_replays.json"):
    hold |= set(json.loads(Path(f).read_text())["matches"])
man = [{**x, "src": S} for S in SRCS for x in json.loads((S / "manifest.json").read_text())]
from bot77.holdout import held_out
hold |= held_out()
assert not any(x["match_id"] in hold for x in man), "a held-out game leaked"
if OUT.exists():
    shutil.rmtree(OUT)
d = OUT / "mined_train"
(d / "images").mkdir(parents=True); (d / "labels").mkdir(parents=True)
for x in man:
    img = cv2.imread(str(x["src"] / "images" / f"{x['name']}.jpg"))
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(d / "images" / f"{x['name']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.copy(x["src"] / "labels" / f"{x['name']}.txt", d / "labels" / f"{x['name']}.txt")
subprocess.run(["tar", "--no-xattrs", "-czf", "data/upload/real_mined.tgz", "-C", str(OUT), "mined_train"],
               check=True, env={**os.environ, "COPYFILE_DISABLE": "1"})
print(len(man), "frames -> data/upload/real_mined.tgz")
