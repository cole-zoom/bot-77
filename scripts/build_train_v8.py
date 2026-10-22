"""Upload bundle of the labels made since V5 (decisions.md, 2026-10-17), one folder per source so the pod can weight
them: mined_train (failure mining, Cole-reviewed), v8live_train (Cole's live training games keyed by their replays,
clock-spot labels), v8rep_train (the Battle Log replays recorded 2026-10-17, minus the replays of the live test games).
Held-out games (bot77.holdout) are refused.

  uv run python scripts/build_train_v8.py      -> data/upload/real_v8.tgz
"""
import json, os, shutil, subprocess
from pathlib import Path

import cv2

from bot77.holdout import held_out

SRCS = {"mined_train": ["data/external/real_mined", "data/external/real_mined_r2"],
        "v8live_train": ["data/external/real_v8_live"], "v8rep_train": ["data/external/real_v8_replay"]}
OUT, MAX_SIDE = Path("data/upload/real_v8_bundle"), 896
hold = held_out()
if OUT.exists():
    shutil.rmtree(OUT)
n = {}
for dst, srcs in SRCS.items():
    d = OUT / dst
    (d / "images").mkdir(parents=True); (d / "labels").mkdir(parents=True)
    man = [{**x, "src": Path(S)} for S in srcs if (Path(S) / "manifest.json").exists()
           for x in json.loads((Path(S) / "manifest.json").read_text())]
    assert not any(x["match_id"] in hold for x in man), f"a held-out game leaked into {dst}"
    if dst == "v8live_train":  # Cole's review of these labels (2026-10-18): only the boxes he kept
        rv = Path("data/ground_truth/reviews/live_labels_review_2026-10-17.json")
        keep = {r["id"].removesuffix(".jpg") for r in json.loads(rv.read_text())["slices"] if r["verdict"] == "keep"}
        man = [x for x in man if x["name"] in keep]
    for x in man:
        img = cv2.imread(str(x["src"] / "images" / f"{x['name']}.jpg"))
        s = MAX_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(d / "images" / f"{x['name']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy(x["src"] / "labels" / f"{x['name']}.txt", d / "labels" / f"{x['name']}.txt")
    n[dst] = len(man)
subprocess.run(["tar", "--no-xattrs", "-czf", "data/upload/real_v8.tgz", "-C", str(OUT), *SRCS],
               check=True, env={**os.environ, "COPYFILE_DISABLE": "1"})
print(n, "-> data/upload/real_v8.tgz")
