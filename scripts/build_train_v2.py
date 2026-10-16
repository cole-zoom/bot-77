"""Assemble the v2 detector training set (decisions.md, 2026-10-09) at data/external/train_v2:

  train: real_v2 frames from ~90% of videos  +  synth_v1 train (12,000 synthetic arenas)
  val:   real_v2 frames from the other ~10% of videos (split BY VIDEO: no near-duplicate frames across the split)
         +  synth_v1 val (800 of KataCR's hand-labelled real frames, the set v1 was scored on)

The benchmark hold-out (config/benchmark_holdout.json) is already absent from real_v2. Files are hard-linked, not
copied (same disk); `tar -h`/rsync copies them for upload. Class list: config/classes_v2.json (219; the first 158
match synth_v1's, checked below).
"""
import collections, json, os, random, shutil
from pathlib import Path

import yaml

REAL, SYN, OUT = Path("data/external/real_v2"), Path("data/external/synth_v1"), Path("data/external/train_v2")
VAL_SHARE, SEED = 0.10, 7
classes = json.loads(Path("config/classes_v2.json").read_text())
syn_names = list(yaml.safe_load(open(SYN / "data.yaml"))["names"].values())
assert syn_names == classes[:len(syn_names)], "synth class order must match classes_v2"

if OUT.exists():
    shutil.rmtree(OUT)
for s in ("train", "val"):
    (OUT / "images" / s).mkdir(parents=True); (OUT / "labels" / s).mkdir(parents=True)

man = json.loads((REAL / "manifest_clean.json").read_text())
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
assert not any(x.get("match_id") in hold for x in man), "benchmark frames leaked into real_v2"

# split by video; make sure every class seen in real data keeps most of its frames in train
vids = sorted({x["video_id"] for x in man})
rng = random.Random(SEED)
rng.shuffle(vids)
# greedy: a video joins val only if every entity keeps >= MIN_TRAIN_SHARE of its frames in train (a random 10% of videos
# took most of Boss Bandit's and evo Minion Horde's frames on the first try)
MIN_TRAIN_SHARE = 0.75
total = collections.Counter(x["entity"] for x in man)
by_vid = collections.defaultdict(collections.Counter)
for x in man:
    by_vid[x["video_id"]][x["entity"]] += 1
val_vids, in_val = set(), collections.Counter()
for v in vids:
    if sum(in_val.values()) >= VAL_SHARE * len(man):
        break
    if all(in_val[e] + n <= (1 - MIN_TRAIN_SHARE) * total[e] for e, n in by_vid[v].items()):
        val_vids.add(v); in_val.update(by_vid[v])


def link(src: Path, dst: Path):
    os.link(src, dst)


per = {"train": collections.Counter(), "val": collections.Counter()}
for x in man:
    s = "val" if x["video_id"] in val_vids else "train"
    link(REAL / "images" / f"{x['name']}.jpg", OUT / "images" / s / f"real_{x['name']}.jpg")
    link(REAL / "labels" / f"{x['name']}.txt", OUT / "labels" / s / f"real_{x['name']}.txt")
    per[s][x["entity"]] += 1
for s in ("train", "val"):
    for img in (SYN / "images" / s).iterdir():
        lbl = SYN / "labels" / s / f"{img.stem}.txt"
        if lbl.exists():
            link(img, OUT / "images" / s / f"syn_{img.name}")
            link(lbl, OUT / "labels" / s / f"syn_{img.stem}.txt")

# entities whose real frames ended up mostly in val would be starved in train: report them
starved = {e: (per["train"][e], per["val"][e]) for e in per["val"] if per["val"][e] > per["train"][e]}
yaml.safe_dump({"path": str(OUT.resolve()), "train": "images/train", "val": "images/val",
                "names": {i: c for i, c in enumerate(classes)}}, open(OUT / "data.yaml", "w"), sort_keys=False)
counts = {s: len(list((OUT / "images" / s).iterdir())) for s in ("train", "val")}
print(f"train {counts['train']} images, val {counts['val']} images ({len(val_vids)} val videos of {len(vids)})")
print("entities with more real frames in val than train:", starved or "none")
json.dump({"val_videos": sorted(val_vids), "seed": SEED}, open(OUT / "split.json", "w"), indent=1)
