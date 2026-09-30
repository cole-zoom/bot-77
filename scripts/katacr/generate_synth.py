"""Generate a static synthetic detection dataset with KataCR's generator [Wu et al. 2025, §3.1,
Alg. 1] (MIT, third_party/katacr), including bot77's new classes, plus a validation split from
KataCR's hand-labelled real frames. Run with the KataCR environment:

  <katacr venv>/bin/python scripts/katacr/generate_synth.py <KataCR repo> <out dir> <n_train> <seed>
"""
import json, random, sys
from pathlib import Path

import cv2
import numpy as np

repo, out, n_train, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, str(repo))
from katacr.build_dataset.generator import Generator  # noqa: E402
from katacr.constants.label_list import unit_list  # noqa: E402

SKIP = {"selected", "text", "padding_0", "padding_1", "padding_2", "padding_belong"}
names = {i: n for i, n in enumerate(unit_list)}
for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

g = Generator(seed=seed, intersect_ratio_thre=0.5, map_update={"mode": "dynamic", "size": 5}, noise_unit_ratio=0.25)
for k in range(n_train):
    g.reset(); g.add_tower(); g.add_unit(40)
    img, box, _ = g.build(box_format="cxcywh", img_size=(576, 896))
    name = f"s{seed}_{k:06d}"
    cv2.imwrite(str(out / "images/train" / f"{name}.jpg"), cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    lines = [f"{int(b[5])} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in box if names[int(b[5])] not in SKIP]
    (out / "labels/train" / f"{name}.txt").write_text("\n".join(lines))
    if k % 500 == 0:
        print(k, flush=True)

# validation: KataCR's real labelled frames (class cx cy w h belong ...)
part2 = repo.parent / "unused"
real = Path(sys.argv[5]) if len(sys.argv) > 5 else None
if real and seed == 0:
    txts = sorted(p for p in real.rglob("*.txt") if p.with_suffix(".jpg").exists() and "annotation" not in p.name)
    random.Random(0).shuffle(txts)
    for p in txts[:800]:
        rows = [r.split() for r in p.read_text().splitlines() if r.strip()]
        lines = [" ".join(r[:5]) for r in rows if int(float(r[0])) in names and names[int(float(r[0]))] not in SKIP]
        stem = "_".join(p.relative_to(real).with_suffix("").parts)
        dst = out / "images/val" / f"{stem}.jpg"
        if not dst.exists():
            dst.symlink_to(p.with_suffix(".jpg").resolve())
        (out / "labels/val" / f"{stem}.txt").write_text("\n".join(lines))
    yaml = "path: " + str(out.resolve()) + "\ntrain: images/train\nval: images/val\nnames:\n" + "".join(
        f"  {i}: {n}\n" for i, n in names.items())
    (out / "data.yaml").write_text(yaml)
    print("val", len(list((out / 'labels/val').glob('*.txt'))))
