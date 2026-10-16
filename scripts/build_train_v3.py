"""Assemble the run-2 training bundle (decisions.md, 2026-10-10) at data/upload/train_v3/:

  real/   real_v3 frames (empty-arena composites, no unlabelled troops), split by video exactly like train_v2
  realfull_val/  real_v2 val frames as they are (full frames, sparse labels): recall on real footage
  katacr_val/    KataCR's 800 hand-labelled real frames (synth_v1 val)
  kit/    KataCR generator (patched) + sprite folder, to make synth_v2 ON the pod (64 cores) instead of uploading it
  pod_assemble.py  run on the pod after generation: writes data.yaml + train.txt (real frames repeated REAL_REPEAT x)

  uv run python scripts/build_train_v3.py <KataCR repo, patched by setup_synth_v2.py> <its dataset dir>
"""
import json, os, shutil, sys
from pathlib import Path

repo, ds = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path("data/upload/train_v3")
REAL3, REAL2, SYN1 = Path("data/external/real_v3"), Path("data/external/real_v2"), Path("data/external/synth_v1")
classes = json.loads(Path("config/classes_v2.json").read_text())
val_vids = set(json.loads(Path("data/external/train_v2/split.json").read_text())["val_videos"])
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
if OUT.exists():
    shutil.rmtree(OUT)


def put(src_img, src_lbl, d, name):
    (OUT / d / "images").mkdir(parents=True, exist_ok=True); (OUT / d / "labels").mkdir(parents=True, exist_ok=True)
    os.link(src_img, OUT / d / "images" / f"{name}.jpg")
    shutil.copy(src_lbl, OUT / d / "labels" / f"{name}.txt")


man3 = json.loads((REAL3 / "manifest.json").read_text())
assert not any(x["match_id"] in hold for x in man3), "benchmark leaked"
n = {"real_train": 0, "real_val": 0}
for x in man3:
    s = "val" if x["video_id"] in val_vids else "train"
    put(REAL3 / "images" / f"{x['name']}.jpg", REAL3 / "labels" / f"{x['name']}.txt", f"real_{s}", x["name"]); n[f"real_{s}"] += 1
man2 = json.loads((REAL2 / "manifest_clean.json").read_text())
for x in man2:
    if x["video_id"] in val_vids:
        put(REAL2 / "images" / f"{x['name']}.jpg", REAL2 / "labels" / f"{x['name']}.txt", "realfull_val", x["name"])
for img in (SYN1 / "images/val").iterdir():
    put(img.resolve(), SYN1 / "labels/val" / f"{img.stem}.txt", "katacr_val", img.stem)
shutil.copytree(repo, OUT / "kit/KataCR", ignore=shutil.ignore_patterns(".git", "runs", "logs", "__pycache__"))
shutil.copytree(ds / "images/segment", OUT / "kit/dataset/images/segment")
for f in ("generate_synth.py", "gen_parallel.sh"):
    shutil.copy(Path("scripts/katacr") / f, OUT / "kit" / f)
shutil.copy("scripts/train_v2.py", OUT / "train_v2.py")
shutil.copy("config/classes_v2.json", OUT / "classes_v2.json")
(OUT / "pod_assemble.py").write_text('''"""On the pod: point the generator at the kit, then write data.yaml / train.txt for training."""
import json, re, sys
from pathlib import Path
root = Path(__file__).resolve().parent
REAL_REPEAT = 3
if sys.argv[1:] == ["kit"]:
    p = root / "kit/KataCR/katacr/build_dataset/constant.py"
    p.write_text(re.sub(r'^path_dataset = Path\\(".*"\\)$', f'path_dataset = Path("{root / "kit/dataset"}")', p.read_text(), flags=re.M))
    print("kit ->", root / "kit/dataset"); sys.exit()
classes = json.loads((root / "classes_v2.json").read_text())
real = sorted(str(p) for p in (root / "real_train/images").glob("*.jpg"))
syn = sorted(str(p) for p in (root / "synth_v2/images/train").glob("*.jpg"))
(root / "train.txt").write_text("\\n".join(real * REAL_REPEAT + syn) + "\\n")
for name, val in (("data.yaml", ["real_val/images", "katacr_val/images"]), ("data_realfull.yaml", ["realfull_val/images"]),
                  ("data_realv3.yaml", ["real_val/images"]), ("data_katacr.yaml", ["katacr_val/images"])):
    (root / name).write_text("path: " + str(root) + "\\ntrain: train.txt\\nval: [" + ", ".join(val) + "]\\nnames:\\n" +
                             "".join(f"  {i}: {c}\\n" for i, c in enumerate(classes)))
print(f"train: {len(real)} real x{REAL_REPEAT} + {len(syn)} synth")
''')
print(n, "realfull_val", len(list((OUT / "realfull_val/images").iterdir())), "katacr_val", len(list((OUT / "katacr_val/images").iterdir())))
