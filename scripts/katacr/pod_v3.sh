#!/bin/bash
# Run 3 on the pod (decisions.md, 2026-10-10): synth_v3 from the game-art sprite folder, then fine-tune from run 2's
# best checkpoint (picked on the play benchmark, not mAP).
#   pod_v3.sh <init weights .pt> [epochs=25] [n_images_per_proc=2000] [procs=24]
# Expects /root/segment_v3.tgz (setup_synth_v3.py's segment folder) and run 2's /root/kit_v3, /root/genv, /root/train_v3.
set -euo pipefail
init=$1; epochs=${2:-25}; per=${3:-2000}; procs=${4:-24}
K=/root/kit_v4; T=/root/train_v3

rm -rf "$K"; mkdir -p "$K/dataset/images"
cp -r /root/kit_v3/KataCR /root/kit_v3/generate_synth.py /root/kit_v3/gen_parallel.sh "$K/"
cp -r /root/kit_v3/dataset/images/backgrounds "$K/dataset/images/" 2>/dev/null || true
tar -xzf /root/segment_v3.tgz -C "$K/dataset/images"
find "$K" -name '._*' -delete
sed -i "s|^path_dataset = Path(\".*\")$|path_dataset = Path(\"$K/dataset\")|" "$K/KataCR/katacr/build_dataset/constant.py"
grep -n "^path_dataset" "$K/KataCR/katacr/build_dataset/constant.py"

rm -rf "$T/synth_v3"
bash "$K/gen_parallel.sh" "$K/KataCR" "$T/synth_v3" /root/genv/bin/python "$procs" "$per"

cd "$T"
python3 - <<'EOF'
from pathlib import Path
root = Path("/root/train_v3")
real = sorted(str(p) for p in (root / "real_train/images").glob("*.jpg"))
syn = sorted(str(p) for p in (root / "synth_v3/images/train").glob("*.jpg"))
(root / "train_v4.txt").write_text("\n".join(real * 3 + syn) + "\n")
y = (root / "data.yaml").read_text().replace("train: train.txt", "train: train_v4.txt")
(root / "data_v4.yaml").write_text(y)
print(f"train_v4: {len(real)} real x3 + {len(syn)} synth_v3")
EOF
setsid bash -c "python3 train_v2.py --data $T/data_v4.yaml --model $init --batch 16 --epochs $epochs --patience 0 --save-period 5 --project /workspace/runs --name v4l > /workspace/v4l.log 2>&1; echo TRAIN_EXIT \$? >> /workspace/v4l.log" </dev/null >/dev/null 2>&1 &
echo LAUNCHED_V4
