#!/bin/bash
# V4 on the pod (decisions.md, 2026-10-11): add the data engine's real_v4 frames to run 3's training list and
# fine-tune from V3's checkpoint picked on the dev set.
#   pod_v4.sh <init weights .pt> [epochs=25] [real_v4 repeat=1]
# Expects /root/real_v4.tgz (build_train_v4.py) and run 3's /root/train_v3 (real_train, synth_v3, data.yaml).
set -euo pipefail
init=$1; epochs=${2:-25}; rep4=${3:-1}
T=/root/train_v3
rm -rf "$T/real4_train" "$T/real4_val"
tar -xzf /root/real_v4.tgz -C "$T"
find "$T" -name '._*' -delete
cd "$T"
REP4=$rep4 python3 - <<'EOF'
import os
from pathlib import Path
root = Path("/root/train_v3")
g = lambda d: sorted(str(p) for p in (root / d).glob("*.jpg"))
real, real4, syn = g("real_train/images"), g("real4_train/images"), g("synth_v3/images/train")
rep4 = int(os.environ["REP4"])
(root / "train_v5.txt").write_text("\n".join(real * 3 + real4 * rep4 + syn) + "\n")
y = (root / "data.yaml").read_text().replace("train: train.txt", "train: train_v5.txt")
y = y.replace("val: [real_val/images, katacr_val/images]", "val: [real_val/images, real4_val/images, katacr_val/images]")
(root / "data_v5.yaml").write_text(y)
print(f"train_v5: {len(real)} real_v3 x3 + {len(real4)} real_v4 x{rep4} + {len(syn)} synth_v3")
EOF
setsid bash -c "python3 train_v2.py --data $T/data_v5.yaml --model $init --batch 16 --epochs $epochs --patience 0 --save-period 5 --project /workspace/runs --name v5l > /workspace/v5l.log 2>&1; echo TRAIN_EXIT \$? >> /workspace/v5l.log" </dev/null >/dev/null 2>&1 &
echo LAUNCHED_V5L
