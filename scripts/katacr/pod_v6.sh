#!/bin/bash
# V6 (decisions.md, 2026-10-16): short fine-tune of V5 best on the real data plus the failure-mining labels (Cole's
# review of never-seen plays, real_mined.tgz, repeated so they weigh in). No synth regeneration: a few epochs from V5
# best only need the real frames, and setup would cost an hour of pod time.
#   pod_v6.sh train <init .pt> [epochs=4] [mined repeat=10]
# Expects in /root: train_v3_real.tar, real_v4.tgz, real_v5r.tgz, real_mined.tgz.
set -euo pipefail
T=/root/train_v3
init=$2; epochs=${3:-4}; repm=${4:-10}
cd /root
[ -d "$T/real5_train" ] || { tar -xf train_v3_real.tar && tar -xzf real_v4.tgz -C "$T" && tar -xzf real_v5r.tgz -C "$T" \
  && tar -xzf real_mined.tgz -C "$T"; find "$T" -name '._*' -delete; }
pip install -q --break-system-packages ultralytics 2>&1 | tail -1
cd "$T"
python3 pod_assemble.py  # data.yaml with the class names
REPM=$repm python3 - <<'EOF'
import os
from pathlib import Path
root = Path("/root/train_v3")
g = lambda d: sorted(str(p) for p in (root / d).glob("*.jpg"))
real, real4, real5, mined = g("real_train/images"), g("real4_train/images"), g("real5_train/images"), g("mined_train/images")
repm = int(os.environ["REPM"])
(root / "train_v7.txt").write_text("\n".join(real * 3 + real4 + real5 + mined * repm) + "\n")
y = (root / "data.yaml").read_text().replace("train: train.txt", "train: train_v7.txt")
y = y.replace("val: [real_val/images, katacr_val/images]", "val: [real_val/images, real4_val/images, real5_val/images, katacr_val/images]")
(root / "data_v7.yaml").write_text(y)
print(f"train_v7: {len(real)} real_v3 x3 + {len(real4)} real_v4 + {len(real5)} real_v5r + {len(mined)} mined x{repm}")
EOF
setsid bash -c "python3 train_v2.py --data $T/data_v7.yaml --model $init --batch 16 --epochs $epochs --patience 0 --save-period 1 --project /workspace/runs --name v7l > /workspace/v7l.log 2>&1; echo TRAIN_EXIT \$? >> /workspace/v7l.log" </dev/null >/dev/null 2>&1 &
echo LAUNCHED_V7L
