#!/bin/bash
# V8 (decisions.md, 2026-10-17): the full V5 recipe (synth_v3 regenerated from the kit, all real labels) plus everything
# labelled since: failure-mining boxes, Cole's live training games and the 2026-10-17 replays (real_v8.tgz). Fine-tuned
# from V5 best; checkpoint picked by val mAP (val matches never train), then the test sets are scored once.
#   pod_v8.sh setup            # pod_v5.sh setup (genv + kit + unpack + synth_v3, CPU ~1 h) and unpack real_v8
#   pod_v8.sh train <init .pt> [epochs=12] [mined repeat=5] [live repeat=3]
# Expects in /root: train_v3_real.tar, kit_v3.tgz, segment_v3.tgz, real_v4.tgz, real_v5r.tgz, real_v8.tgz, pod_v5.sh.
set -euo pipefail
T=/root/train_v3
if [ "$1" = setup ]; then
  bash /root/pod_v5.sh setup
  tar -xzf /root/real_v8.tgz -C "$T" && find "$T" -name '._*' -delete
  echo SETUP_V8_OK
  exit
fi
init=$2; epochs=${3:-12}; repm=${4:-5}; repl=${5:-3}
cd "$T"
python3 pod_assemble.py
REPM=$repm REPL=$repl python3 - <<'PY'
import os
from pathlib import Path
root = Path("/root/train_v3")
g = lambda d: sorted(str(p) for p in (root / d).glob("*.jpg"))
real, real4, real5, syn = g("real_train/images"), g("real4_train/images"), g("real5_train/images"), g("synth_v3/images/train")
mined, live, rep = g("mined_train/images"), g("v8live_train/images"), g("v8rep_train/images")
m, l = int(os.environ["REPM"]), int(os.environ["REPL"])
(root / "train_v8.txt").write_text("\n".join(real * 3 + real4 + real5 + syn + mined * m + live * l + rep) + "\n")
y = (root / "data.yaml").read_text().replace("train: train.txt", "train: train_v8.txt")
y = y.replace("val: [real_val/images, katacr_val/images]", "val: [real_val/images, real4_val/images, real5_val/images, katacr_val/images]")
(root / "data_v8.yaml").write_text(y)
print(f"train_v8: {len(real)} real_v3 x3 + {len(real4)} real_v4 + {len(real5)} real_v5r + {len(syn)} synth + "
      f"{len(mined)} mined x{m} + {len(live)} live x{l} + {len(rep)} replays 10-17")
PY
setsid bash -c "python3 train_v2.py --data $T/data_v8.yaml --model $init --batch 16 --epochs $epochs --patience 0 --save-period 2 --project /workspace/runs --name v8l > /workspace/v8l.log 2>&1; echo TRAIN_EXIT \$? >> /workspace/v8l.log" </dev/null >/dev/null 2>&1 &
echo LAUNCHED_V8L
