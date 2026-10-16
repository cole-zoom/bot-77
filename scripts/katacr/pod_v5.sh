#!/bin/bash
# V5 on a fresh pod (decisions.md, 2026-10-14): rebuild V4's training set (real_v3 x3, real_v4 x1, synth_v3 regenerated
# from the kit), add the replay clock-spot labels (real_v5r, build_train_v5.py) and fine-tune from V4 (v5l epoch 15).
#   pod_v5.sh setup                      # genv + kit + unpack + synth_v3 (CPU, ~1 h)
#   pod_v5.sh train <init .pt> [epochs=15] [real_v5r repeat=1]
# Expects in /root: train_v3_real.tar, kit_v3.tgz, segment_v3.tgz, real_v4.tgz, real_v5r.tgz.
set -euo pipefail
T=/root/train_v3; K=/root/kit_v4
if [ "$1" = setup ]; then
  # nproc shows the host's CPUs; the pod's share is the cgroup quota (16 vCPUs on the 4090 pod)
  cpus=$(awk '$1 != "max" {print int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || true)
  [ -z "$cpus" ] && cpus=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 1) ))
  [ "$cpus" -lt 2 ] 2>/dev/null && cpus=$(nproc)  # no quota
  procs=${2:-$(( cpus - 2 ))}; per=$(( 48000 / procs + 1 ))
  cd /root
  tar -xf train_v3_real.tar && tar -xzf kit_v3.tgz
  tar -xzf real_v4.tgz -C "$T" && tar -xzf real_v5r.tgz -C "$T"
  pip install -q --break-system-packages uv ultralytics 2>&1 | tail -1
  uv venv -q --python 3.12 /root/genv
  VIRTUAL_ENV=/root/genv uv pip install -q "numpy<2" opencv-python-headless pillow tqdm scipy matplotlib "jax==0.4.26" "jaxlib==0.4.26"
  rm -rf "$K"; mkdir -p "$K/dataset/images"
  cp -r /root/kit_v3/KataCR /root/kit_v3/generate_synth.py /root/kit_v3/gen_parallel.sh "$K/"
  cp -r /root/kit_v3/dataset/images/backgrounds "$K/dataset/images/" 2>/dev/null || true
  tar -xzf /root/segment_v3.tgz -C "$K/dataset/images"
  find /root/kit_v3 "$K" "$T" -name '._*' -delete
  sed -i "s|^path_dataset = Path(\".*\")$|path_dataset = Path(\"$K/dataset\")|" "$K/KataCR/katacr/build_dataset/constant.py"
  rm -rf "$T/synth_v3"
  bash "$K/gen_parallel.sh" "$K/KataCR" "$T/synth_v3" /root/genv/bin/python "$procs" "$per"
  echo SETUP_OK synth $(ls "$T/synth_v3/images/train" | wc -l)
  exit
fi
init=$2; epochs=${3:-15}; rep5=${4:-1}
cd "$T"
python3 pod_assemble.py  # data.yaml with the class names
REP5=$rep5 python3 - <<'EOF'
import os
from pathlib import Path
root = Path("/root/train_v3")
g = lambda d: sorted(str(p) for p in (root / d).glob("*.jpg"))
real, real4, real5, syn = g("real_train/images"), g("real4_train/images"), g("real5_train/images"), g("synth_v3/images/train")
rep5 = int(os.environ["REP5"])
(root / "train_v6.txt").write_text("\n".join(real * 3 + real4 + real5 * rep5 + syn) + "\n")
y = (root / "data.yaml").read_text().replace("train: train.txt", "train: train_v6.txt")
y = y.replace("val: [real_val/images, katacr_val/images]", "val: [real_val/images, real4_val/images, real5_val/images, katacr_val/images]")
(root / "data_v6.yaml").write_text(y)
print(f"train_v6: {len(real)} real_v3 x3 + {len(real4)} real_v4 + {len(real5)} real_v5r x{rep5} + {len(syn)} synth_v3")
EOF
setsid bash -c "python3 train_v2.py --data $T/data_v6.yaml --model $init --batch 16 --epochs $epochs --patience 0 --save-period 5 --project /workspace/runs --name v6l > /workspace/v6l.log 2>&1; echo TRAIN_EXIT \$? >> /workspace/v6l.log" </dev/null >/dev/null 2>&1 &
echo LAUNCHED_V6L
