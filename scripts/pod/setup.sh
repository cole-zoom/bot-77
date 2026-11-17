#!/usr/bin/env bash
# The pro-replay read job on a rented Linux CUDA box (step 5 of the input rebuild, 2026-11-16). Run on the pod, from
# /workspace/bot77 (the code tarball unpacked by scripts/pod/pack.sh's instructions):
#   bash scripts/pod/setup.sh
# Installs ffmpeg and the reader's Python deps on top of the image's CUDA torch, then checks the detector sees the GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq ffmpeg rsync > /dev/null
# the image's torch (CUDA) stays; the rest pinned to what the Mac runs (uv.lock)
pip install -q "ultralytics==8.4.160" "opencv-python-headless>=5.0.0.93" "numpy>=2.0" pillow pyyaml
python - <<'EOF'
import torch, cv2, ultralytics
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("opencv", cv2.__version__, "ultralytics", ultralytics.__version__)
import sys; sys.path.insert(0, "src")
import bot77.live.state, bot77.detect.yolo  # the reader's imports (no macOS-only modules)
print("bot77 imports ok")
EOF
ffmpeg -version | head -1
nvidia-smi --query-gpu=name,memory.total --format=csv
nproc
