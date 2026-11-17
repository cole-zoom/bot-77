#!/bin/bash
# The brain retrain with the new input block on a rented GPU (2026-11-17, step 6 of the input rebuild).
#   on the Mac:  bash scripts/pod/train.sh pack            # -> data/cache/pod_train.tgz (shards_t02, school_v3, review_v2,
#                                                          #    games_v10, v8, match results json; ~0.4 GB)
#   copy up:     rsync -a -e "ssh -p PORT" data/cache/pod_train.tgz root@IP:/workspace/
#   on the pod:  bash /workspace/bot-77/scripts/pod/train.sh setup   (clone, uv sync builds the engine, unpack, GPU check)
#                bash /workspace/bot-77/scripts/pod/train.sh run      (v10 = v9's recipe + the block; v10x = 3x the steps)
#   back:        rsync -a -e "ssh -p PORT" root@IP:/workspace/bot-77/data/models/bc_school_v10{,x} data/models/
# Plain ssh / rsync / bash: runs on any CUDA box, not only RunPod.
set -euo pipefail
W=/workspace
R=$W/bot-77
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$W/.uv-cache UV_PYTHON_INSTALL_DIR=$W/.uv-python
COMMON="--shards data/cache/bc/shards_t02 --demos data/cache/bc/school_v3 --demo-share 0.7 --tempo-share 0.333 \
  --init data/models/bc_school_v8/epoch03.pth --window 256 --burn-in 64 --play-weight 50 --steps-per-epoch 310 \
  --reviews data/cache/bc/review_v2 --games data/cache/bc/games_v10 --games-share 0.25 --lead 3 --extra-obs \
  --device auto --match-meta data/cache/bc/train_meta.json"

case "${1:-}" in
pack)
  cd "$(dirname "$0")/../.."
  COPYFILE_DISABLE=1 tar -czhf data/cache/pod_train.tgz data/cache/bc/shards_t02 data/cache/bc/school_v3 \
    data/cache/bc/review_v2 data/cache/bc/games_v10 data/cache/bc/train_meta.json data/models/bc_school_v8/epoch03.pth
  ls -la data/cache/pod_train.tgz
  ;;
setup)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -d $R ] || git clone --recursive https://github.com/cole-zoom/bot-77.git $R
  cd $R && git pull --ff-only && git submodule update --init --recursive
  tar -xzf $W/pod_train.tgz -C $R
  uv sync
  .venv/bin/python -c "import torch, clash_royale_env; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  nproc
  ;;
run)
  cd $R
  nohup .venv/bin/python scripts/train_bc.py $COMMON --epochs 4 --out data/models/bc_school_v10 > $W/train_v10.log 2>&1 &
  nohup .venv/bin/python scripts/train_bc.py $COMMON --epochs 12 --out data/models/bc_school_v10x > $W/train_v10x.log 2>&1 &
  echo "started: tail -f $W/train_v10.log $W/train_v10x.log"
  ;;
*) echo "usage: $0 pack|setup|run"; exit 1 ;;
esac
