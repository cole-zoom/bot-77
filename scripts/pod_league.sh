#!/bin/bash
# A self-play league round on a rented RunPod GPU pod (2026-11-08; Cole: "I have $12 in credits, u can use $10").
#
#   on the pod:  bash /workspace/bot-77/scripts/pod_league.sh setup              # clone, build, check the GPU
#                bash /workspace/bot-77/scripts/pod_league.sh smoke              # 8 episodes: speed + GPU path
#                bash /workspace/bot-77/scripts/pod_league.sh run <hours> <envs> <init> <run-name> [selfplay_league args]
#
# Expects /workspace/bot77_data.tgz (copied up beforehand): the gitignored files the league needs (the school
# checkpoints, the pro clone, the pro decks, the base's gate results). `run` trains for <hours> of wall clock, gates
# the last snapshot against school v7 (scripts/league_gate.py), then stops its own pod (if runpodctl and the pod id are
# there) so the credits stop running out; /workspace survives a stop.
set -euo pipefail
W=/workspace
R=$W/bot-77
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=$W/.uv-cache UV_PYTHON_INSTALL_DIR=$W/.uv-python

case "${1:-}" in
setup)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  [ -d $R ] || git clone --recursive https://github.com/cole-zoom/bot-77.git $R
  cd $R && git pull --ff-only && git submodule update --init --recursive
  tar -xzf $W/bot77_data.tgz -C $R
  uv sync
  uv pip install gymnasium==1.3.0 tensorboard
  .venv/bin/python -c "import torch, clash_royale_env; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  nproc
  ;;
smoke)
  cd $R
  rm -rf $W/league_smoke
  time .venv/bin/python scripts/selfplay_league.py --run $W/league_smoke --episodes 8 --envs "${2:-14}" --critic-warmup 2
  ;;
run)
  hours=$2 envs=$3 init=$4 name=$5
  shift 5
  cd $R
  .venv/bin/python scripts/selfplay_league.py --init "$init" --anchor data/models/bc_school_v7/epoch03.pth \
      --run data/models/$name --episodes 1000000 --hours "$hours" --envs "$envs" --snapshot-every 1000 "$@" \
      > $W/train_$name.log 2>&1 || echo "trainer exited $?" >> $W/train_$name.log
  last=$(ls data/models/$name/historical_checkpoints/ep*.pth | sort | tail -1)
  .venv/bin/python scripts/league_gate.py --cand "$last" --n 200 --workers "$(nproc)" > $W/gate_$name.log 2>&1 || true
  echo ALLDONE >> $W/gate_$name.log
  if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl stop pod "$RUNPOD_POD_ID"; fi
  ;;
*)
  echo "usage: $0 setup | smoke [envs] | run <hours> <envs> <init> <run-name> [selfplay_league args]"; exit 1
  ;;
esac
