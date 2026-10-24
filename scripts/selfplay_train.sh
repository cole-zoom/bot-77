#!/usr/bin/env bash
# Train the ClashRoyaleEnv fork's own PPO stack (MIT, third_party/clash-royale-env/python_ai)
# for OUR deck: Giant, Mini PEKKA, Musketeer, Goblin Hut, Fireball, Arrows, Skeletons, Minions.
#
#   scripts/selfplay_train.sh              # start, or resume if the checkpoint exists
#   RUN_DIR=data/models/foo scripts/selfplay_train.sh   # a separate arm
#   SMOKE_SECONDS=300 scripts/selfplay_train.sh         # stop after N seconds (SIGINT -> saves)
#
# Phase 1 (trainers/train.py): PPO vs the scripted UtilityTeacher on the 11-rung
# lookahead ladder, opponent decks drawn per episode from the 16-deck meta pool
# (the fork's default; the mirror alone leaves cards dead -- fork .claude/CLAUDE.md
# "THE OPPONENT PLAYS REAL META DECKS"). It hands off to phase 2
# (train_selfplay.py, PFSP league) by itself, in a subprocess that inherits this env.
#
# Everything lands under $RUN_DIR (gitignored data/models/):
#   model_weights.pth            phase-1 live checkpoint (+ .prev), resumed automatically
#   model_weights_selfplay.pth   phase-2 live checkpoint
#   runs/phase1, runs/phase1_selfplay   TensorBoard
#   runs/training_selfplay_pfsp*.log    phase-2 stdout/stderr after the handoff
#   historical_checkpoints/ stage_checkpoints/ replays/   (CLASH_RUN_ROOT)
#   logs/phase1.<ts>.log         this script's console log
#
# Resume = run the same command again. Keep CLASH_* identical across resumes: the
# checkpoint stamps them and warns on drift.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="$ROOT/third_party/clash-royale-env"
RUN_DIR="${RUN_DIR:-data/models/selfplay_v1}"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$ROOT/$RUN_DIR" ;; esac
mkdir -p "$RUN_DIR/logs"

export CLASH_DECK="${CLASH_DECK:-Giant,Mini PEKKA,Musketeer,Goblin Hut,Fireball,Arrows,Skeletons,Minions}"
export CLASH_WEIGHTS="$RUN_DIR/model_weights.pth"      # absolute: honoured verbatim
export CLASH_LOGDIR="$RUN_DIR/runs/phase1"             # absolute; phase 2 uses ${CLASH_LOGDIR}_selfplay
export CLASH_RUN_ROOT="$RUN_DIR"                       # snapshots/replays off the fork tree
# 10-core M5 (4P+6E) shared with other jobs: 7 env workers. Rollout (workers) and
# the PPO update (torch threads in the main process) do not overlap, so peak use
# is ~7 cores either way.
export CLASH_NUM_ENVS="${CLASH_NUM_ENVS:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-7}"
export CLASH_SAVE_EVERY="${CLASH_SAVE_EVERY:-250}"
export PYTHONUNBUFFERED=1

LOG="$RUN_DIR/logs/phase1.$(date +%Y%m%d-%H%M%S).log"
echo "run dir: $RUN_DIR"
echo "log:     $LOG"
[ -f "$CLASH_WEIGHTS" ] && echo "resuming from $CLASH_WEIGHTS" || echo "fresh start"

# --no-sync: the engine is a path dependency another thread rebuilds; never let a
# launch or resume trigger a rebuild. gymnasium/tensorboard come from the overlay,
# not the project's pyproject (same overlay as the fork's test suite).
CMD=(uv run --no-sync --project "$ROOT" --with gymnasium==1.3.0 --with tensorboard
     python -u python_ai/trainers/train.py)

cd "$FORK"
if [ -n "${SMOKE_SECONDS:-}" ]; then
  # Job control on: without it bash starts background jobs with SIGINT ignored,
  # and Python then never raises the KeyboardInterrupt that saves the checkpoint.
  set -m
  caffeinate -is "${CMD[@]}" > "$LOG" 2>&1 &
  pid=$!
  sleep "$SMOKE_SECONDS"
  # caffeinate keeps $! and execs uv; python train.py is a descendant. SIGINT only
  # the trainer (not its env workers) so its Ctrl-C handler saves.
  trainer=$(pgrep -f "bin/python -u python_ai/trainers/train.py" | while read -r p; do
    q=$p; while [ "$q" -gt 1 ]; do [ "$q" = "$pid" ] && { echo "$p"; break; }; q=$(ps -o ppid= -p "$q" | tr -d ' '); done
  done | head -1)
  kill -INT "${trainer:-$pid}" 2>/dev/null || true
  while kill -0 "${trainer:-$pid}" 2>/dev/null; do sleep 2; done
  wait "$pid" 2>/dev/null || true
  echo "smoke done; log: $LOG"
else
  exec caffeinate -is "${CMD[@]}" > >(tee -a "$LOG") 2>&1
fi
