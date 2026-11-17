#!/usr/bin/env bash
# The pro-replay board reads on the pod (step 5, 2026-11-16). From /workspace/bot77:
#   bash scripts/pod/run.sh [WORKERS]          (default: 6 processes sharing the GPU)
# Reads /workspace/pod_job/{manifest.json,cards.json,videos/} and writes /workspace/pod_job/reads/<match>.npz, one per
# match (resumable: a restart skips the done ones). Log: /workspace/pod_job/reads.log. ~0.2 GB of reads for all 484 (~1.9 KB per match second, every raw detection kept).
set -euo pipefail
cd "$(dirname "$0")/../.."
W=${1:-6}
JOB=/workspace/pod_job
nohup python scripts/read_replay_boards.py --manifest $JOB/manifest.json --videos $JOB/videos --out $JOB/reads \
    --weights data/models/bot77_detector/v6l/best.pt --device cuda:0 --workers "$W" >> $JOB/reads.log 2>&1 &
echo "started (pid $!): tail -f $JOB/reads.log"
