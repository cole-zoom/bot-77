#!/bin/bash
# V4 mining chain A (decisions.md, 2026-10-10): wait for the frame sync and V3's epoch-10 checkpoint, detect every
# extracted play-window frame on the mining pod, pull the detections back. Prints MINING_DONE.
#   mine_v4_chainA.sh <train host> <train port> <mine host> <mine port> <sync log> [checkpoint=epoch10.pt]
set -uo pipefail
TH=$1; TP=$2; MH=$3; MP=$4; SYNCLOG=$5; CK=${6:-epoch10.pt}
S="-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
cd "$(dirname "$0")/.."
until grep -q SYNC_FINISHED "$SYNCLOG"; do sleep 60; done
echo "frames synced $(date +%H:%M)"
rsync -rt -e "ssh $S -p $MP" data/upload/mine_v4/manifest.jsonl root@$MH:/root/mine_v4/
until ssh $S -p $TP root@$TH "test -f /workspace/runs/v4l/weights/$CK"; do sleep 120; done
mkdir -p data/models/bot77_detector/v4l
scp $S -P $TP root@$TH:/workspace/runs/v4l/weights/$CK data/models/bot77_detector/v4l/mine_$CK
scp $S -P $MP data/models/bot77_detector/v4l/mine_$CK root@$MH:/root/mine_weights.pt
echo "detecting with v4l $CK $(date +%H:%M)"
ssh $S -p $MP root@$MH "cd /root && python3 mine_v4_detect.py detect /root/mine_weights.pt /root/mine_v4 > /root/detect.log 2>&1; tail -2 /root/detect.log"
scp $S -P $MP root@$MH:/root/mine_v4/detections.jsonl data/upload/mine_v4/detections.jsonl
wc -l data/upload/mine_v4/detections.jsonl data/upload/mine_v4/manifest.jsonl
echo MINING_DONE
