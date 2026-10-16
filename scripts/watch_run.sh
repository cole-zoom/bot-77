#!/bin/bash
# Pull each periodic checkpoint of a pod training run and score it on the play benchmark as it lands
# (decisions.md, 2026-10-10: checkpoints are chosen by the benchmark, not by val mAP50-95).
#   watch_run.sh <ssh host> <ssh port> <remote run dir> <local dir> <tag prefix> <remote log>
host=$1; port=$2; rdir=$3; ldir=$4; tag=$5; rlog=$6
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30 -p $port root@$host"
mkdir -p "$ldir"
cd "$(dirname "$0")/.."
while true; do
  for w in $($SSH "ls $rdir/weights/ 2>/dev/null | grep -E '^epoch[0-9]+\.pt$'" </dev/null); do
    [ -f "$ldir/$w" ] && continue
    sleep 20  # let the write finish
    scp -q -o BatchMode=yes -P "$port" "root@$host:$rdir/weights/$w" "$ldir/$w" </dev/null || continue
    n=${w#epoch}; n=${n%.pt}
    uv run python scripts/eval_v2.py "$ldir/$w" "${tag}_e$n" > "data/logs/eval_${tag}_e$n.out" 2>&1
    echo "SCORED $w $(date +%H:%M)" >> "$ldir/watch.log"
  done
  if $SSH "grep -q TRAIN_EXIT $rlog" </dev/null; then
    for f in weights/best.pt weights/last.pt results.csv results.png args.yaml; do
      scp -q -o BatchMode=yes -P "$port" "root@$host:$rdir/$f" "$ldir/" </dev/null
    done
    scp -q -o BatchMode=yes -P "$port" "root@$host:$rlog" "$ldir/" </dev/null
    echo "RUN_DONE $(date +%H:%M)" >> "$ldir/watch.log"
    exit 0
  fi
  sleep 120
done
