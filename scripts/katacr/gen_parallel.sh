#!/bin/bash
# Run KataCR's generator on many cores (synth_v2, decisions.md 2026-10-10): one process per seed, each writing
# n_per images. Seeds are distinct per process, so images never repeat.
#   gen_parallel.sh <KataCR repo> <out dir> <python> <n_procs> <n_per>
repo=$1; out=$2; py=$3; procs=$4; per=$5
here=$(cd "$(dirname "$0")" && pwd)
cd "$repo"
for s in $(seq 1 "$procs"); do
  "$py" "$here/generate_synth.py" "$repo" "$out" "$per" "$((100 + s))" > "$out.gen_$s.log" 2>&1 &
done
wait
echo GEN_DONE $(ls "$out/images/train" | wc -l)
