#!/bin/bash
# Spell stage on a pod (decisions.md, 2026-10-14). Expects /root/spell_clips.tgz (and later /root/spell_streams.tgz).
#   pod_spell.sh train [epochs=14]     # unpack clips, train in the background -> /workspace/spell, log /workspace/spell.log
#   pod_spell.sh infer                 # unpack streams, run the classifier over whole games -> /workspace/spell_probs
set -euo pipefail
cd /root
if [ "$1" = train ]; then
  tar -xf spell_clips.tgz && find spell_clips -name '._*' -delete
  pip install -q --break-system-packages opencv-python-headless 2>&1 | tail -1
  python3 -c "from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights as W; r2plus1d_18(weights=W.KINETICS400_V1)"
  setsid bash -c "python3 /root/train_spell.py --data /root/spell_clips --out /workspace/spell --epochs ${2:-14} > /workspace/spell.log 2>&1; echo TRAIN_EXIT \$? >> /workspace/spell.log" </dev/null >/dev/null 2>&1 &
  echo LAUNCHED_SPELL
elif [ "$1" = infer ]; then
  tar -xf spell_streams.tgz && find spell_streams -name '._*' -delete
  python3 /root/infer_spell.py --model /workspace/spell/best.pt --streams /root/spell_streams --out /workspace/spell_probs > /workspace/infer.log 2>&1
  tail -1 /workspace/infer.log
fi
