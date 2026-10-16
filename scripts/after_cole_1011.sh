#!/bin/zsh
# Cole 2026-10-11: one match, both sides played by Cole, both recorded (A 2:47:28 PM, B 2:47:53 PM). Process, then pair.
cd "$(dirname "$0")/.."
uv run python scripts/run_all.py data/logs/run_all_jobs_c1011.json > data/logs/run_all_c1011.out 2>&1
uv run python - <<'PY' > data/logs/pair_c1011.log 2>&1
from bot77.pairing import write_pair_truth
print(write_pair_truth("cole_2026-10-11_a", "cole_2026-10-11_b", -25.0))
PY
echo "C1011_DONE $(date +%H:%M)" >> data/logs/run_all.log
