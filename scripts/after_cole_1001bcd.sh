#!/bin/zsh
# Cole's hero Valkyrie / evo Baby Dragon / evo Goblin Drill recordings: process serially beside batch 5,
# auto-label, recount, click page (Cole's recordings sort first).
cd "$(dirname "$0")/.."
uv run python scripts/run_serial.py data/logs/run_all_jobs_c1001bcd.json > data/logs/run_serial_c1001bcd.out 2>&1
echo "C1001BCD_DONE $(date +%H:%M)" >> data/logs/run_all.log
V=cole_2026-10-01_b,cole_2026-10-01_c,cole_2026-10-01_d
uv run python scripts/autolabel_all.py 0 1 c1001bcd $V > data/logs/autolabel_c1001bcd.log 2>&1
uv run python scripts/coverage.py 2>&1 | grep -vi warn > data/logs/coverage_c1001bcd.log
uv run python scripts/build_click_short.py data/logs/coverage_c1001bcd.log data/reviews/click_short5a.html > data/logs/click_short5a.log 2>&1
echo "AFTER_C1001BCD_DONE $(date +%H:%M)" >> data/logs/run_all.log
