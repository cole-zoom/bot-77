#!/bin/zsh
# Cole's Spirit Empress ground / evo Wizard / hero Dark Prince dismounted recording: after the b/c/d chain,
# process serially, auto-label, ability click page for Dark Prince, recount, click page.
cd "$(dirname "$0")/.."
until grep -q AFTER_C1001BCD_DONE data/logs/run_all.log; do sleep 60; done
uv run python scripts/run_serial.py data/logs/run_all_jobs_c1001e.json > data/logs/run_serial_c1001e.out 2>&1
uv run python scripts/autolabel_all.py 0 1 c1001e cole_2026-10-01_e > data/logs/autolabel_c1001e.log 2>&1
uv run python scripts/build_click_abilities.py data/reviews/click_abilities2.html cole_2026-10-01_e > data/logs/click_abilities2.log 2>&1
uv run python scripts/coverage.py 2>&1 | grep -vi warn > data/logs/coverage_c1001e.log
uv run python scripts/build_click_short.py data/logs/coverage_c1001e.log data/reviews/click_short5b.html > data/logs/click_short5b.log 2>&1
echo "AFTER_C1001E_DONE $(date +%H:%M)" >> data/logs/run_all.log
