#!/bin/zsh
# A Cole session recording: process, auto-label, Dark Prince ability page, re-clean, recount, play counts.
# usage: scripts/after_session.sh <video_id> <jobs.json>
cd "$(dirname "$0")/.."
V=$1; J=$2
uv run python scripts/run_all.py "$J" > "data/logs/run_all_$V.out" 2>&1
uv run python scripts/autolabel_all.py 0 1 "$V" "$V" > "data/logs/autolabel_$V.log" 2>&1
uv run python scripts/build_click_abilities.py "data/reviews/click_abilities_$V.html" "$V" > "data/logs/click_abilities_$V.log" 2>&1
uv run python scripts/label_clicks.py > /dev/null 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > "data/logs/clean_$V.log" 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > "data/logs/coverage_$V.log"
uv run python scripts/session_click_page.py "$V" > "data/logs/session_page_$V.log" 2>&1
echo "SESSION_${V}_DONE $(date +%H:%M)" >> data/logs/run_all.log
