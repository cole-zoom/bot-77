#!/bin/zsh
# Session A (evo Goblin Giant, evo Inferno Dragon, hero Dark Prince): convert now; process after re-read round 1 so
# only one job writes the events table at a time; then auto-label, Dark Prince ability page, re-clean, recount.
cd "$(dirname "$0")/.."
nice -n 10 ffmpeg -v error -y -i data/raw/cole_2026-10-07.mov -vf "crop=876:1900:1074:64,scale=1080:2340:flags=lanczos,fps=30" \
  -c:v libx264 -preset fast -crf 18 -an data/raw/cole_2026-10-07.mp4
until grep -q REPROCESS_r1_DONE data/logs/run_all.log; do sleep 60; done
uv run python scripts/run_all.py data/logs/run_all_jobs_c1007.json > data/logs/run_all_c1007.out 2>&1
uv run python scripts/autolabel_all.py 0 1 c1007 cole_2026-10-07 > data/logs/autolabel_c1007.log 2>&1
uv run python scripts/build_click_abilities.py data/reviews/click_abilities3.html cole_2026-10-07 > data/logs/click_abilities3.log 2>&1
rm -rf data/external/real_v2
uv run python scripts/clean_dataset.py > data/logs/clean_c1007.log 2>&1
uv run python scripts/coverage.py data/external/real_v2 2>&1 | grep -vi warn > data/logs/coverage_c1007.log
echo "C1007_DONE $(date +%H:%M)" >> data/logs/run_all.log
