#!/bin/zsh
# After batch 4: auto-label its videos (2 workers), recount coverage, build the next click page.
cd "$(dirname "$0")/.."
until [ "$(grep -c BATCH4_DONE data/logs/run_all.log)" -ge 1 ]; do sleep 60; done
V=0m7iCgw8sYc,H35F0vVMTHg,Pn15Nh1xWoM,XuM18Vfd-xc,ZXPr1gwF9ys,xONsnIlyu1E
uv run python scripts/autolabel_all.py 0 2 b4_0 $V > data/logs/autolabel_b4_0.log 2>&1 &
uv run python scripts/autolabel_all.py 1 2 b4_1 $V > data/logs/autolabel_b4_1.log 2>&1 &
wait
uv run python scripts/coverage.py 2>&1 | grep -vi warn > data/logs/coverage_batch4.log
uv run python scripts/build_click_short.py data/logs/coverage_batch4.log data/reviews/click_short4.html > data/logs/click_short4.log 2>&1
echo "AFTER_BATCH4_DONE $(date +%H:%M)" >> data/logs/run_all.log
