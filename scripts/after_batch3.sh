#!/bin/zsh
# Chain for batch 3: wait for downloads, pick layouts, process (2 CPU + 2 GPU), rerun zero-match, auto-label, recount.
cd "$(dirname "$0")/.."
uv run python scripts/prepare_batch3.py > data/logs/prepare_batch3_stdout.log 2>&1
cp data/logs/run_all_jobs3.json data/logs/run_all_jobs.json.batch3
uv run python scripts/run_all.py data/logs/run_all_jobs3.json > data/logs/run_all3_stdout.log 2>&1
python3 - <<'PY'
import json; j=json.load(open("data/logs/run_all_jobs3.json")); json.dump(j, open("data/logs/run_all_jobs.json","w"), indent=1)
PY
uv run python scripts/rerun_zero_match.py > data/logs/rerun3.log 2>&1
uv run python scripts/run_all.py data/logs/rerun_jobs.json >> data/logs/rerun3.log 2>&1
rm -rf data/external/real_v1/images data/external/real_v1/labels data/external/real_v1/manifest_[01].json
(uv run python scripts/autolabel_all.py 0 2 > data/logs/autolabel_0.log 2>&1 & uv run python scripts/autolabel_all.py 1 2 > data/logs/autolabel_1.log 2>&1 & wait)
uv run python scripts/label_clicks.py > data/logs/label_clicks.log 2>&1
echo BATCH3_DONE >> data/logs/run_all.log
