"""Run jobs one at a time (CPU stage then GPU stage) next to a running batch, when memory has room for one
more job but not a second 2+2 pool. Same stages and logs as run_all.py."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_all import log, stage_cpu, stage_gpu

for job in json.loads(Path(sys.argv[1]).read_text()):
    try:
        stage_gpu(stage_cpu(job))
    except Exception as e:  # noqa: BLE001
        log(f"ERROR {e}")
