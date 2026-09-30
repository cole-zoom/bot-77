"""Process a batch of videos with the CPU and GPU busy at the same time.

Stage 1 (CPU pool): convert iPhone Mirroring recordings to the phone region, then `bot77 process`
(clock, hand, elixir, POV plays). Stage 2 (GPU pool): unit detection + DINOv2 crops. A video moves to
stage 2 as soon as its stage 1 finishes. Pool sizes are chosen for an M5 with 16 GB: 2 + 2 keeps the
CPU and GPU busy at ~10 GB peak (batch-16 training at 16 GB thrashed; see decisions.md).
Each job logs to data/logs/run_all_<video>.log; progress goes to data/logs/run_all.log.
"""
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data/logs/run_all.log"
CPU_WORKERS, GPU_WORKERS = 2, 2
lock = threading.Lock()


def log(msg: str) -> None:
    with lock:
        with LOG.open("a") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def sh(cmd: list[str], logfile: Path) -> None:
    with logfile.open("a") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{cmd[:4]} failed ({r.returncode}), see {logfile.name}")


def stage_cpu(job: dict) -> dict:
    lf = ROOT / f"data/logs/run_all_{job['id']}.log"
    t0 = time.time()
    if job.get("convert"):
        src, crop = job["convert"]
        if not (ROOT / job["video"]).exists():
            sh(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"crop={crop},scale=1080:2340:flags=lanczos,fps=30",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", job["video"]], lf)
        log(f"{job['id']}: converted ({time.time() - t0:.0f}s)")
    sh(["uv", "run", "bot77", "process", "--video", job["video"], "--layout", job["layout"], "--creator", job["creator"]], lf)
    log(f"{job['id']}: processed ({time.time() - t0:.0f}s)")
    return job


def stage_gpu(job: dict) -> None:
    lf = ROOT / f"data/logs/run_all_{job['id']}.log"
    t0 = time.time()
    code = ("from pathlib import Path; from bot77.detect.run import detect_video; from bot77.mine import build_crops; "
            f"detect_video('{job['id']}', Path('data/lancedb')); print(build_crops('{job['id']}'), 'crops')")
    sh(["uv", "run", "python", "-c", code], lf)
    log(f"{job['id']}: detected + embedded ({time.time() - t0:.0f}s)")


def main(jobs: list[dict]) -> None:
    log(f"start: {len(jobs)} videos, {CPU_WORKERS} CPU + {GPU_WORKERS} GPU workers")
    errors = []
    with ThreadPoolExecutor(CPU_WORKERS) as cpu, ThreadPoolExecutor(GPU_WORKERS) as gpu:
        gpu_futs = []

        def done_cpu(fut):
            try:
                gpu_futs.append(gpu.submit(stage_gpu, fut.result()))
            except Exception as e:  # noqa: BLE001
                errors.append(str(e)); log(f"ERROR {e}")

        for j in jobs:
            cpu.submit(stage_cpu, j).add_done_callback(done_cpu)
        cpu.shutdown(wait=True)
        for f in list(gpu_futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                errors.append(str(e)); log(f"ERROR {e}")
    log(f"all done; {len(errors)} errors")


if __name__ == "__main__":
    import json
    main(json.loads(Path(sys.argv[1]).read_text()))
