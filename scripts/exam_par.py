"""The situation exam split across processes (2026-10-29): one scripts/situation_exam.py per template (and one for
Cole's reference starts), run side by side, the outputs joined in template order. Same seeds, same numbers as one
serial run (every start is seeded on its own); just ~10x faster on a 10-core machine.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/exam_par.py --out data/models/X/exam.txt -- --ckpt C --n 50 --reference

Everything after `--` goes to situation_exam.py unchanged (except --only / --reference-only, set here).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--templates", nargs="*", default=None, help="default: all of bot77.rl.situations.TEMPLATES")
    a = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    if a.templates is None:
        sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
        from bot77.rl import situations as sit
        a.templates = [t.__name__ for t in sit.TEMPLATES]
    reference = "--reference" in rest
    base = [x for x in rest if x != "--reference"]
    jobs = [(t, base + ["--only", t]) for t in a.templates]
    if reference:
        jobs.append(("_reference", base + ["--reference", "--reference-only"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = out.parent / (out.stem + "_parts")
    parts.mkdir(exist_ok=True)
    cmd0 = ["uv", "run", "--no-sync", "--with", "gymnasium==1.3.0", "python", str(ROOT / "scripts/situation_exam.py")]
    running, pending, t0 = [], list(jobs), time.time()
    while pending or running:
        while pending and len(running) < a.workers:
            name, args = pending.pop(0)
            args = [x if not x.endswith(".jsonl") else str(parts / f"{name}.jsonl") for x in args]
            f = open(parts / f"{name}.txt", "w")
            running.append((name, subprocess.Popen(cmd0 + args, stdout=f, stderr=subprocess.DEVNULL, cwd=ROOT), f))
        for r in list(running):
            if r[1].poll() is not None:
                r[2].close()
                running.remove(r)
                print(f"{r[0]} done ({time.time() - t0:.0f} s)", flush=True)
        time.sleep(2)
    header, rows, tail = None, [], []
    for name, _ in jobs:
        lines = [l for l in (parts / f"{name}.txt").read_text().splitlines() if "warn" not in l.lower()]
        if name == "_reference":
            tail = [l for l in lines if l.startswith("Cole") or l.startswith("score")]
            continue
        header = header or lines[0]
        rows += [l for l in lines[1:] if not l.startswith("score")]
        tail = tail or [l for l in lines if l.startswith("score")]
    out.write_text("\n".join([header or ""] + rows + [""] + tail) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
