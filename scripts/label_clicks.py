"""All Cole's click exports -> real-frame labels (manifest_clicks.json).

Incremental (2026-10-08): each click file's result is cached by its content hash plus a version of the click-to-box
code, so a new review round only processes the new file (a full re-label took ~1 h). Pass --force to redo all, and
bump CODE_VERSION whenever label_clicks/box logic changes.
"""
import glob, hashlib, json, sys
from pathlib import Path

CODE_VERSION = "2026-10-08a"  # detector box when it contains the click; drop SAM boxes on the deploy stopwatch
CACHE = Path("data/external/real_v1/clicks_cache"); CACHE.mkdir(parents=True, exist_ok=True)
force = "--force" in sys.argv
out, todo = [], []
for f in sorted(glob.glob("data/ground_truth/reviews/clicks_*.json")):
    key = hashlib.sha1(Path(f).read_bytes() + CODE_VERSION.encode()).hexdigest()[:16]
    c = CACHE / f"{Path(f).stem}_{key}.json"
    if c.exists() and not force:
        out += json.loads(c.read_text())
    else:
        todo.append((f, c))
if todo:
    from bot77.autolabel import AutoLabeler, label_clicks
    al = AutoLabeler()
    for f, c in todo:
        res = label_clicks(al, Path(f))
        c.write_text(json.dumps(res)); out += res
Path("data/external/real_v1/manifest_clicks.json").write_text(json.dumps(out, indent=1))
print(len(out), "click frames;", len(todo), "click files (re)labelled")
