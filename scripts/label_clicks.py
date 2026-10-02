import json, glob
from pathlib import Path
from bot77.autolabel import AutoLabeler, label_clicks
al = AutoLabeler()
out = []
for f in sorted(glob.glob("data/ground_truth/reviews/clicks_*.json")):
    out += label_clicks(al, Path(f))
Path("data/external/real_v1/manifest_clicks.json").write_text(json.dumps(out, indent=1))
print(len(out), "click frames")
