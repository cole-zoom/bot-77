"""Turn mined crops Cole approved (✓ on data/reviews/mining.html) into full training frames in real_v1, written by
the auto-labeler's frame writer so the other units in the frame get the same grey-box handling. The cleaner treats
them as verified (how = "mined").
  uv run python scripts/mine_import.py <mining export.json>"""
import json, sys
from pathlib import Path
from bot77.autolabel import AutoLabeler, OUT

verdict = {f["id"]: f.get("verdict") for f in json.loads(Path(sys.argv[1]).read_text())["frames"]}
mined = [m for m in json.loads(Path("data/external/mined_v1/manifest_0.json").read_text()) if verdict.get(m["name"]) == "right"]
al, out = AutoLabeler(), []
for m in mined:
    v = al.videos.get(m["video_id"])
    if v is None:
        continue
    from bot77.layout import Layout
    lay = Layout.load(v["layout_id"])
    t = m["t_video"]
    old = al.dets.search().where(f"match_id = '{m['match_id']}' AND t_video >= {t - 0.15} AND t_video <= {t + 0.15}") \
        .limit(10**4).to_arrow().to_pylist()
    item = {"video_id": m["video_id"], "match_id": m["match_id"], "card": m["entity"], "source": f"mined-{m['team']}"}
    r = al._write(item, v, lay, t, (m["x0"], m["y0"], m["x1"], m["y1"]), m["entity"], "mined", m["p"], old)
    if r:
        out.append(r)
prev = OUT / "manifest_mined.json"
old_m = json.loads(prev.read_text()) if prev.exists() else []
names = {x["name"] for x in old_m}
prev.write_text(json.dumps(old_m + [x for x in out if x["name"] not in names], indent=1))
print(len(out), "mined frames written of", len(mined), "approved")
