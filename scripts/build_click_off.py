"""Click page from candidate frames Cole marked "~ box off" (right unit, bad box): he clicks the unit and the
click becomes the label (SAM from the click point), same as other click pages."""
import json, sys
from pathlib import Path
import lancedb
from bot77.autolabel import MULTI, UNITS
from bot77.clicks import write_click_page_times

review, out = Path(sys.argv[1]), Path(sys.argv[2])
off = {f["id"] for f in json.loads(review.read_text())["frames"] if f.get("verdict") == "off"}
cand = [x for x in json.loads(Path("data/external/real_v1_candidates/manifest_0.json").read_text()) if x["name"] in off]
videos = {v["video_id"]: v for v in lancedb.connect("data/lancedb").open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
targets = [{"card": f"{x['card']} ({x['entity']})", "entities": MULTI.get(x["entity"], [x["entity"]]), "multi": x["card"] in UNITS,
            "video": videos[x["video_id"]]["path"], "layout": videos[x["video_id"]]["layout_id"], "video_id": x["video_id"],
            "match_id": x["match_id"], "times": [round(x["t_video"], 2)]} for x in cand]
targets.sort(key=lambda t: t["card"])
n = write_click_page_times(Path(targets[0]["video"]), targets[0]["layout"], targets, out, targets[0]["video_id"], show_w=440)
out.write_text(out.read_text().replace("Click to cut · Cole recording", "Click · candidates marked box-off")
               .replace('"card": "Cole recording"', '"card": "boxoff"'))
print(n, "frames")
