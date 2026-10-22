"""Detector labels from Cole's review of failure-mining candidates (2026-10-16).

Each kept candidate (scripts/mine_failures.py -> `mined_candidates`, reviewed on the build_mined_review.py page) becomes
one full-frame YOLO label through the same writer as the clock-spot labels (bot77.autolabel.AutoLabeler._write): the
mined box, SAM-tightened, named by the answer key (form too: evo / hero), plus every other confident V4 detection in
that frame so the rest of the frame isn't taught as background.

  uv run python scripts/label_mined.py <review.json> [manifest.json] [out dir]
        -> data/external/real_mined/{images,labels,manifest.json}
"""
import json, sys
from pathlib import Path

import lancedb

from bot77.autolabel import AutoLabeler, slug
from bot77.layout import Layout

review = json.loads(Path(sys.argv[1]).read_text())["slices"]
man_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path(f"data/reviews/mined_{sys.argv[1].rsplit('_', 1)[1].removesuffix('.json')}/manifest.json")
manifest = {Path(x["file"]).name: x for x in json.loads(man_path.read_text())}
keep = [manifest[r["id"]] for r in review if r["verdict"] == "keep" and not r["id"].startswith("neg_")]
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
for f in ("config/v5_check_games.json", "config/replay_test_games.json", "config/heldback_replays.json"):
    hold |= set(json.loads(Path(f).read_text())["matches"])
from bot77.holdout import held_out
hold |= held_out()
assert not any(k["match_id"] in hold for k in keep), "a held-out game leaked"

out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/external/real_mined")
al = AutoLabeler(out=out, dets_table="unit_detections_v5l_e15")
db = lancedb.connect("data/lancedb")
cands = {c["cand_id"]: c for c in db.open_table("mined_candidates").search().limit(10**5).to_arrow().to_pylist()}
rows = []
for k in keep:
    c = cands[k["cand_id"]]
    truth = json.loads(Path(f"data/ground_truth/opponent_truth_replay_{c['match_id']}.json").read_text())["items"]
    it = min((i for i in truth if i["label"] == c["card"]), key=lambda i: abs(i["t_video"] - c["t_play"]))
    entity = slug(c["card"], it.get("form"))
    if entity not in al.cidx:
        entity = slug(c["card"], None)
    v = al.videos[c["video_id"]]
    lay = Layout.load(v["layout_id"])
    old = [d for d in al.dets.search().where(f"match_id = '{c['match_id']}' AND t_video >= {c['t_video'] - 0.3} AND "
                                             f"t_video <= {c['t_video'] + 0.3}").limit(10**4).to_arrow().to_pylist()]
    if old:  # the frame closest to the candidate's
        tt = min({d["t_video"] for d in old}, key=lambda t: abs(t - c["t_video"]))
        old = [d for d in old if d["t_video"] == tt]
    item = {"match_id": c["match_id"], "video_id": c["video_id"], "card": c["card"], "source": "failure-mining"}
    r = al._write(item, v, lay, c["t_video"], (c["x0"], c["y0"], c["x1"], c["y1"]), entity, "track", c["similarity"], old)
    if r:
        rows.append({**r, "cand_id": c["cand_id"], "det_cls": c["det_cls"]})
(out / "manifest.json").write_text(json.dumps(rows, indent=1))
print(len(rows), "labelled frames of", len(keep), "kept;", dict(al.stats))
