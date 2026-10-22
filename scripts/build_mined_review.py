"""Review page for failure-mining candidates (2026-10-16): detector boxes mined at never-seen plays
(scripts/mine_failures.py -> LanceDB `mined_candidates`), plus recording "no spell" clips the replay-trained spell
model calls Goblin Curse / Vines (possible unlabelled opponent spells). Keep / reject each; the page exports JSON.

  uv run python scripts/build_mined_review.py [--negs scored.json] -> data/reviews/mined_<date>/review.html
"""
import argparse, datetime, json
from pathlib import Path

import cv2
import lancedb
import numpy as np

from bot77.gallery import write_gallery

ap = argparse.ArgumentParser()
ap.add_argument("--table", default="mined_candidates")
ap.add_argument("--negs", default="", help="score_negs output (recording negatives scored by spell_v2)")
ap.add_argument("--neg-cards", default="Goblin Curse,Vines")
ap.add_argument("--neg-min", type=float, default=0.5)
ap.add_argument("--max", type=int, default=150)
ap.add_argument("--round", default="", help="suffix for the output folder (a second review round)")
ap.add_argument("--skip-reviewed", default="", help="comma list of earlier manifest.json files: leave their items out")
a = ap.parse_args()

out = Path(f"data/reviews/mined_{datetime.date.today()}{a.round}")
seen = {x.get("cand_id") for f in a.skip_reviewed.split(",") if f for x in json.loads(Path(f).read_text())}
(out / "img").mkdir(parents=True, exist_ok=True)
items = []
if a.negs:
    want = set(a.neg_cards.split(","))
    for c in sorted((x for x in json.loads(Path(a.negs).read_text()) if x["top"] in want and x["p"] >= a.neg_min),
                    key=lambda x: -x["p"]):
        strip = cv2.imread(f"data/upload/spell_clips/clips/{c['name']}.jpg")
        fr = strip.reshape(28, strip.shape[0] // 28, strip.shape[1], 3)
        f = out / "img" / f"neg_{c['name']}.jpg"
        cv2.imwrite(str(f), np.hstack([fr[k] for k in (4, 10, 16, 22)]))
        items.append({"file": str(f), "card": c["top"], "t_video": round(c["t"], 1), "match_id": c["match_id"],
                      "entity": c["top"].lower().replace(" ", "-"),
                      "caption": f"Recording clip labelled 'no spell'. Keep = the opponent casts {c['top']} here "
                                 f"(model {c['p']:.2f})."})
db = lancedb.connect("data/lancedb")
rows = db.open_table(a.table).search().limit(10**5).select(
    ["cand_id", "match_id", "t_play", "t_video", "card", "det_cls", "det_conf", "similarity", "image"]).to_arrow().to_pylist()
for r in sorted(rows, key=lambda r: (r["card"], -r["similarity"])):
    if r["cand_id"] in seen:
        continue
    if len(items) >= a.max:
        break
    f = out / "img" / f"{r['cand_id'].replace(' ', '_').replace('.', '-')}.jpg"
    f.write_bytes(r["image"])
    items.append({"file": str(f), "card": r["card"], "t_video": round(r["t_video"], 1), "match_id": r["match_id"],
                  "cand_id": r["cand_id"],
                  "caption": f"Answer key: {r['card']} played at {r['t_play']:.1f}s. Keep = the yellow box is that unit "
                             f"(detector said {r['det_cls']} {r['det_conf']:.2f})."})
(out / "manifest.json").write_text(json.dumps(items, indent=1))
n = write_gallery(out / "manifest.json", out / "review.html",
                  hint="Keep = the boxed unit (or clip) really is the named card. Unsure = reject.")
print(n, "items ->", out / "review.html")
