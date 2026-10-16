#!/bin/zsh
# Session F: process, then a click page with EVERY Electro Giant play (Cole picks evo or normal), then recount.
cd "$(dirname "$0")/.."
scripts/after_session.sh cole_2026-10-09_f data/logs/run_all_jobs_c1009f.json
uv run python - <<'PY' > data/logs/session_page_f_all.log 2>&1
import lancedb
from pathlib import Path
from bot77.clicks import write_click_page_times
db = lancedb.connect("data/lancedb")
v = {x["video_id"]: x for x in db.open_table("videos").search().limit(10000).to_arrow().to_pylist()}["cole_2026-10-09_f"]
ev = db.open_table("events").search().where("video_id = 'cole_2026-10-09_f' and kind = 'play' and card = 'Electro Giant'").limit(1000) \
    .select(["match_id", "t_video"]).to_arrow().to_pylist()
tg = [{"card": "Electro Giant — pick evo or normal", "entities": ["electro-giant-evolution", "electro-giant"], "multi": False,
       "video": v["path"], "layout": v["layout_id"], "video_id": v["video_id"], "match_id": e["match_id"],
       "times": [round(e["t_video"] + o, 1) for o in (2.5, 4, 5.5)]} for e in sorted(ev, key=lambda e: e["t_video"])]
out = Path("data/reviews/click_session_cole_2026-10-09_f.html")
n = write_click_page_times(Path(v["path"]), v["layout_id"], tg, out, v["video_id"], show_w=440)
out.write_text(out.read_text().replace("Click to cut · Cole recording", "Click · session F: Electro Giant, evo or normal").replace('"card": "Cole recording"', '"card": "sessionF"'))
print(len(tg), "plays,", n, "frames")
PY
echo "SESSION_F_PAGE_DONE $(date +%H:%M)" >> data/logs/run_all.log
