#!/bin/zsh
# Final session: evo Electro Giant + evo Skeleton Army. Electro Giant: every play, Cole picks evo/normal (the reader can't
# tell greyed normal from evo); Skeleton Army: evo plays (click the General).
cd "$(dirname "$0")/.."
scripts/after_session.sh cole_2026-10-09_g data/logs/run_all_jobs_c1009g.json
uv run python - <<'PY' > data/logs/session_page_g_all.log 2>&1
import lancedb
from pathlib import Path
from bot77.clicks import write_click_page_times
db = lancedb.connect("data/lancedb")
v = {x["video_id"]: x for x in db.open_table("videos").search().limit(10000).to_arrow().to_pylist()}["cole_2026-10-09_g"]
ev = db.open_table("events").search().where("video_id = 'cole_2026-10-09_g' and kind = 'play' and card in ('Electro Giant','Skeleton Army')") \
    .limit(1000).select(["match_id", "t_video", "card", "form"]).to_arrow().to_pylist()
tg = []
for e in sorted(ev, key=lambda e: e["t_video"]):
    if e["card"] == "Electro Giant":
        tg.append({"card": "Electro Giant — pick evo or normal", "entities": ["electro-giant-evolution", "electro-giant"]})
    elif e["form"] == "evo":
        tg.append({"card": "evo Skeleton Army — click the General", "entities": ["skeleton-army-evolution"]})
    else:
        continue
    tg[-1].update({"multi": False, "video": v["path"], "layout": v["layout_id"], "video_id": v["video_id"], "match_id": e["match_id"],
                   "times": [round(e["t_video"] + o, 1) for o in (2.5, 4, 5.5)]})
while sum(len(t["times"]) for t in tg) > 150:
    big = max({t["card"] for t in tg}, key=lambda c: sum(t["card"] == c for t in tg)); tg.remove(next(t for t in reversed(tg) if t["card"] == big))
out = Path("data/reviews/click_session_cole_2026-10-09_g.html")
n = write_click_page_times(Path(v["path"]), v["layout_id"], tg, out, v["video_id"], show_w=440)
out.write_text(out.read_text().replace("Click to cut · Cole recording", "Click · final session").replace('"card": "Cole recording"', '"card": "sessionG"'))
print(len(tg), "plays,", n, "frames")
PY
echo "SESSION_G_PAGE_DONE $(date +%H:%M)" >> data/logs/run_all.log
