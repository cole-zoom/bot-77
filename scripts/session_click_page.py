"""Click page for a Cole session recording: every evo/hero play of the declared deck cards (config/recording_decks.json),
3 frames each, capped at 150 frames (decisions.md, 2026-10-09). Group cards ask for each unit to be clicked.
  uv run python scripts/session_click_page.py <video_id>"""
import json, sys
from pathlib import Path
import lancedb
from bot77.autolabel import FORM_CLASS, UNITS, slug
from bot77.clicks import write_click_page_times

vid = sys.argv[1]
deck = json.loads(Path("config/recording_decks.json").read_text())[vid]
db = lancedb.connect("data/lancedb")
v = {x["video_id"]: x for x in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}[vid]
ev = db.open_table("events").search().where(f"video_id = '{vid}' and kind = 'play'").limit(10**5) \
    .select(["match_id", "t_video", "card", "form"]).to_arrow().to_pylist()
tg = []
for e in sorted(ev, key=lambda e: e["t_video"]):
    form = deck.get(e["card"])
    if not form or (form == "evo" and e["form"] != "evo"):
        continue
    ent = FORM_CLASS.get(slug(e["card"], form), slug(e["card"], form))
    note = " — click EACH unit separately" if e["card"] in UNITS else ""
    tg.append({"card": f"{form} {e['card']}{note}", "entities": [ent], "multi": e["card"] in UNITS, "video": v["path"],
               "layout": v["layout_id"], "video_id": vid, "match_id": e["match_id"],
               "times": [round(e["t_video"] + o, 1) for o in (2.5, 4, 5.5)]})
while sum(len(t["times"]) for t in tg) > 150:  # keep the page short: thin the most common card first
    big = max({t["card"] for t in tg}, key=lambda c: sum(t["card"] == c for t in tg))
    tg.remove(next(t for t in reversed(tg) if t["card"] == big))
out = Path(f"data/reviews/click_session_{vid}.html")
n = write_click_page_times(Path(v["path"]), v["layout_id"], tg, out, vid, show_w=440)
out.write_text(out.read_text().replace("Click to cut · Cole recording", f"Click · session {vid}").replace('"card": "Cole recording"', '"card": "session"'))
print(len(tg), "plays,", n, "frames ->", out)
