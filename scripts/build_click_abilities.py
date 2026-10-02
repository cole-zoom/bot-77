"""Click page for hero forms that only appear after the ability (hero Dark Prince dismounted). Ability events
the reader didn't attribute to a card are matched to the last hero played in the same match (Cole, 2026-10-01:
"in my own footage we should have a lot of it"). Frames shortly after each ability use."""
import json, sys
from pathlib import Path
import lancedb
from bot77.clicks import write_click_page_times

HERO, ENTS, OFFS = "Dark Prince", ["dark-prince-hero-dismounted", "dark-prince-hero"], (0.5, 1.5, 3.0, 5.0)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/reviews/click_abilities.html")
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None  # restrict to these videos
db = lancedb.connect("data/lancedb")
videos = {v["video_id"]: v for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
ev = db.open_table("events").search().where("kind in ('play', 'ability')").limit(10**7) \
    .select(["video_id", "match_id", "t_video", "kind", "card", "form"]).to_arrow().to_pylist()
ev.sort(key=lambda e: (e["video_id"], e["t_video"]))
targets, last = [], {}
for e in ev:
    if e["kind"] == "play" and e["form"] == "hero":
        last[e["match_id"]] = e
    elif e["kind"] == "ability":
        h = last.get(e["match_id"])
        card = e["card"] or (h["card"] if h and e["t_video"] - h["t_video"] < 60 else None)
        if card == HERO and (ONLY is None or e["video_id"] in ONLY):
            v = videos[e["video_id"]]
            targets.append({"card": f"{HERO} (hero, ability)", "entities": ENTS, "multi": False, "video": v["path"],
                            "layout": v["layout_id"], "video_id": e["video_id"], "match_id": e["match_id"],
                            "times": [round(e["t_video"] + o, 1) for o in OFFS]})
n = write_click_page_times(Path(targets[0]["video"]), targets[0]["layout"], targets, out, targets[0]["video_id"], show_w=440)
out.write_text(out.read_text().replace("Click to cut · Cole recording", "Click · hero Dark Prince after ability")
               .replace('"card": "Cole recording"', '"card": "abilities"'))
print(len(targets), "ability uses,", n, "frames")
