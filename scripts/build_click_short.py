"""Step 2 click page: frames from not-yet-clicked plays of every entity still under the 40 bar."""
import json, glob, math, collections
from pathlib import Path
import lancedb
from bot77.autolabel import FORM_CLASS, UNITS, slug, spirit_empress_form
from bot77.clicks import write_click_page_times

db = lancedb.connect("data/lancedb")
videos = {v["video_id"]: v for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
ev = db.open_table("events").search().limit(10**7).select(["video_id", "match_id", "t_video", "kind", "card", "form", "confidence", "measured_cost"]).to_arrow().to_pylist()
clicked = collections.defaultdict(set)
for f in glob.glob("data/ground_truth/reviews/clicks_*.json"):
    for c in json.load(open(f))["clicks"]:
        clicked[c["entity"]].add((c["video_id"], round(c["t"])))
need = {l.split()[0]: int(l.split()[-1]) for l in open(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "data/logs/coverage_after_clicks.log") if "need" in l}
ALIAS = {"spirit-empress-ground": "spirit-empress", "tomb-queen": "tombstone-hero"}
ENTS = {"tomb-queen": ["tomb-queen", "tombstone-hero"],
        "dark-prince-hero": ["dark-prince-hero", "dark-prince-hero-dismounted"]}
# Goblin Drill (Cole, 2026-10-01): it reaches the tower about 4 s after placement (at +3 s every frame showed it
# halfway there), and the evo then dives every ~2 s and pops back up 2 more times
# evo Giant Snowball (Cole, 2026-10-01): it flies from the king tower, lands and rolls for a second or two; when
# depends on how far from the king tower it was placed, so sample densely across the window
OFFS = {"tomb-queen": (5, 8, 11, 14), "spirit-empress-ground": (1.5, 3, 5, 7), "goblin-drill-evolution": (4, 5, 7, 9),
        "giant-snowball-evolution": (1.5, 2.25, 3, 3.75),
        # evo Zap (Cole, 2026-10-08): truer blue than Zap and zaps twice, the second about a second after the first
        "zap-evolution": (0.3, 1.0, 1.3)}
DEDICATED = {"cole_2026-09-30", "cole_2026-10-01", "cole_2026-10-01_b", "cole_2026-10-01_c", "cole_2026-10-01_d", "cole_2026-10-01_e", "cole_2026-10-06", "cole_2026-10-07", "cole_2026-10-09", "cole_2026-10-09_c", "cole_2026-10-09_d", "cole_2026-10-09_e", "cole_2026-10-09_f", "cole_2026-10-09_g"}
MAX_FRAMES = 150  # Cole, 2026-10-05: keep review pages short
targets, plan = [], {}
for e, n in need.items():
    if e.startswith("spirit-empress-"):  # form comes from the measured cost (3 ground, 6 air)
        plays = [x for x in ev if x["kind"] == "play" and x["card"] == "Spirit Empress" and spirit_empress_form(x) == e]
    else:
        plays = [x for x in ev if x["kind"] == "play" and x["card"] and x["confidence"] == "high" and FORM_CLASS.get(slug(x["card"], x["form"]), slug(x["card"], x["form"])) == ALIAS.get(e, e)]
    if e == "ice-wizard-hero":  # the hand reader often misses the hero frame on Ice Wizard; hero videos only
        plays += [x for x in ev if x["kind"] == "play" and x["card"] == "Ice Wizard" and x["video_id"] in ("5BjmAo2-b2k", "7wwlikDO7J0", "-ErsZhzGemc")]
    # Cole's dedicated recordings first (one card alone in open space), latest plays first: by the end of a
    # session he leaves units walking alone for longer (Cole, 2026-10-01)
    plays.sort(key=lambda x: (x["video_id"] not in DEDICATED, -x["t_video"] if x["video_id"] in DEDICATED else 0))
    plays = [p for p in plays if not any(v == p["video_id"] and abs(t - p["t_video"]) < 6 for v, t in clicked[e])]
    if not plays:
        continue
    per = min(4, max(2, math.ceil(n * 1.3 / len(plays))))
    offs = OFFS.get(e, (2.5, 4, 5.5, 7))[:per]
    for p in plays[: math.ceil(n * 1.3 / per) + 2]:
        v = videos[p["video_id"]]
        targets.append({"card": f"{p['card']} ({e})" + (" — click EACH unit separately" if p["card"] in UNITS else ""), "entities": ENTS.get(e, [e]), "multi": p["card"] in UNITS, "video": v["path"], "layout": v["layout_id"],
                        "video_id": p["video_id"], "match_id": p["match_id"], "times": [round(p["t_video"] + o, 1) for o in offs]})
    plan[e] = {"need": n, "plays": len(plays), "frames": per * min(len(plays), math.ceil(n * 1.3 / per) + 2)}
targets = targets  # trimmed below
while sum(len(t['times']) for t in targets) > MAX_FRAMES:
    big = max({t['card'] for t in targets}, key=lambda c: sum(1 for t in targets if t['card'] == c))
    targets.remove(next(t for t in reversed(targets) if t['card'] == big))
n = write_click_page_times(Path(targets[0]["video"]), targets[0]["layout"], targets, Path(__import__("sys").argv[2] if len(__import__("sys").argv) > 2 else "data/reviews/click_short.html"), targets[0]["video_id"], show_w=440)
p = Path(__import__("sys").argv[2] if len(__import__("sys").argv) > 2 else "data/reviews/click_short.html"); p.write_text(p.read_text().replace("Click to cut · Cole recording", "Click · cards still under 40").replace('"card": "Cole recording"', '"card": "short"'))
print(n, "frames"); print(json.dumps(plan, indent=1))
