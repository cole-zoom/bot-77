"""After the batch: pair the new dual-POV recordings, then build click pages (one per group) with
frames 1.5 s and 3 s after every labelled POV play of a card or form we still need slices for.
Auto-cut from the old detector's tracks was tried first and wasn't reliable enough (decisions.md)."""
import collections
import json
from pathlib import Path

import lancedb

from bot77.clicks import write_click_page_times
from bot77.pairing import write_pair_truth

DB = Path("data/lancedb")
db = lancedb.connect(DB)
NEW = {"Berserker", "Boss Bandit", "Goblin Curse", "Goblin Demolisher", "Goblin Machine", "Suspicious Bush", "Vines",
       "Void", "Ronin", "Rune Giant", "Spirit Empress", "Minion Giant", "Goblinstein"}
DUAL = {"cole_2026-09-29_b", "cole_2026-09-29_c", "friend_2026-09-29_a", "friend_2026-09-29_b", "friend_2026-09-29_c"}
MULTI = {"Goblinstein": ["goblinstein-doctor", "goblinstein-monster"],
         "Spirit Empress": ["spirit-empress-ground", "spirit-empress-air"],
         "Tombstone": ["tombstone-hero", "tomb-queen"],
         "Dark Prince": ["dark-prince-hero", "dark-prince-hero-dismounted"]}
PLAYS = {"new": 8, "evo": 6, "hero": 6}  # plays per (card, form), 2 frames each
OFFSETS = (1.5, 3.0)

write_pair_truth("cole_2026-09-29_c", "friend_2026-09-29_c", 67)  # Cole 6:11:20 PM, friend 6:10:13 PM

videos = {v["video_id"]: v for v in db.open_table("videos").search().limit(10_000).to_arrow().to_pylist()}
events = db.open_table("events").search().limit(10**7).select(
    ["video_id", "match_id", "t_video", "kind", "card", "form", "confidence"]).to_arrow().to_pylist()


def slug(card, form):
    base = card.lower().replace(" ", "-").replace(".", "")
    return base if form in (None, "normal") else f"{base}-{'evolution' if form == 'evo' else 'hero'}"


groups = collections.defaultdict(lambda: collections.defaultdict(list))
for e in events:
    if e["kind"] != "play" or e["confidence"] != "high" or not e["card"]:
        continue
    if e["form"] in ("evo", "hero"):
        groups[e["form"]][(e["card"], e["form"])].append(e)
    elif e["card"] in NEW and e["video_id"] in DUAL:
        groups["new"][(e["card"], "normal")].append(e)

summary = {}
for g, by in groups.items():
    targets = []
    for (card, form), plays in sorted(by.items()):
        plays.sort(key=lambda e: (e["video_id"], e["t_video"]))
        step = max(1, len(plays) // PLAYS[g])
        for e in plays[::step][:PLAYS[g]]:
            ents = MULTI.get(card, [slug(card, form)]) if form in (None, "normal") or card in ("Tombstone", "Dark Prince") else [slug(card, form)]
            v = videos[e["video_id"]]
            targets.append({"card": f"{card}{'' if form in (None, 'normal') else ' (' + form + ')'}", "entities": ents,
                            "video": v["path"], "layout": v["layout_id"], "video_id": e["video_id"], "match_id": e["match_id"],
                            "times": [round(e["t_video"] + o, 1) for o in OFFSETS]})
        summary.setdefault(g, {})[f"{card} ({form})"] = {"plays_available": len(plays), "on_page": min(len(plays), PLAYS[g])}
    n = write_click_page_times(Path(targets[0]["video"]), targets[0]["layout"], targets, Path(f"data/reviews/click_{g}.html"),
                               targets[0]["video_id"], show_w=440)
    page = Path(f"data/reviews/click_{g}.html")
    page.write_text(page.read_text().replace("Click to cut · Cole recording", f"Click to cut · {g} cards")
                    .replace('"card": "Cole recording"', f'"card": "{g}"'))
    summary[g]["_frames"] = n
Path("data/logs/review_queue_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
