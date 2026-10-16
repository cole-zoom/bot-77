"""Add plays the POV detector missed to the existing paired / replay opponent truth (bot77.pairing.next_card_plays,
decisions.md 2026-10-15). Only appends items tagged "next-card advance"; rerunning replaces those, nothing else
in the files changes.

  uv run python scripts/add_next_card_plays.py
"""
import glob, json
from pathlib import Path

import lancedb

from bot77.pairing import next_card_plays


def main():
    db = lancedb.connect("data/lancedb")
    names = {c["id"]: c["name"] for c in db.open_table("cards").search().limit(5000).select(["id", "name"]).to_arrow().to_pylist()}
    ev = db.open_table("events")
    added = files = 0
    for f in sorted(glob.glob("data/ground_truth/opponent_truth_replay_*.json") + glob.glob("data/ground_truth/opponent_truth_auto_*.json")):
        T = json.loads(Path(f).read_text())
        src, target = T.get("paired_with"), T["matches"][0]
        if not src:
            continue
        sign = 1 if "offset_s" in T else 0
        plays = ev.search().where(f"match_id = '{src}' AND kind = 'play'").limit(10**5).select(["t_video", "card"]).to_arrow().to_pylist()
        new = [{"match_id": target, "t_video": round(x["t_video"] - sign * T.get("offset_s", 0.0), 2), "label": x["card"],
                "form": None, "source": f"next-card advance in {src}", "source_confidence": "medium"}
               for x in next_card_plays(db, src, plays, names)]
        items = [i for i in T["items"] if not str(i.get("source", "")).startswith("next-card advance")]
        T["items"] = sorted(items + new, key=lambda i: i["t_video"])
        Path(f).write_text(json.dumps(T, indent=1))
        added += len(new); files += 1
    print(f"{files} answer keys, {added} plays added")


if __name__ == "__main__":
    main()
