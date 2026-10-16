"""Why opponent plays go wrong on the development set (eval_dev.py must have run for <tag>).

For each true play: matched right / wrong card / missed; for a miss, did the detector see that card at all near the
play (enemy or any team, +-3 s), and how far off in time was the nearest event? For each fake event: the card, its
units, whether a POV (own) play of the same card is near (our unit called enemy), whether a true play of the same
card is a bit further than the window (timing), or a same-card true play is long before (re-detection / walking unit).
  uv run python scripts/analyze_dev.py <det tag> [events tag]
"""
import collections, glob, json, sys
from pathlib import Path

import lancedb

from bot77.detect.names import class_to_card
from bot77.review import MATCH_WINDOW_S

tag = sys.argv[1]
ev_tag = sys.argv[2] if len(sys.argv) > 2 else tag  # eval_dev.py --out suffix
db = lancedb.connect("data/lancedb")
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
names = [c["name"] for c in db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name"]).to_arrow().to_pylist()]
ev_t = db.open_table(f"opponent_events_dev_{ev_tag}")
det_t = db.open_table(f"unit_detections_{tag}")
miss_why, fake_why, wrong_pairs, offsets = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(list)
miss_cards, fake_cards = collections.Counter(), collections.Counter()
for f in sorted(glob.glob("data/ground_truth/opponent_truth_auto_*.json")):
    t = json.loads(Path(f).read_text()); m = t["matches"][0]
    if m in hold:
        continue
    items = t["items"]
    evs = ev_t.search().where(f"match_id = '{m}'").limit(10**5).select(["t_video", "card", "units", "classes"]).to_arrow().to_pylist()
    dets = det_t.search().where(f"match_id = '{m}'").limit(10**7).select(["t_video", "cls", "team", "conf"]).to_arrow().to_pylist()
    pov = db.open_table("events").search().where(f"match_id = '{m}'").limit(10**5).select(["t_video", "card"]).to_arrow().to_pylist()
    dcard = [(d["t_video"], class_to_card(d["cls"], names)[0], d["team"]) for d in dets if d["conf"] >= 0.5]
    # same one-to-one matching as score_opponent
    used_t, used_e, match = set(), set(), {}
    for exact in (True, False):
        for i, e in sorted(enumerate(evs), key=lambda ie: ie[1]["t_video"]):
            if i in used_e:
                continue
            c = [(abs(x["t_video"] - e["t_video"]), k) for k, x in enumerate(items) if k not in used_t
                 and abs(x["t_video"] - e["t_video"]) <= MATCH_WINDOW_S and (x["label"] == e["card"]) == exact]
            if c:
                _, k = min(c); used_t.add(k); used_e.add(i); match[k] = (i, exact)
    for k, x in enumerate(items):
        lab = x["label"]
        if k in match:
            i, exact = match[k]
            offsets[lab].append(evs[i]["t_video"] - x["t_video"])
            if not exact:
                wrong_pairs[(lab, evs[i]["card"])] += 1
            continue
        miss_cards[lab] += 1
        if lab in ("__ability",):
            miss_why["hero ability (never output)"] += 1; continue
        if lab == "Mirror":
            miss_why["Mirror (no visual)"] += 1; continue
        seen_enemy = any(c == lab and tm == "enemy" and -1 <= dt - x["t_video"] <= 4 for dt, c, tm in dcard)
        seen_any = any(c == lab and -1 <= dt - x["t_video"] <= 4 for dt, c, tm in dcard)
        near_ev = min((abs(e["t_video"] - x["t_video"]) for e in evs), default=99)
        same_ev = [e["t_video"] - x["t_video"] for e in evs if e["card"] == lab]
        late = min((d for d in same_ev if 0 < d <= 6), default=None)
        if late is not None:
            miss_why["same-card event but outside 1.5 s (timing)"] += 1
        elif seen_enemy:
            miss_why["detector saw it (enemy) but no event"] += 1
        elif seen_any:
            miss_why["detector saw it, wrong team only"] += 1
        else:
            miss_why["detector never saw this card"] += 1
    for i, e in enumerate(evs):
        if i in used_e:
            continue
        fake_cards[e["card"]] += 1
        if any(p["card"] == e["card"] and -1.5 <= e["t_video"] - p["t_video"] <= 6 for p in pov):
            fake_why["own (POV) unit of the same card"] += 1
        elif any(x["label"] == e["card"] and 1.5 < e["t_video"] - x["t_video"] <= 6 for x in items):
            fake_why["late duplicate of a true play (timing / re-birth)"] += 1
        elif any(x["label"] == e["card"] and 6 < e["t_video"] - x["t_video"] <= 40 for x in items):
            fake_why["same card played 6-40 s before (walking / resurfacing / spawn)"] += 1
        elif e["card"] is None:
            fake_why["no deck card"] += 1
        else:
            fake_why["card not played around then (misdetection)"] += 1
print("MISSED by reason:", dict(miss_why.most_common()))
print("FAKE by reason:", dict(fake_why.most_common()))
print("missed cards:", miss_cards.most_common(25))
print("fake cards:", fake_cards.most_common(25))
print("wrong card (truth -> called):", wrong_pairs.most_common(20))
import statistics
print("timing (event - truth, s), median per card:",
      {c: round(statistics.median(v), 2) for c, v in sorted(offsets.items(), key=lambda kv: -len(kv[1])) if len(v) >= 3})
