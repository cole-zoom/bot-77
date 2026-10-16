"""Label the OPPONENT's spells in paired matches (decisions.md, 2026-10-11). Spells are team-coloured, and every
spell label so far came from POV plays (our blue spells), so the detector never saw a red Goblin Curse and reads it
as clone/poison. Mirrored spell spots were 0/9 right (round 1), so the spot is not used: the paired truth gives the
card and the time, and the detector's NEW spell-class boxes just after it (whatever class it called them) are
relabelled as the true spell. Skipped when we cast a spell ourselves near that time. Benchmark excluded.

  uv run python scripts/autolabel_paired_spells.py <dets table> <out dir>
"""
import collections, json, random, sys
from pathlib import Path

import lancedb

from bot77.autolabel import ROLLING, SPAWNING_SPELLS, AutoLabeler, collect_items
from bot77.detect.names import class_to_card
from bot77.layout import Layout

table, out = sys.argv[1], Path(sys.argv[2])
WIN = (0.3, 2.5)  # spell effects show from ~0.3 s after the play (Vines grows, Curse spreads)
MIN_CONF, POV_GAP_S, FRAMES = 0.3, 3.0, 2
# Which called classes can be the true spell, by look (spot check 2026-10-11: unconstrained, ~1 in 4 boxes was another
# spell or a unit). Curse is a green-yellow circle like poison/clone; Void is already detected as void.
LOOKS_LIKE = {"goblin-curse": {"goblin-curse", "poison", "clone", "rage"}, "void": {"void"},
              "vines": {"vines", "poison", "clone"}}
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
al = AutoLabeler(out=out, dets_table=table)
db = lancedb.connect("data/lancedb")
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
names = [c["name"] for c in cards]; ctype = {c["name"]: c["type"] for c in cards}
spell_cls = lambda c: ctype.get(class_to_card(c, names)[0]) == "spell" and class_to_card(c, names)[0] not in SPAWNING_SPELLS
items = [i for i in collect_items() if i["match_id"] not in hold and i["source"] == "paired"
         and ctype.get(i["card"]) == "spell" and i["card"] not in SPAWNING_SPELLS and i["entity"] not in ROLLING]
stats, rows = collections.Counter(), []
rng = random.Random(0)
for it in items:
    t0 = it["t_video"]
    pov = db.open_table("events").search().where(
        f"match_id = '{it['match_id']}' AND t_video >= {t0 - POV_GAP_S} AND t_video <= {t0 + POV_GAP_S}").limit(1000) \
        .select(["card"]).to_arrow().to_pylist()
    if any(ctype.get(p["card"]) == "spell" for p in pov if p["card"]):
        stats["skip_our_spell_near"] += 1
        continue
    dets = al.dets.search().where(f"match_id = '{it['match_id']}' AND t_video >= {t0 - 1} AND t_video <= {t0 + WIN[1]}") \
        .limit(10**6).to_arrow().to_pylist()
    before = [d for d in dets if d["t_video"] < t0 and spell_cls(d["cls"]) and d["conf"] >= MIN_CONF]
    cx = lambda d: ((d["x0"] + d["x1"]) / 2, (d["y0"] + d["y1"]) / 2)
    ok = LOOKS_LIKE.get(it["entity"])
    new = [d for d in dets if WIN[0] <= d["t_video"] - t0 <= WIN[1] and spell_cls(d["cls"]) and d["conf"] >= MIN_CONF
           and (ok is None or d["cls"] in ok)
           and not any(abs(cx(d)[0] - cx(b)[0]) + abs(cx(d)[1] - cx(b)[1]) < 40 for b in before)]
    if not new:
        stats["no_new_spell_box"] += 1
        continue
    # one spell: the cluster of new boxes around the most confident one
    top = max(new, key=lambda d: d["conf"])
    clus = [d for d in new if abs(cx(d)[0] - cx(top)[0]) + abs(cx(d)[1] - cx(top)[1]) < 80]
    by_t = collections.defaultdict(list)
    for d in dets:
        by_t[round(d["t_video"], 3)].append(d)
    picked, last = [], -9
    for d in sorted(clus, key=lambda d: d["t_video"]):
        if d["t_video"] - last >= 0.5 and len(picked) < FRAMES:
            picked.append(d); last = d["t_video"]
    v = al.videos[it["video_id"]]
    lay = Layout.load(v["layout_id"])
    for d in picked:
        r = al._write(it, v, lay, d["t_video"], (d["x0"], d["y0"], d["x1"], d["y1"]), it["entity"], "spell-relabel", None,
                      by_t[round(d["t_video"], 3)])
        if r:
            r["called_as"] = d["cls"]
            rows.append(r)
    stats["plays_labelled"] += 1
(out / "manifest.json").write_text(json.dumps(rows, indent=1))
print(dict(stats), "frames", len(rows), "called as:", collections.Counter(r["called_as"] for r in rows).most_common(8))
