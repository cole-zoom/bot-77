"""Tune the play decoder on the development set with a split: fit on one half of the paired matches, check on the
other (cole_* vs friend_* recordings), so a setting only counts if it carries over. Benchmark matches are never loaded.
Candidates (deploys after the v2 play rules) are computed once and kept in memory; decoding + scoring is fast.

  uv run python scripts/tune_decoder.py <det tag> [--rules '{}']
"""
import argparse, copy, glob, itertools, json, random
from pathlib import Path

import lancedb

from bot77.decode import DecodeParams, decode
from bot77.layout import Layout
from bot77.playrules import apply_rules
from bot77.review import MATCH_WINDOW_S
from bot77.spawns import deploys, drop_pov_units, infer_deck, resolve_composites

ap = argparse.ArgumentParser()
ap.add_argument("tag"); ap.add_argument("--rules", default="{}")
a = ap.parse_args()
db = lancedb.connect("data/lancedb")
hold = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).select(["name", "type"]).to_arrow().to_pylist()
names = [c["name"] for c in cards]; ctype = {c["name"]: c["type"] for c in cards}
is_spell = lambda c: ctype.get(c) == "spell"
data = {}
for f in sorted(glob.glob("data/ground_truth/opponent_truth_auto_*.json")):
    t = json.loads(Path(f).read_text()); m = t["matches"][0]
    if m in hold:
        continue
    vid = m.rsplit("_m", 1)[0]
    video = db.open_table("videos").search().where(f"video_id = '{vid}'").to_arrow().to_pylist()[0]
    lay = Layout.load(video["layout_id"]); ax0, ay0, ax1, ay1 = lay.katacr_arena; tile = (ax1 - ax0) / 18
    dets = db.open_table(f"unit_detections_{a.tag}").search().where(f"match_id = '{m}'").limit(10**7).to_arrow().to_pylist()
    ev = deploys(dets, names, tile, lambda c: ctype.get(c) == "building", is_spell, (ax0, ay0))
    pov = [(p["t_video"], p["card"]) for p in db.open_table("events").search().where(f"match_id = '{m}'").limit(10**5)
           .select(["t_video", "card"]).to_arrow().to_pylist() if p["card"]]
    enemy, _ = drop_pov_units(resolve_composites([e for e in ev if e.team == "enemy"]), pov)
    enemy = apply_rules(enemy, pov, ctype, json.loads(a.rules))
    data[m] = (enemy, t["items"])
print("matches", len(data), {k: len(v[1]) for k, v in data.items()})


def score(events, items):
    used_t, used_e, right, found = set(), set(), 0, 0
    for exact in (True, False):
        for i, e in sorted(enumerate(events), key=lambda ie: ie[1].t):
            if i in used_e:
                continue
            c = [(abs(x["t_video"] - e.t), k) for k, x in enumerate(items) if k not in used_t
                 and abs(x["t_video"] - e.t) <= MATCH_WINDOW_S and (x["label"] == e.card) == exact]
            if c:
                _, k = min(c); used_t.add(k); used_e.add(i); found += 1; right += exact
    return right, found, len(events) - found, len(items)


def run(p, ms, use_decoder=True):
    R = F = X = N = 0
    for m in ms:
        enemy, items = data[m]
        ev = copy.deepcopy(enemy)
        abil = [e for e in ev if e.card == "__ability"]
        rest = [e for e in ev if e.card != "__ability"]
        deck = infer_deck(rest)
        if use_decoder:
            rest = decode(rest, deck, is_spell, p, mirror_in_deck=True)
        else:
            from bot77.spawns import apply_deck
            apply_deck(rest, set(deck))
        r, f, x, n = score(sorted(rest + abil, key=lambda e: e.t), items)
        R += r; F += f; X += x; N += n
    return R / N, X / len(ms), R, N


A = [m for m in data if m.startswith("cole")]; B = [m for m in data if m.startswith("friend")]
print("no decoder  A %.3f fake/m %.1f | B %.3f fake/m %.1f" % (*run(None, A, False)[:2], *run(None, B, False)[:2]))
grid = {"keep_bias": [-1, 0, 1, 2, 3, 4], "cycle_penalty": [0, 1.5, 3], "mirror_prior": [-3, -1, 0.5],
        "keep_off_deck": [-1.5, -4], "keep_per_det": [0.04, 0.08, 0.15]}
best = []
for vals in itertools.product(*grid.values()):
    p = DecodeParams(**dict(zip(grid, vals)))
    ra, fa, _, _ = run(p, A)
    best.append((ra, fa, vals))
best.sort(key=lambda t: -t[0])
print("top on A (fit half) and their B (check half):")
for ra, fa, vals in best[:8]:
    p = DecodeParams(**dict(zip(grid, vals)))
    rb, fb, _, _ = run(p, B)
    print(f"  A {ra:.3f} (fake/m {fa:.1f})  B {rb:.3f} (fake/m {fb:.1f})  {dict(zip(grid, vals))}")
