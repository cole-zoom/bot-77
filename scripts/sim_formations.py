"""Squad formations at deploy: the engine's spawn offsets vs how the bodies sit at their first detections in the
replays (docs/simulator-validation.md flag 3, 2026-10-24). Prints, per squad card, the bodies' spread around their
centre (std in x and y, our units) in both.

  uv run python scripts/sim_formations.py
"""
import collections

import clash_royale_env as cre
import numpy as np

from bot77.sim import cards as sc
from bot77.sim import store
from bot77.sim.geometry import Y_SCALE

SQUADS = ["Skeletons", "Goblins", "Spear Goblins", "Rascals", "Royal Recruits", "Archers", "Barbarians", "Minions",
          "Guards", "Skeleton Army", "Minion Horde", "Bats", "Goblin Gang", "Royal Hogs", "Wall Breakers",
          "Elite Barbarians", "Three Musketeers", "Skeleton Dragons", "Firecracker"]
games = [m["match_id"] for m in store.read("matches", "source IN ('replay_tv', 'replay_bl') AND pov_side = 'bottom' "
                                           "AND split = 'train'")][:200]
real = collections.defaultdict(list)
for g in games:
    plays = [p for p in store.read("plays", f"game_id = '{g}' AND source = 'hand' AND kind = 'play'")
             if p["card"] in SQUADS]
    if not plays:
        continue
    dets = store.read("detections", f"match_id = '{g}' AND run_id = 'v5l_e15'")
    by_t = collections.defaultdict(list)
    for d in dets:
        by_t[round(d["t"], 2)].append(d)
    ts = sorted(by_t)
    for p in plays:
        team = "friendly" if p["side"] == "bottom" else "enemy"
        n_exp = cre.get_card_stats(sc.engine_id(p["card"]))["bodies"]
        # the first frame in the 1.5 s after the play showing the most bodies of this card (deploying: not moved yet)
        best = None
        for t in ts:
            if p["t"] <= t <= p["t"] + 1.0:
                pts = [(d["x"], d["y"]) for d in by_t[t] if d["card"] == p["card"] and d["team"] == team]
                if pts and (best is None or len(pts) > len(best)):
                    best = pts
        if best and len(best) >= max(2, n_exp - 1):
            a = np.array(best)
            # keep the bodies near the median (another copy of the card can be on the board)
            a = a[np.hypot(*(a - np.median(a, axis=0)).T) < 3.5]
            if len(a) >= 2:
                real[p["card"]].append(a.std(axis=0))
print(f"{'card':18s} {'bodies':>6s} | engine std x, y | replay std x, y (median over plays, n)")
for c in SQUADS:
    e = sc.engine_id(c)
    if e is None:
        continue
    offs = np.array(cre.get_card_stats(e)["spawn_offsets"], float)
    eng = (offs[:, 0].std(), offs[:, 1].std() * Y_SCALE)
    r = np.median(real[c], axis=0) if real[c] else (np.nan, np.nan)
    print(f"{c:18s} {len(offs):6d} | {eng[0]:5.2f} {eng[1]:5.2f}    | {r[0]:5.2f} {r[1]:5.2f}  ({len(real[c])})")

# --- the fit: cards with >= MIN_PLAYS replay plays get their offsets scaled to the replay's spread -------------------
MIN_PLAYS = 20
OUT = "third_party/clash-royale-env/include/core/SpawnFormations.h"
fits = []
for c in SQUADS:
    e = sc.engine_id(c)
    if e is None or len(real[c]) < MIN_PLAYS:
        continue
    offs = np.array(cre.get_card_stats(e)["spawn_offsets"], float)
    if len(offs) < 2:  # Rascals: the girls come from a deploy effect, not these offsets (left as is)
        continue
    rx, ry = np.median(real[c], axis=0)
    ex, ey = offs[:, 0].std(), offs[:, 1].std()
    kx = rx / ex if ex > 1e-3 else 1.0
    ky = (ry / Y_SCALE) / ey if ey > 1e-3 else 1.0
    stagger = 0.0
    if ey <= 1e-3 and ry > 0.15:  # a flat line in the engine, depth in the game: alternate bodies forward / back
        pat = np.array([1.0 if i % 2 else -1.0 for i in range(len(offs))])
        pat -= pat.mean()
        stagger = (ry / Y_SCALE) / pat.std()
    if abs(kx - 1) < 0.15 and abs(ky - 1) < 0.15 and not stagger:
        continue
    fits.append((e, c, kx, ky, stagger, len(real[c])))
lines = ["#pragma once", "#include <algorithm>", "#include <vector>", "#include \"Entity.h\"", "",
         "// Squad formations at deploy, fitted to the replays (bot-77 scripts/sim_formations.py: the bodies' spread at",
         "// their first detections, 200 train games). Scales a card's spawn offsets (x, y) and, where the engine had a flat",
         "// line but the game has depth, staggers alternate bodies (x order) forward / back by `stagger` engine rows.",
         "struct FormationFit { float kx, ky, stagger; };", "",
         "inline FormationFit formationFitForCard(int cardId) {", "    switch (cardId) {"]
lines += [f"        case {e}: return {{{kx:.3f}f, {ky:.3f}f, {st:.3f}f}};  // {c} (n {n})" for e, c, kx, ky, st, n in fits]
lines += ["        default: return {1.0f, 1.0f, 0.0f};", "    }", "}", "",
          "inline std::vector<Vector2D> fittedOffsets(int cardId, std::vector<Vector2D> offs) {",
          "    const FormationFit f = formationFitForCard(cardId);",
          "    for (auto& o : offs) { o.x *= f.kx; o.y *= f.ky; }",
          "    if (f.stagger != 0.0f && offs.size() > 1) {",
          "        std::vector<size_t> order(offs.size());",
          "        for (size_t i = 0; i < order.size(); ++i) order[i] = i;",
          "        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return offs[a].x < offs[b].x; });",
          "        float mean = 0.0f;",
          "        for (size_t r = 0; r < order.size(); ++r) mean += (r % 2) ? 1.0f : -1.0f;",
          "        mean /= static_cast<float>(order.size());",
          "        for (size_t r = 0; r < order.size(); ++r) offs[order[r]].y += f.stagger * (((r % 2) ? 1.0f : -1.0f) - mean);",
          "    }",
          "    return offs;", "}", ""]
open(OUT, "w").write("\n".join(lines))
print(f"\n{len(fits)} formations fitted -> {OUT}")
for f in fits:
    print("  ", f)
