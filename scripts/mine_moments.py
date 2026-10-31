"""Mine the bot's live games for the moments it fails (2026-10-31, Cole: "make a pipeline that lets us 'mine for these
moments'? I'll keep reviewing, but set it up in a way that I can just mark yes or no so we can train on all of the
moments where our model is failing").

  uv run --no-sync python scripts/mine_moments.py --runs run40v4 run40v4b --date 2026-10-31 [--out data/reviews/mined_<date>]

Each bot game is paired with its Battle Log replay (by the opponent's name on the replay's entry, as
scripts/build_loss_review.py does) and read from the ingested replay (scripts/ingest_replays.py: both sides' plays,
both hands every frame, units every 0.25 s, towers). Detectors, the failure kinds of Cole's first review
(decisions.md 2026-10-31: placement and timing, counters, missed spells, and spam):

  missed_spell    their units bunched so one Arrows / Fireball would hit at least SPELL_MIN elixir of them (a body's
                  share of its card's cost), on our half or at our towers, while that spell is in our hand and we can
                  afford it, and we don't cast it within SPELL_WINDOW s.
  counter_spent   they play their win condition and it reaches our tower (its unit stands within CONNECT_R of one of
                  our standing towers for CONNECT_S s within WINCON_WINDOW s: tower HP reads are sparse in replays)
                  while our hand holds none of our counters to it; the moment is our last
                  counter played before it, when their win condition was in their hand (Cole: "We know he has battle
                  ram, so we should try to save a card for that").
  tower_lost      one of our towers falls: the TOWER_LEAD s before it.
  leaked          our elixir sits at 10 for LEAK_S s or more.

Counters (our deck, Cole 2026-10-31: "use our mini pekka or goblin hut for the battle ram", "mega minion on their battle
ram"): a ground win condition is answered by Mini P.E.K.K.A, Goblin Hut or Mega Minion; an air one by Mega Minion.
Candidates go to moments.json for the yes / no page (scripts/build_moment_review.py). Nothing here is a training
label: only the moments Cole marks yes become examples (decisions.md: the bot's own play is never positive signal).
"""
import argparse
import json
import sys
from pathlib import Path

import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_loss_review as blr  # noqa: E402

from bot77.sim import cards as sc  # noqa: E402
from bot77.sim.geometry import ours_to_engine  # noqa: E402
from bot77.sim.validate import load_game, tower_frac  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# radius (tiles), cost, SPELL_MIN elixir hit, bodies at least. The first cut (4 / 5 elixir, 2 bodies, one 0.5 s read)
# found ~7 a game on 10-29: a passing bunch reads as a target for a frame; now on our half and held for 1 s
SPELLS = {"Arrows": (4.0, 3, 5.0, 3), "Fireball": (2.5, 4, 6.0, 2)}
OWN_TOWERS = {"own_l": (3.0, 6.0), "own_r": (14.0, 6.0), "own_king": (8.5, 2.5)}  # engine board (ArenaLayout)
CONNECT_R, CONNECT_S = 3.5, 2.0  # a win condition "connects": within 3.5 tiles of a standing tower of ours for 2 s
SPELL_WINDOW = 3.0
WINCON = {"Hog Rider", "Battle Ram", "Ram Rider", "Royal Hogs", "Giant", "Golem", "Royal Giant", "Goblin Giant",
          "Electro Giant", "Elixir Golem", "Balloon", "Lava Hound", "Miner", "Goblin Barrel", "Graveyard", "Wall Breakers",
          "X-Bow", "Mortar", "Goblin Drill", "Skeleton Barrel", "Three Musketeers", "P.E.K.K.A", "Mega Knight",
          "Prince", "Dark Prince", "Sparky", "Goblin Hut", "Furnace", "Rascals", "Barbarians", "Elite Barbarians"}
AIR = {"Balloon", "Lava Hound", "Skeleton Barrel"}
COUNTERS_GROUND = {"Mini P.E.K.K.A", "Goblin Hut", "Mega Minion"}
COUNTERS_AIR = {"Mega Minion"}
WINCON_WINDOW, TOWER_HIT, TOWER_LEAD, LEAK_S = 12.0, 0.15, 10.0, 3.0
CARD_IDS = {}


def worth(name: str) -> float:
    """Elixir per body of the card a unit belongs to (the cost shared over the card's bodies)."""
    import clash_royale_env as cre
    if name not in CARD_IDS:
        e = sc.engine_id(name)
        if e is None:
            CARD_IDS[name] = 0.0
        else:
            st = dict(cre.get_card_stats(int(e)))
            CARD_IDS[name] = sc.cost(e) / max(1, int(st.get("bodies", 1) or 1))
    return CARD_IDS[name]


_NAMES: dict[int, str] = {}


def hud(db, match_id: str) -> list[tuple[float, list[str], float]]:
    """(video t, hand as card names, elixir) for one side of a replay, in time order. The HUD holds the game's own card
    ids (26000003); the `cards` table names them."""
    if not _NAMES:
        _NAMES.update({r["id"]: r["name"] for r in db.open_table("cards").search().select(["id", "name"])
                       .limit(10**5).to_arrow().to_pylist()})
    rows = db.open_table("hud_states").search().where(f"match_id = '{match_id}'").select(
        ["t_video", "hand", "elixir"]).limit(10**6).to_arrow().to_pylist()
    return [(r["t_video"], [_NAMES.get(c, "?") if c is not None else "?" for c in (r["hand"] or [])], r["elixir"])
            for r in sorted(rows, key=lambda r: r["t_video"])]


def at(series, t):
    import bisect
    i = bisect.bisect_right([s[0] for s in series], t) - 1
    return series[i] if i >= 0 else None


def mmss(t_match: float) -> str:
    left = 180 - t_match
    if left >= 0:
        return f"{int(left // 60)}:{int(left % 60):02d}"
    ot = 120 + left
    return f"{int(max(0, ot) // 60)}:{int(max(0, ot) % 60):02d} OT"


def mine_game(db, g, gm: dict) -> list[dict]:
    """Candidates in one game; times in game seconds (t_match)."""
    out = []
    ours = hud(db, g.game_id)
    theirs = hud(db, g.game_id.replace("_m01", "_top_m01"))
    my_plays = [p for p in g.plays if p["side"] == "bottom" and p["t_match"] is not None]
    their_plays = [p for p in g.plays if p["side"] == "top" and p["t_match"] is not None]
    tm = lambda t: t - g.offset  # video t -> game t

    # missed spells
    last = -99.0
    held: dict[str, int] = {}
    for t in g.frame_ts[::2]:  # every 0.5 s
        h = at(ours, t)
        if h is None:
            continue
        _, hand, el = h
        foes = []
        for u in g.frames.get(t, []):
            if u["team"] != "opp":
                continue
            x, y = ours_to_engine(u["x"], u["y"])
            foes.append((x, y, worth(u["card"]), u["card"]))
        for spell, (r, cost, need, bodies) in SPELLS.items():
            if spell not in hand or (el or 0) < cost or tm(t) - last < 8:
                held[spell] = 0
                continue
            best = None
            for cx, cy, _, _ in foes:
                if cy > 15.5:  # on our half only (at our towers or coming down on them)
                    continue
                hit = [f for f in foes if (f[0] - cx) ** 2 + (f[1] - cy) ** 2 <= r * r]
                v = sum(f[2] for f in hit)
                if best is None or v > best[0]:
                    best = (v, cx, cy, hit)
            held[spell] = held.get(spell, 0) + 1 if best and best[0] >= need and len(best[3]) >= bodies else 0
            if held[spell] >= 2:  # the same bunch for two reads in a row (1 s)
                held[spell] = 0
                cast = any(p["card"] == spell and tm(t) - 1 <= p["t_match"] <= tm(t) + SPELL_WINDOW for p in my_plays)
                if not cast:
                    names = sorted({f[3] for f in best[3]})
                    out.append({"kind": "missed_spell", "t": round(tm(t), 1), "spell": spell,
                                "x": round(best[1], 1), "y": round(best[2], 1),
                                "text": f"{spell} in hand at {el:.0f} elixir, not cast: one would hit "
                                        f"{best[0]:.1f} elixir of {', '.join(names)}"})
                    last = tm(t)
                    break

    # counters spent before their win condition
    for wp in their_plays:
        if wp["card"] not in WINCON:
            continue
        tw = wp["t_match"]
        hit = None
        near: dict[str, float] = {}
        for t in g.frame_ts:
            if not tw <= tm(t) <= tw + WINCON_WINDOW:
                continue
            for u in g.frames.get(t, []):
                if u["team"] != "opp" or u["card"] != wp["card"]:
                    continue
                x, y = ours_to_engine(u["x"], u["y"])
                for key, (tx, ty) in OWN_TOWERS.items():
                    if tower_frac(g, key, t) == 0.0 or (x - tx) ** 2 + (y - ty) ** 2 > CONNECT_R ** 2:
                        continue
                    near.setdefault(key, tm(t))
                    if tm(t) - near[key] >= CONNECT_S and hit is None:
                        hit = (key, tm(t))
        if not hit:
            continue
        counters = COUNTERS_AIR if wp["card"] in AIR else COUNTERS_GROUND
        h = at(ours, tw + g.offset)
        if h is None or any(c in counters for c in h[1]):
            continue
        spent = [p for p in my_plays if p["card"] in counters and tw - 20 <= p["t_match"] < tw]
        if not spent:
            continue
        sp = spent[-1]
        th = at(theirs, sp["t_match"] + g.offset)
        in_hand = th is not None and wp["card"] in th[1]
        target = next((p["card"] for p in reversed(their_plays) if p["t_match"] <= sp["t_match"]), "?")
        out.append({"kind": "counter_spent", "t": round(sp["t_match"], 1), "card": sp["card"], "wincon": wp["card"],
                    "text": f"spent {sp['card']} (on their {target}) at {mmss(sp['t_match'])}"
                            f"{', their ' + wp['card'] + ' in their hand' if in_hand else ''}; "
                            f"their {wp['card']} came at {mmss(tw)} with no counter in our hand "
                            f"({', '.join(h[1])}); it reached our {dict(own_l='left', own_r='right', own_king='King')[hit[0]]} "
                            f"tower at {mmss(hit[1])}"})

    # towers lost
    for key, name in (("own_l", "left"), ("own_r", "right")):
        fell = None
        for t in g.frame_ts:
            f = tower_frac(g, key, t)
            if f == 0.0:
                fell = tm(t)
                break
        if fell is not None:
            out.append({"kind": "tower_lost", "t": round(max(0.0, fell - TOWER_LEAD), 1),
                        "text": f"our {name} tower fell at {mmss(fell)}: the {TOWER_LEAD:.0f} s before it"})

    # leaked elixir
    run = None
    for t, _hand, el in ours:
        if (el or 0) >= 9.95:
            run = run if run is not None else t
        else:
            if run is not None and t - run >= LEAK_S:
                out.append({"kind": "leaked", "t": round(tm(run), 1),
                            "text": f"sat at 10 elixir for {t - run:.0f} s from {mmss(tm(run))}"})
            run = None
    for c in out:
        c.update(game=gm["gid"], replay=g.game_id, opp=gm["opp"], result=gm["result"], clock=mmss(c["t"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    db = lancedb.connect(ROOT / "data/lancedb")
    games = blr.parse_runs(a.runs)
    blr.match_replays(games, blr.replay_texts(a.date))
    cands, skipped = [], []
    for gm in games:
        mp4 = gm.pop("_replay", None)
        gm.pop("_replay_score", None)
        if mp4 is None:
            skipped.append((gm["gid"], "no replay"))
            continue
        rid = f"bl_{a.date}_r{mp4.stem.split('_')[1]}_m01"
        try:
            g = load_game(rid, "bot", "replay_bot")
        except Exception as e:  # noqa: BLE001
            skipped.append((gm["gid"], f"replay {rid}: {e}"))
            continue
        if not g.plays:
            skipped.append((gm["gid"], f"replay {rid} not ingested"))
            continue
        found = mine_game(db, g, gm)
        cands += found
        print(f"{gm['gid']} vs {gm['opp']:18s} {gm['result']:4s} {rid}: "
              + ", ".join(f"{k} {sum(c['kind'] == k for c in found)}" for k in ("missed_spell", "counter_spent", "tower_lost", "leaked")))
    out = Path(a.out or ROOT / f"data/reviews/mined_{a.date}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "moments.json").write_text(json.dumps({"games": {g["gid"]: {k: g[k] for k in ("gid", "opp", "result", "run", "n")}
                                                            for g in games}, "moments": cands}, indent=1))
    print(f"{len(cands)} moments from {len(games) - len(skipped)} games -> {out / 'moments.json'}; skipped {len(skipped)}: "
          + "; ".join(f"{g} ({why})" for g, why in skipped[:6]))


if __name__ == "__main__":
    main()
