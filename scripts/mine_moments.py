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
  spent_counter   a push with their win condition takes TOWER_HIT of a tower of ours within WINCON_WINDOW s while our
                  hand holds none of our counters to it; the moment is our last
                  counter played before it, when their win condition was in their hand (Cole: "We know he has battle
                  ram, so we should try to save a card for that").
  push_through    a tower of ours loses TOWER_HIT of its HP within WINCON_WINDOW s (and spent_counter doesn't
                  explain it): the moment is their last win-condition play (else their last play) before it, so the
                  clip shows their push and our answer (which card, where, when). Tower HP, not unit positions: a
                  Battle Ram dies on contact and its Barbarians aren't read by that name (2026-10-31: the position
                  version fired 7 times in 70 games with 56 towers lost).
  (2026-11-02, the failure modes of Cole's review of 7 losses, scripts/failure_modes.py)
  pointless_push  Giant / Battle Ram at the bridge, or a defender sent over to back a push, with no elixir lead
                  (both HUDs) or into a stacked lane, not when behind on towers, defending, or counterpushing.
  wasted_spell    a spell of ours (the spot the bot tapped, bot_actions) that hits less than its cost.
  missed_finish   their princess tower low enough for a spell in hand (Fireball 7%, Arrows 3%), not cast.
  too_early       a defender played on our half while their push is still on their side of the river.
  bad_placement   a defender in the other lane from their push, or Mini P.E.K.K.A / Bomber dropped on top of it.
  no_defence      their troops on a standing tower of ours, elixir and a troop in hand, no play for 4 s.
  Every moment carries `blind`: the live reader had a card locked out then (decisions.md 2026-11-02).
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
# 2026-11-02 (Cole's review, "ARE WE ALLERGIC TO ARROWS"): it found 1 of his 8 missed spells. His are spam at the
# bridge (Skeletons, Goblins, Spear Goblins crossing: ~4 elixir at a third to two thirds each), so the bunch may sit up
# to SPELL_UP_TO on their side of the river, and a spell needs its cost plus a little (Arrows 3.5, Fireball 5)
SPELLS = {"Arrows": (3.5, 3, 3.5, 3), "Fireball": (2.5, 4, 5.0, 2)}
SPELL_UP_TO = 21.0  # engine y: the river is 16.5, so ~4.5 tiles past it
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
    # plays with a read card and time (a few hand reads have no card)
    my_plays = [p for p in g.plays if p["side"] == "bottom" and p["t_match"] is not None and p["card"]]
    their_plays = [p for p in g.plays if p["side"] == "top" and p["t_match"] is not None and p["card"]]
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
                if cy > SPELL_UP_TO:  # on our half or crossing (at the bridge), not deep in theirs
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

    # pushes that got through: a tower of ours loses TOWER_HIT of its HP within WINCON_WINDOW s (tower_hp reads; a
    # unit's position isn't enough: a Battle Ram dies on contact and turns into Barbarians, 2026-10-31)
    names = dict(own_l="left", own_r="right", own_king="King")
    last_hit = -99.0
    for t0 in range(0, int(tm(g.frame_ts[-1])) - 2):
        if t0 - last_hit < 15:
            continue
        drop = None
        for key in OWN_TOWERS:
            f0, f1 = tower_frac(g, key, t0 + g.offset), tower_frac(g, key, t0 + WINCON_WINDOW + g.offset)
            if f0 is not None and f1 is not None and f0 - f1 >= TOWER_HIT:
                drop = (key, f0 - f1)
                break
        if drop is None:
            continue
        # where it started dropping, to the second
        start = next((t for t in range(t0, t0 + int(WINCON_WINDOW))
                      if (tower_frac(g, drop[0], t + 1 + g.offset) or 1) < (tower_frac(g, drop[0], t0 + g.offset) or 1) - 0.01), t0)
        last_hit = start
        theirs_before = [p for p in their_plays if start - 15 <= p["t_match"] <= start]
        ours_before = [p for p in my_plays if start - 15 <= p["t_match"] <= start + 3]
        wcs = [p for p in theirs_before if p["card"] in WINCON]
        origin = (wcs or theirs_before or [{"t_match": start - 8}])[-1]["t_match"]
        h = at(ours, origin + g.offset)
        counters = COUNTERS_AIR if wcs and wcs[-1]["card"] in AIR else COUNTERS_GROUND
        spent = [p for p in my_plays if p["card"] in counters and origin - 20 <= p["t_match"] < origin]
        if wcs and h is not None and not any(c in counters for c in h[1]) and spent:
            sp = spent[-1]
            th = at(theirs, sp["t_match"] + g.offset)
            w = wcs[-1]["card"]
            target = next((p["card"] for p in reversed(their_plays) if p["t_match"] <= sp["t_match"]), "?")
            out.append({"kind": "spent_counter", "t": round(sp["t_match"], 1), "card": sp["card"], "wincon": w,
                        "text": f"spent {sp['card']} (on their {target}) at {mmss(sp['t_match'])}"
                                f"{', their ' + w + ' in their hand' if th is not None and w in th[1] else ''}; "
                                f"their {w} came at {mmss(wcs[-1]['t_match'])} with no counter in our hand "
                                f"({', '.join(h[1])}): our {names[drop[0]]} tower -{drop[1]:.0%}"})
            continue
        out.append({"kind": "push_through", "t": round(max(0.0, origin - 1), 1),
                    "text": f"our {names[drop[0]]} tower lost {drop[1]:.0%} from {mmss(start)}; their plays before it: "
                            f"{', '.join(p['card'] + ' ' + mmss(p['t_match']) for p in theirs_before) or 'none read'}; "
                            f"ours: {', '.join(p['card'] + ' ' + mmss(p['t_match']) for p in ours_before) or 'none'}"})

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
    out += mine_modes(db, g, gm, ours, theirs, my_plays, their_plays)
    stuck = blank_stretches(gm["gid"])
    for c in out:
        c.update(game=gm["gid"], replay=g.game_id, opp=gm["opp"], result=gm["result"], clock=mmss(c["t"]))
        # the live reader had a card locked out (decisions.md 2026-11-02): the agent was playing on 3 cards
        c["blind"] = any(a - 2 <= c["t"] <= b for a, b in stuck)
    return out


def blank_stretches(gid: str, min_s: float = 5.0) -> list[tuple[float, float]]:
    """(from, to) in game seconds where a hand slot read blank for min_s or more in the live log (decisions.json)."""
    f = blr.GAMES / gid / "decisions.json"
    if not f.exists():
        return []
    rows = [r for r in json.loads(f.read_text()) if r.get("clock") is not None]
    out, start = [], {}
    for r in rows:
        for k in range(4):
            if r["hand"][k] is None:
                start.setdefault(k, r["clock"])
            elif k in start:
                a = start.pop(k)
                if r["clock"] - a >= min_s:
                    out.append((a, r["clock"]))
    out += [(a, rows[-1]["clock"]) for a in start.values() if rows and rows[-1]["clock"] - a >= min_s]
    return out


# Failure modes from Cole's review of 7 losses (2026-11-02, scripts/failure_modes.py). Engine board: our half y < 16.5
OPP_TOWERS = {"opp_l": (3.0, 28.0), "opp_r": (14.0, 28.0)}
OUR_WINCON = {"Giant", "Battle Ram"}
DEFENDERS = {"Mini P.E.K.K.A", "Mega Minion", "Bomber"}
MELEE_DROP = {"Mini P.E.K.K.A", "Bomber"}  # dropped on top of an attacker, they die before they deploy
AIR_UNITS = {"Minions", "Mega Minion", "Bats", "Minion Horde", "Baby Dragon", "Inferno Dragon", "Electro Dragon",
             "Skeleton Dragons", "Balloon", "Lava Hound", "Lava Pups", "Phoenix", "Flying Machine"}
RIVER = 16.5
FINISH = {"Fireball": 0.07, "Arrows": 0.03}  # a princess tower this low: the spell's tower damage takes it
DEFEND_NEAR, DEFEND_WINDOW, DEFEND_HIT = 6.0, 4.0, 0.05


def lane(x: float) -> str:
    return "left" if x < 9 else "right"


def units_at(g, t_match: float, team: str) -> list[tuple[str, float, float, float]]:
    """(card, x, y, elixir worth) of one side's units at the frame nearest a game time, engine board."""
    if not g.frame_ts:
        return []
    import bisect
    tv = t_match + g.offset
    i = min(max(bisect.bisect_left(g.frame_ts, tv), 0), len(g.frame_ts) - 1)
    if i > 0 and abs(g.frame_ts[i - 1] - tv) < abs(g.frame_ts[i] - tv):
        i -= 1
    out = []
    for u in g.frames.get(g.frame_ts[i], []):
        if u["team"] == team and u["card"]:
            x, y = ours_to_engine(u["x"], u["y"])
            out.append((u["card"], x, y, worth(u["card"])))
    return out


def towers_up(g, t_match: float) -> tuple[int, int]:
    """(ours, theirs) princess towers standing."""
    tv = t_match + g.offset
    return (sum((tower_frac(g, k, tv) or 1) > 0 for k in ("own_l", "own_r")),
            sum((tower_frac(g, k, tv) or 1) > 0 for k in ("opp_l", "opp_r")))


def our_spells(gid: str) -> list[dict]:
    """The bot's own spells with the spot it tapped (bot_actions; a replay spots only ~21% of spells), game time from
    the live log's clock at that moment. Two rows for one cast (a repeated tap) are one."""
    import lancedb as _l
    db = _l.connect(ROOT / "data/lancedb")
    rows = db.open_table("bot_actions").search().where(f"game_id = '{gid}'").limit(1000).to_arrow().to_pylist()
    f = blr.GAMES / gid / "decisions.json"
    dec = [r for r in json.loads(f.read_text()) if r.get("clock") is not None] if f.exists() else []
    out = []
    for r in sorted(rows, key=lambda r: r["t"]):
        if r["card"] not in SPELLS:
            continue
        clock = min(dec, key=lambda d: abs(d["t"] - r["t"]))["clock"] if dec else 180 - (r["time_left_s"] or 0)
        if out and out[-1]["card"] == r["card"] and clock - out[-1]["t"] < 1.5:
            continue
        x, y = ours_to_engine(r["x_tiles"], r["y_tiles"])
        out.append({"card": r["card"], "t": float(clock), "x": x, "y": y})
    return out


def mine_modes(db, g, gm: dict, ours, theirs, my_plays: list[dict], their_plays: list[dict]) -> list[dict]:
    out = []
    tv = lambda t: t + g.offset  # game t -> video t
    hand_at = lambda side, t: at(side, tv(t)) or (0, [], 0.0)
    spot = lambda p: ours_to_engine(*p["spot"]) if p.get("spot") else None
    names = lambda us: ", ".join(sorted({u[0] for u in us}))

    # pointless push: a win condition at the bridge, or a defender sent over to back a push, with no elixir lead or
    # into a stacked lane (Cole: "WAIT", "play giant in the back if we're not up elixir, or pressing")
    for p in my_plays:
        sp = spot(p)
        if p["card"] not in OUR_WINCON | DEFENDERS or sp is None:
            continue
        x, y = sp
        if y < (12.5 if p["card"] in OUR_WINCON else RIVER - 1):  # in the back, or a defender on our side: fine
            continue
        mine_n, theirs_n = towers_up(g, p["t_match"])
        if mine_n < theirs_n:  # behind on towers: pushing is the job
            continue
        before = p["t_match"] - 1.5  # read before the card left the hand (plays are seen ~1-2 s after the tap)
        el_us, el_them = hand_at(ours, before)[2] or 0.0, hand_at(theirs, before)[2] or 0.0
        foes = units_at(g, before, "opp")
        if sum(u[3] for u in foes if u[2] < RIVER and lane(u[1]) == lane(x)) >= 2:  # a push of theirs here: defending
            continue
        mates = sum(u[3] for u in units_at(g, before, "own") if lane(u[1]) == lane(x) and u[2] > 10)
        if p["card"] in OUR_WINCON and mates >= 3:  # a counterpush with what defended
            continue
        if p["card"] in DEFENDERS and mates >= 4:  # backing a real push
            continue
        cost = sc.cost(sc.engine_id(p["card"])) if sc.engine_id(p["card"]) is not None else 4
        wall = [u for u in foes if u[2] >= RIVER and lane(u[1]) == lane(x)]
        lead = el_us - el_them
        if lead < -0.5 or sum(u[3] for u in wall) >= cost + 1:
            why = [f"{el_us:.1f} elixir to their {el_them:.1f}"]
            if wall:
                why.append(f"into {names(wall)} on that side ({sum(u[3] for u in wall):.0f} elixir)")
            if p["card"] in DEFENDERS:
                why.append(f"backing {mates:.0f} elixir of ours")
            out.append({"kind": "pointless_push", "t": round(p["t_match"] - 1, 1), "card": p["card"],
                        "text": f"{p['card']} at the {lane(x)} {'bridge' if y < RIVER + 1 else 'side of theirs'}: "
                                + "; ".join(why)})

    # wasted spell: one that hits less than its cost (a tower hit counts 1), from the spot the bot tapped
    for sp in our_spells(gm["gid"]):
        r, cost = SPELLS[sp["card"]][0], SPELLS[sp["card"]][1]
        if not any(q["card"] == sp["card"] and -1 <= q["t_match"] - sp["t"] <= 4 for q in my_plays):
            continue  # the tap didn't cast it
        hit = max(([u for u in units_at(g, sp["t"] + d, "opp") if (u[1] - sp["x"]) ** 2 + (u[2] - sp["y"]) ** 2 <= r * r]
                   for d in (0.3, 0.8, 1.3)), key=lambda us: sum(u[3] for u in us))
        tower = any((tower_frac(g, k, tv(sp["t"])) or 1) > 0 and (tx - sp["x"]) ** 2 + (ty - sp["y"]) ** 2 <= (r + 1.5) ** 2
                    for k, (tx, ty) in OPP_TOWERS.items())
        v = sum(u[3] for u in hit) + (1 if tower else 0)
        if v < cost - 0.5:
            out.append({"kind": "wasted_spell", "t": round(sp["t"], 1), "card": sp["card"],
                        "text": f"{sp['card']} ({cost} elixir) hit {names(hit) or 'no troops'}"
                                f"{' and a tower' if tower else ''}: about {v:.1f} elixir"})

    # didn't finish: their princess tower low enough for a spell in hand, and it isn't cast
    last = {}
    for t in range(5, int(g.frame_ts[-1] - g.offset) - 6):
        for k in OPP_TOWERS:
            f = tower_frac(g, k, tv(t))
            if not f or f <= 0 or t - last.get(k, -99) < 20:
                continue
            _, hand, el = hand_at(ours, t)
            ok = [s for s, lim in FINISH.items() if f <= lim and s in hand and (el or 0) >= SPELLS[s][1]]
            if not ok or (tower_frac(g, k, tv(t + 6)) or 1) <= 0:
                continue
            if any(q["card"] in SPELLS and t - 1 <= q["t_match"] <= t + 5 for q in my_plays):
                continue
            last[k] = t
            out.append({"kind": "missed_finish", "t": float(t), "text": f"their {'left' if k == 'opp_l' else 'right'} "
                        f"tower at {f:.0%} with {' / '.join(ok)} in hand at {el:.0f} elixir, not cast"})

    # didn't wait for it to cross / dropped on top / wrong lane: a defender played on our half
    for p in my_plays:
        sp = spot(p)
        if p["card"] not in DEFENDERS or sp is None or sp[1] >= RIVER - 1:
            continue
        x, y = sp
        before = p["t_match"] - 1.5
        foes = units_at(g, before, "opp")
        here = [u for u in foes if u[2] < RIVER and lane(u[1]) == lane(x)]
        there = [u for u in foes if u[2] < RIVER and lane(u[1]) != lane(x)]
        coming = [u for u in foes if RIVER <= u[2] <= RIVER + 7 and lane(u[1]) == lane(x) and u[0] not in AIR_UNITS]
        if not here and not there and sum(u[3] for u in coming) >= 3:
            out.append({"kind": "too_early", "t": round(before, 1), "card": p["card"],
                        "text": f"{p['card']} on our {lane(x)} side while their {names(coming)} hadn't crossed"})
            continue
        if not here and sum(u[3] for u in there) >= 3:
            out.append({"kind": "bad_placement", "t": round(before, 1), "card": p["card"],
                        "text": f"{p['card']} in the {lane(x)} lane; their push ({names(there)}) is on the other one"})
            continue
        at_play = units_at(g, p["t_match"] - 0.5, "opp")
        ontop = [u for u in at_play if p["card"] in MELEE_DROP and u[0] not in AIR_UNITS and u[3] >= 1.5
                 and (u[1] - x) ** 2 + (u[2] - y) ** 2 <= 1.5 ** 2]
        if ontop:
            out.append({"kind": "bad_placement", "t": round(before, 1), "card": p["card"],
                        "text": f"{p['card']} dropped right on top of their {names(ontop)}: it can be hit while it deploys"})

    # spent the counter: our last answer to their win condition used on something else while that win condition is
    # in their hand (Cole: "That was our last defensive card in our hand ... We know he has battle ram"); no tower
    # damage needed (the version in mine_game wants one)
    for p in my_plays:
        if p["card"] not in COUNTERS_GROUND:
            continue
        their_hand = hand_at(theirs, p["t_match"] - 1)[1]
        wcs = [c for c in their_hand if c in WINCON and c not in AIR]
        target = next((q["card"] for q in reversed(their_plays) if q["t_match"] <= p["t_match"]), None)
        if not wcs or target in WINCON:
            continue
        left = hand_at(ours, p["t_match"] + 0.5)[1]
        if "?" in left or any(c in COUNTERS_GROUND for c in left):
            continue
        out.append({"kind": "spent_counter", "t": round(p["t_match"] - 1, 1), "card": p["card"],
                    "text": f"{p['card']} on their {target or '?'} with their {' / '.join(wcs)} in hand; "
                            f"left us {', '.join(left)}: no answer to it"})

    # didn't defend / kite: their troops on a standing tower of ours, elixir and a troop in hand, no play
    last_t = -99.0
    for t in range(5, int(g.frame_ts[-1] - g.offset) - 6):
        if t - last_t < 15:
            continue
        foes = units_at(g, t, "opp")
        for key, (tx, ty) in OWN_TOWERS.items():
            if key == "own_king":
                continue
            f0 = tower_frac(g, key, tv(t))
            if not f0 or f0 <= 0:
                continue
            on = [u for u in foes if (u[1] - tx) ** 2 + (u[2] - ty) ** 2 <= DEFEND_NEAR ** 2]
            if sum(u[3] for u in on) < 3:
                continue
            _, hand, el = hand_at(ours, t)
            troops = [c for c in hand if c in OUR_WINCON | DEFENDERS | {"Goblin Hut"} and sc.engine_id(c) is not None
                      and sc.cost(sc.engine_id(c)) <= (el or 0)]
            if not troops or any(t - 2 <= q["t_match"] <= t + DEFEND_WINDOW + 1.5 for q in my_plays):
                continue
            f1 = tower_frac(g, key, tv(t + 6))
            if f1 is None or f0 - f1 < DEFEND_HIT:
                continue
            last_t = t
            out.append({"kind": "no_defence", "t": float(t),
                        "text": f"their {names(on)} on our {'left' if key == 'own_l' else 'right'} tower; we had "
                                f"{el:.1f} elixir and {', '.join(troops)}, no play for {DEFEND_WINDOW:.0f} s: "
                                f"tower -{f0 - f1:.0%}"})
            break
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
              + ", ".join(f"{k} {sum(c['kind'] == k for c in found)}" for k in ("missed_spell", "pointless_push", "spent_counter", "wasted_spell", "too_early", "bad_placement", "no_defence", "missed_finish", "push_through", "tower_lost", "leaked")))
    out = Path(a.out or ROOT / f"data/reviews/mined_{a.date}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "moments.json").write_text(json.dumps({"games": {g["gid"]: {k: g[k] for k in ("gid", "opp", "result", "run", "n")}
                                                            for g in games}, "moments": cands}, indent=1))
    print(f"{len(cands)} moments from {len(games) - len(skipped)} games -> {out / 'moments.json'}; skipped {len(skipped)}: "
          + "; ".join(f"{g} ({why})" for g, why in skipped[:6]))


if __name__ == "__main__":
    main()
