"""How well the LIVE opponent tracker reads the opponent's plays (2026-11-16), keyed by the bot games' Battle Log replays.

The live run doesn't log the tracker's plays (decisions.json holds only the bot's own decisions), so the live stack's
reader + tracker are re-run over each game's recording as live_play.py --policy selfplay --decision-s 0.1 builds them
(LiveReader, layout a16_live, the adopted live detector, our 8-card deck, fast mode; OpponentTracker), at 10 reads a
second (dump_live_inputs.py does the same). Each new tracker play also keeps where the enemy units of its card stood.

Answer key: the game's replay (bot_pairs, pair_bot_replays.py), its top hand's plays (`plays`, source hand, side top),
each placed at the first enemy detection of its card (bot77.playbook.placed over bot77.spawns.deploys, as
build_placements does). Live video time -> replay video time: our own plays as the live cycle tracker saw them vs the
replay's bottom-hand plays (best constant offset, same card within 0.5 s).

Pairing: by the opponent's name (read live, in the chunk logs; read on the replay, `matches`), then our own plays.

A truth play is found when a tracker play of the same card fires within --window s of it (one-to-one, closest first);
"wrong card": an unmatched tracker play of another card in the window. Leftover tracker plays are phantoms, sorted
into causes: our own card (in our deck, not in theirs), a spawn of one of their cards (Goblin Hut -> Spear Goblins),
a recount (their card, already played earlier, no play of it near), late/early (a play of that card 2-8 s away),
not in either deck (a misread), other.

  uv run --no-sync python scripts/eval_live_tracker.py --limit 10            # run + score the first 10 paired games
  uv run --no-sync python scripts/eval_live_tracker.py --score-only          # score what's cached
  uv run --no-sync python scripts/eval_live_tracker.py --score-only --tracker new --split held
      # the new PlayTracker (bot77.live.opponent) replayed over the cached units, on the held-out half; also scores
      # its likely hand and elixir against the replay's top HUD
Answer keys are cached in CACHE/_truth (--rerun-truth rebuilds them).
Cache: /Volumes/LaCie/bot-77-data/tracker_eval/<game>.json (big; the laptop disk is nearly full).
"""
import argparse
import json
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

import lancedb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
CACHE = Path("/Volumes/LaCie/bot-77-data/tracker_eval")
DECK = "Giant,Battle Ram,Mini P.E.K.K.A,Mega Minion,Bomber,Goblin Hut,Fireball,Arrows"


def run_game(game: str, cards: list[dict], det, fps: float) -> dict:
    from bot77.layout import Layout
    from bot77.live.state import LiveReader, OpponentTracker
    from bot77.video import iter_frames
    by_name = {c["name"]: c["id"] for c in cards}
    reader = LiveReader(Layout.load("a16_live"), cards, det, deck=[by_name[n] for n in DECK.split(",")])
    reader.fast = True
    opp = OpponentTracker()
    out = {"game": game, "fps": fps, "plays": [], "units": [], "clock": []}
    last_units = None
    for t, im in iter_frames(ROOT / "data/raw/live_games" / game / "game.mkv", fps):
        st = reader.read(im, t)
        if not st.in_battle:
            continue
        if not out["clock"] or t - out["clock"][-1][0] >= 1.0:
            out["clock"].append((round(t, 2), st.elapsed_s))
        if st.units is not last_units and (last_units is None or st.units != last_units):
            out["units"].append([round(t, 2), [[u.card, u.team, round(u.x, 2), round(u.y, 2), round(u.conf, 2)]
                                               for u in st.units]])
            last_units = st.units
        for p in opp.update(st):
            at = [(round(u.x, 2), round(u.y, 2)) for u in st.units if u.team == "enemy" and u.card == p["card"] and u.conf >= 0.5]
            out["plays"].append({"t": round(t, 2), "card": p["card"], "elapsed_s": st.elapsed_s, "at": at})
    names = {c["id"]: c["name"] for c in cards}
    out["own_plays"] = [{"t": round(e.t, 2), "card": names.get(e.old)} for e in (reader.tracker.events if reader.tracker else [])
                        if e.kind == "play"]
    return out


def truth_for(db, rep: str, cards: list[dict]) -> tuple[list[dict], list[dict]]:
    """(opponent plays with x, y; our plays) of a replay match, in replay video time."""
    from bot77.layout import Layout
    from bot77.playbook import placed
    from bot77.spawns import deploys
    rows = db.open_table("plays").search().where(f"game_id = '{rep}' AND source = 'hand'") \
        .select(["side", "t", "card", "slot", "elixir_before", "elixir_after"]).limit(10**4).to_arrow().to_pylist()
    top = sorted(({"t": r["t"], "card": r["card"], "elixir_before": r["elixir_before"], "elixir_after": r["elixir_after"]}
                  for r in rows if r["side"] == "top" and r["card"]), key=lambda p: p["t"])
    bot = sorted(({"t": r["t"], "card": r["card"]} for r in rows if r["side"] == "bottom" and r["card"]), key=lambda p: p["t"])
    names = [c["name"] for c in cards]
    ctype = {c["name"]: c["type"] for c in cards}
    d = db.open_table("unit_detections_v5l_e15").search().where(f"match_id = '{rep}'").limit(10**7).to_arrow().to_pylist()
    v = db.open_table("videos").search().where(f"video_id = '{rep.rsplit('_m', 1)[0]}'").to_arrow().to_pylist()[0]
    ax0, ay0, ax1, _ = Layout.load(v["layout_id"]).katacr_arena
    dep = deploys(d, names, (ax1 - ax0) / 18, lambda c: ctype.get(c) == "building", lambda c: ctype.get(c) == "spell",
                  (ax0, ay0), spell_team=None)
    return placed(top, dep, "enemy"), placed(bot, dep, "friendly")


def offset(own: list[dict], rep_own: list[dict]) -> tuple[float, float]:
    """(replay t - live t, share of our live plays matched within 0.5 s at it)."""
    if not own or not rep_own:
        return 0.0, 0.0
    score = lambda o: sum(1 for p in own if any(q["card"] == p["card"] and abs(q["t"] - o - p["t"]) <= 0.5 for q in rep_own))  # noqa: E731
    seeds = [q["t"] - p["t"] for p in own for q in rep_own if q["card"] == p["card"]]
    best = max(seeds, key=lambda s: (score(s), -abs(s)))
    best = max((best + d / 20 for d in range(-10, 11)), key=lambda o: (score(o), -abs(o - best)))
    hits = [q["t"] - best - p["t"] for p in own for q in rep_own if q["card"] == p["card"] and abs(q["t"] - best - p["t"]) <= 0.5]
    if hits:  # centre the window on the matched plays
        best += stats.median(hits)
    return best, score(best) / len(own)


def spawn_map(cards: list[dict]) -> dict[str, set[str]]:
    try:
        from bot77.sim import cards as sc
        from bot77.sim.validate import produces
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for c in cards:
        e = sc.engine_id(c["name"])
        if e is not None:
            try:
                out[c["name"]] = set(produces(e)) - {c["name"]}
            except Exception:  # noqa: BLE001
                pass
    return out


def live_opponent_names() -> dict[str, str]:
    """bot game -> the opponent name the live loop read (its chunk logs: "game X starting", then "Game N  vs NAME")."""
    import re
    out = {}
    for f in sorted((ROOT / "data/live").glob("*_chunk*.log")):
        cur = None
        for line in f.read_text(errors="replace").splitlines():
            m = re.search(r"game (bot_\S+) starting", line)
            if m:
                cur = m[1]; continue
            m = re.search(r"Game \d+\s+vs (.*)$", line)
            if m and cur:
                out[cur] = m[1].strip(); cur = None
    return out


NAME_MIN = 0.7


def name_candidates(db, dates: tuple) -> dict[str, list[tuple[str, float]]]:
    """bot game -> [(replay match, name similarity)] for the replays (0-2 days later, not a duplicate copy) whose
    opponent name fits the one read live. Pairing by our own plays alone (bot_pairs) is not enough on the phone: the
    bot's opening is so regular that another game's replay matched bot_2026-11-14_160808 at 89% (2026-11-16)."""
    import difflib
    import re
    from bot77.pairing import replay_can_be_of
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())  # noqa: E731
    reps = db.open_table("matches").search().where("source = 'replay_bot' AND pov_side = 'bottom' AND split != 'duplicate'") \
        .select(["match_id", "opponent_name"]).limit(10**5).to_arrow().to_pylist()
    out = {}
    for g, n in live_opponent_names().items():
        if not g.startswith(dates):
            continue
        sc = [(r["match_id"], difflib.SequenceMatcher(None, norm(n), norm(r["opponent_name"])).ratio()) for r in reps
              if replay_can_be_of(g, r["match_id"])]
        out[g] = sorted((x for x in sc if x[1] >= NAME_MIN), key=lambda x: -x[1])
    return out


def split_of(game: str) -> str:
    """The new tracker's tune / held-out halves (2026-11-16): by the game id's md5, so it never depends on the list."""
    import hashlib
    return "tune" if int(hashlib.md5(game.encode()).hexdigest(), 16) % 2 == 0 else "held"


def top_hud(db, rep: str, names: dict) -> tuple[list[str], list]:
    """(their deck, [(replay video t, their 4 cards, their elixir)] every 0.5 s) from the replay's top-side video."""
    m = db.open_table("matches").search().where(f"match_id = '{rep}'").select(["pair_match_id"]).to_arrow().to_pylist()
    top = m[0]["pair_match_id"] if m else None
    if not top:
        return [], []
    tm = db.open_table("matches").search().where(f"match_id = '{top}'").select(["deck_names"]).to_arrow().to_pylist()
    rows = db.open_table("hud_states").search().where(f"match_id = '{top}'").select(["t_video", "hand", "hand_state", "elixir"]) \
        .limit(10**6).to_arrow().to_pylist()
    out, last = [], -1e9
    for r in sorted(rows, key=lambda r: r["t_video"]):
        if r["t_video"] - last < 0.5 or not all(s == "card" for s in r["hand_state"]):
            continue
        last = r["t_video"]
        out.append((round(r["t_video"], 2), [names.get(c) for c in r["hand"]], r["elixir"]))
    return (tm[0]["deck_names"] if tm else []), out


def paired_truth(db, g: str, cands: list, cards: list[dict], rerun: bool = False) -> dict:
    """The answer key of one bot game, cached in CACHE/_truth (loading it from lancedb takes most of a scoring run):
    among the replays whose opponent name fits, the one our own plays line up with best; its top-hand plays (with
    where they landed and their elixir), the time offset, their deck and hand / elixir every 0.5 s."""
    f = CACHE / "_truth" / f"{g}.json"
    if f.exists() and not rerun:
        return json.loads(f.read_text())
    run = json.loads((CACHE / f"{g}.json").read_text())
    best = None
    for rep, ratio in cands:
        truth, rep_own = truth_for(db, rep, cards)
        off, agree = offset(run["own_plays"], rep_own)
        if best is None or agree > best["agree"]:
            best = {"rep": rep, "truth": truth, "off": off, "agree": agree, "ratio": ratio, "own": rep_own}
    best["deck"], best["hud"] = top_hud(db, best["rep"], {c["id"]: c["name"] for c in cards})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(best, default=float))
    return best


def own_spots(own: list[dict], rep_own: list[dict], off: float) -> list[dict]:
    """Our plays as the live cycle tracker logged them, with where they landed: live, the bot knows where it tapped;
    here the replay's first sighting of the unit stands in for the tap (same card, within 2 s)."""
    out, used = [], set()
    for o in own:
        c = [(abs(q["t"] - off - o["t"]), k) for k, q in enumerate(rep_own)
             if k not in used and q["card"] == o["card"] and abs(q["t"] - off - o["t"]) <= 2.0 and q.get("x") is not None]
        if c:
            k = min(c)[1]; used.add(k)
            out.append({**o, "x": rep_own[k]["x"], "y": rep_own[k]["y"]})
        else:
            out.append(dict(o))
    return out


def replay_new(run: dict, cards: list[dict], pr: dict | None = None) -> dict:
    """bot77.live.opponent.PlayTracker over a game's cached units and own plays, as the live loop would feed it."""
    from bot77.live.opponent import PlayTracker, replay
    trk = PlayTracker.from_cards(cards, our_deck=DECK.split(","))
    own = own_spots(run["own_plays"], pr["own"], pr["off"]) if pr and pr.get("own") else run["own_plays"]
    plays, hand = replay(trk, run["units"], run["clock"], own)
    return {"plays": plays, "hand_series": hand}


def score_hand(run: dict, pr: dict, off: float, lo: float, hi: float, g: str) -> list[dict]:
    """Likely hand and elixir vs the replay's top hand every 0.5 s (live t -> replay t + off)."""
    hud = pr.get("hud") or []
    if not hud:
        return []
    import bisect
    ts = [h[0] for h in hud]
    out = []
    for t, hand, sure, el in run["hand_series"]:
        if not lo + 1.0 <= t <= hi:
            continue
        i = bisect.bisect_left(ts, t + off)
        if i >= len(ts) or abs(ts[i] - t - off) > 0.6:
            continue
        real = set(hud[i][1])
        out.append({"game": g, "t": t, "hit": len(real & set(hand)), "n": len([c for c in hand if c]), "sure": sure,
                    "el": el, "el_real": hud[i][2], "deck_known": len(set(pr.get("deck") or [])) })
    return out


def report_hand(rows: list[dict]) -> None:
    import numpy as np
    print(f"\nlikely hand vs their real hand ({len(rows)} samples, every 0.5 s):")
    for lab, sel in (("all", rows), ("8 cards known", [r for r in rows if r["sure"]]),
                     ("deck partly known", [r for r in rows if not r["sure"]])):
        if not sel:
            continue
        print(f"  {lab:18s} n {len(sel):6d}  cards right of 4: {np.mean([r['hit'] for r in sel]):.2f}  "
              f"all 4 right {np.mean([r['hit'] == 4 for r in sel]):.0%}  named {np.mean([r['n'] for r in sel]):.2f}")
    e = [r for r in rows if r["el_real"] is not None and r["el"] is not None]
    if e:
        err = np.array([r["el"] - r["el_real"] for r in e])
        print(f"elixir estimate - their real elixir: MAE {np.abs(err).mean():.2f}, median {np.median(err):+.2f}, "
              f"within 1 {np.mean(np.abs(err) <= 1):.0%}, within 2 {np.mean(np.abs(err) <= 2):.0%} (n {len(e)})")


OWN_UNIT_S = 15.0  # a unit of a card we played this recently can be ours


def score_game(run: dict, truth: list[dict], off: float, window: float, our_deck: set[str], spawns: dict) -> dict:
    tp = [{**p, "t_live": p["t"] - off} for p in truth]
    tr = [dict(p) for p in run["plays"]]
    pairs = sorted(((abs(q["t"] - p["t_live"]), i, j) for i, p in enumerate(tp) for j, q in enumerate(tr)
                    if q["card"] == p["card"] and abs(q["t"] - p["t_live"]) <= window), key=lambda x: x[0])
    ut, uq = set(), set()
    for _, i, j in pairs:
        if i in ut or j in uq:
            continue
        ut.add(i); uq.add(j)
        tp[i]["match"] = j; tr[j]["match"] = i
    for i, p in enumerate(tp):  # a tracker play of another card in the window: found, wrong card
        if i in ut:
            continue
        c = [(abs(q["t"] - p["t_live"]), j) for j, q in enumerate(tr) if j not in uq and abs(q["t"] - p["t_live"]) <= window]
        if c:
            j = min(c)[1]; uq.add(j); tp[i]["wrong"] = j; tr[j]["wrong"] = i
    opp_cards = {p["card"] for p in truth}
    prod = set().union(*(spawns.get(c, set()) for c in opp_cards)) if opp_cards else set()
    for j, q in enumerate(tr):
        if "match" in q or "wrong" in q:
            continue
        near = [abs(p["t_live"] - q["t"]) for p in tp if p["card"] == q["card"]]
        before = [p for p in tp if p["card"] == q["card"] and p["t_live"] < q["t"]]
        own_before = [o for o in run["own_plays"] if o["card"] == q["card"] and 0 <= q["t"] - o["t"] <= OWN_UNIT_S]
        own_near = [o["card"] for o in run["own_plays"] if 0 <= q["t"] - o["t"] <= 3.0]
        q["after_own"] = own_near[-1] if own_near else None
        if q["card"] in our_deck and (own_before or q["card"] not in opp_cards):
            q["cause"] = "our own card"
        elif q["card"] in prod and q["card"] not in opp_cards:
            q["cause"] = "spawn of their card"
        elif near and min(near) <= 8:
            q["cause"] = "late/early (2-8 s)"
        elif before:
            q["cause"] = "recount"
        elif q["card"] not in opp_cards and q["card"] not in our_deck:
            q["cause"] = "in neither deck"
        else:
            q["cause"] = "other"
    for k, p in enumerate(tp):
        p["since_prev"] = p["t"] - tp[k - 1]["t"] if k else None
    return {"truth": tp, "tracker": tr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2026-11-14,2026-11-15")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--games", default=None, help="comma list (default: every paired game on --dates)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--tracker", choices=("old", "new"), default="old",
                    help="old: the live OpponentTracker's cached plays; new: bot77.live.opponent.PlayTracker replayed "
                         "over the cached units (seconds, no video)")
    ap.add_argument("--play-time", choices=("seen", "known"), default="seen",
                    help="new tracker: score a play at its first sighting (its time as the board shows it; default) "
                         "or when it was counted (the old tracker's plays are both, 2 reads after the first sighting)")
    ap.add_argument("--split", choices=("all", "tune", "held"), default="all",
                    help="tune / held: the half of the games the new tracker was tuned on, or the other half")
    ap.add_argument("--rerun-truth", action="store_true", help="rebuild the cached answer keys (CACHE/_truth)")
    ap.add_argument("--out", default=None, help="write the per-play rows here (json)")
    a = ap.parse_args()
    db = lancedb.connect(str(ROOT / "data/lancedb"))
    cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
    dates = tuple(f"bot_{d}" for d in a.dates.split(","))
    cands = name_candidates(db, dates)
    games = a.games.split(",") if a.games else sorted(cands)
    games = [g for g in games if cands.get(g)][: a.limit]
    live_n = {r["game_id"]: r["opp_plays"] for r in db.open_table("bot_games").search().select(["game_id", "opp_plays"])
              .limit(10**5).to_arrow().to_pylist()}
    CACHE.mkdir(parents=True, exist_ok=True)
    det = None
    for n, g in enumerate(games):
        f = CACHE / f"{g}.json"
        if a.score_only or (f.exists() and not a.rerun):
            continue
        if det is None:
            from bot77.detect.yolo import Bot77Detector
            from bot77.models import adopted_path
            det = Bot77Detector(Path(adopted_path("detector_live", "data/models/bot77_detector/v6l/best.pt")))
        r = run_game(g, cards, det, a.fps)
        f.write_text(json.dumps(r, default=float))
        print(f"[{n + 1}/{len(games)}] {g}: {len(r['plays'])} tracker plays, {len(r['own_plays'])} own plays", flush=True)

    our_deck = set(DECK.split(","))
    spawns = spawn_map(cards)
    allt, allq, per_game, hands = [], [], [], []
    for g in games:
        f = CACHE / f"{g}.json"
        if not f.exists():
            continue
        if a.split != "all" and split_of(g) != a.split:
            continue
        pr = paired_truth(db, g, cands[g], cards, a.rerun_truth)
        rep, truth, off, agree, ratio = pr["rep"], pr["truth"], pr["off"], pr["agree"], pr["ratio"]
        if agree < 0.6:
            print(f"{g}: own plays agree only {agree:.0%} with {rep} (name {ratio:.2f}): skipped")
            continue
        run = json.loads(f.read_text())
        if not run["clock"]:
            continue
        if a.tracker == "new":
            run = {**run, **replay_new(run, cards, pr)}
            if a.play_time == "known":
                run["plays"] = [{**{k: v for k, v in q.items() if k != "t_known"}, "t": q["t_known"]} for q in run["plays"]]
        lo, hi = run["clock"][0][0], run["clock"][-1][0]  # the live recording's battle span (it can be cut short)
        truth = [p for p in truth if lo + 1.0 <= p["t"] - off <= hi - a.window]
        run = {**run, "plays": [q for q in run["plays"] if lo + 1.0 <= q["t"] <= hi - a.window]}
        s = score_game(run, truth, off, a.window, our_deck, spawns)
        for p in s["truth"]:
            p.update(game=g, replay=rep)
        for q in s["tracker"]:
            q.update(game=g, replay=rep, replay_t=round(q["t"] + off, 2))
        allt += s["truth"]; allq += s["tracker"]
        if a.tracker == "new":
            hands += score_hand(run, pr, off, lo, hi, g)
        nf = sum("match" in p for p in s["truth"])
        per_game.append((g, rep, round(ratio, 2), len(truth), len(s["tracker"]), nf, round(off, 2), round(agree, 2),
                         f"live {live_n.get(g)} / offline {len(run['plays'])}", f"span {hi - lo:.0f} s"))
    print(f"tracker: {a.tracker}; games: {a.split}")
    report(allt, allq, per_game)
    if hands:
        report_hand(hands)
    if a.out:
        Path(a.out).write_text(json.dumps({"truth": allt, "tracker": allq, "games": per_game}, default=str))


def report(allt, allq, per_game):
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "0/0"  # noqa: E731
    print(f"\n{len(per_game)} games; truth plays {len(allt)}, tracker plays {len(allq)}")
    for g in per_game:
        print("  ", *g)
    found = [p for p in allt if "match" in p]
    wrong = [p for p in allt if "wrong" in p]
    print(f"recall (right card, within window): {pct(len(found), len(allt))}; found with a wrong card: {len(wrong)}")
    good = [q for q in allq if "match" in q]
    print(f"precision (tracker plays that are a real play of that card): {pct(len(good), len(allq))}")
    print(f"card accuracy (of found plays, card right): {pct(len(found), len(found) + len(wrong))}")
    qs = {(q["game"], i): q for i, q in enumerate(allq)}  # noqa: F841
    by_game = defaultdict(list)
    for q in allq:
        by_game[q["game"]].append(q)
    dts, dist = [], []
    for p in found:
        q = by_game[p["game"]][p["match"]]
        dts.append(q["t"] - p["t_live"])
        if p.get("x") is not None and q["at"]:
            dist.append(min(((x - p["x"]) ** 2 + (y - p["y"]) ** 2) ** 0.5 for x, y in q["at"]))
    lat = [q["t_known"] - q["t"] for q in allq if q.get("t_known") is not None]
    if lat:
        sl = sorted(lat)
        print(f"latency (counted - first seen): median {stats.median(sl):.2f} s, 90% {sl[int(0.9 * len(sl))]:.2f} s")
    if dts:
        print(f"timing (tracker - truth): median {stats.median(dts):+.2f} s, IQR {sorted(dts)[len(dts) // 4]:+.2f}..{sorted(dts)[3 * len(dts) // 4]:+.2f}")
    if dist:
        sd = sorted(dist)
        print(f"place: distance to the truth tile, median {stats.median(sd):.1f} tiles, <= 2 tiles {sum(d <= 2 for d in sd) / len(sd):.0%}, "
              f"<= 4 {sum(d <= 4 for d in sd) / len(sd):.0%} (n {len(sd)} with both spots)")
    print("\nby card (truth n, found, wrong card, tracker plays, phantoms):")
    cs = Counter(p["card"] for p in allt) + Counter(q["card"] for q in allq)
    for c, _ in cs.most_common():
        n = sum(p["card"] == c for p in allt)
        f = sum(p["card"] == c and "match" in p for p in allt)
        w = sum(p["card"] == c and "wrong" in p for p in allt)
        tq = sum(q["card"] == c for q in allq)
        ph = sum(q["card"] == c and "match" not in q and "wrong" not in q for q in allq)
        print(f"  {c:20s} {n:4d} {f:4d} ({f / n if n else 0:4.0%}) {w:3d} {tq:4d} {ph:4d}")
    print("\nby time since the opponent's previous play:")
    for lo, hi in ((None, None), (0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)):
        sel = [p for p in allt if (lo is None and p["since_prev"] is None) or
               (lo is not None and p["since_prev"] is not None and lo <= p["since_prev"] < hi)]
        lab = "first play" if lo is None else f"{lo}-{hi if hi < 1e8 else '+'} s"
        print(f"  {lab:12s} {pct(sum('match' in p for p in sel), len(sel))}")
    ph = [q for q in allq if "match" not in q and "wrong" not in q]
    print(f"\nphantoms {len(ph)} ({len(ph) / max(len(per_game), 1):.1f} a game), by cause:")
    for c, n in Counter(q["cause"] for q in ph).most_common():
        ex = [f"{q['game']} clock {round(q['elapsed_s'] or 0, 1)} s {q['card']}" for q in ph if q["cause"] == c][:3]
        print(f"  {c:22s} {n:4d}  e.g. {'; '.join(ex)}")
        if c == "in neither deck":
            print("      cards:", Counter(q["card"] for q in ph if q["cause"] == c).most_common(6))
            print("      within 3 s after our play of:", Counter(q.get("after_own") for q in ph if q["cause"] == c).most_common(6))


if __name__ == "__main__":
    main()
