"""How often the new input block's spell slots hold the opponent's real spells before they land (2026-11-16).

Replays bot77.live.extra_obs.OppInputs (PlayTracker + SpellTracker, the live code path) over the cached detections of
the bot games (scripts/eval_live_tracker.py's cache on the LaCie) and scores its spell slots against the games'
Battle Log replays: each real opponent spell (their top hand's plays: the moment the card left their hand, "cast")
is found when a slot of theirs holds that card within [cast - 0.5, cast + 3] s.

Reported per card: found, the delay from cast to the first slot, the slot's predicted time to impact then, and how
far ahead of the impact it was: impact ~ cast + (engine tiles from their King to the slot's landing spot) / the
game-file speed (Fireball 10 tiles/s, Rocket 5.8, Arrows 18); and slots of theirs with no real spell (phantoms).

  uv run --no-sync python scripts/eval_spell_slots.py [--split all|tune|held]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_live_tracker import CACHE, DECK, own_spots, split_of  # noqa: E402
from bot77.live.extra_obs import OppInputs  # noqa: E402
from bot77.live.spells import THEIR_KING, edist  # noqa: E402

OWN_LAG = 1.5
FILE_SPEED = {"Fireball": 10.0, "Rocket": 5.83, "Arrows": 18.0, "Goblin Barrel": 6.7, "Giant Snowball": 13.3}


def run_game(run: dict, pr: dict, cards: list[dict]) -> tuple[list, OppInputs]:
    """[(t, [(track id, card, owner, tti, land)])] after every read, and the inputs object."""
    opp = OppInputs(cards, DECK.split(","))
    # our plays as the live loop will know them: the bot's tap, before its unit or spell shows. The cache has the cycle
    # tracker's plays, logged up to ~1.5 s after the unit shows (the replay's own plays are aligned to them by the
    # offset, so they carry the same lag): moved OWN_LAG earlier, with the replay's landing spot
    own = sorted(({**o, "t": o["t"] - OWN_LAG} for o in own_spots(run["own_plays"], pr["own"], pr["off"]))
                 if pr.get("own") else run["own_plays"], key=lambda o: o["t"])
    k, ci, series, tracks = 0, 0, [], {}
    clock = run["clock"]
    for t, us in run["units"]:
        while k < len(own) and own[k]["t"] <= t:
            opp.own_play(own[k]["card"], own[k]["t"], own[k].get("x"), own[k].get("y"))
            k += 1
        while ci + 1 < len(clock) and clock[ci + 1][0] <= t:
            ci += 1
        el = clock[ci][1] + (t - clock[ci][0]) if clock and clock[ci][0] <= t and clock[ci][1] is not None else None
        opp.update_units(t, us, el)
        row = []
        for tr in opp.spells.slots(3):
            tracks[id(tr)] = tr
            row.append((id(tr), tr.card, tr.owner, round(tr.tti, 2), tr.land))
        series.append((t, row))
    return series, opp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("all", "tune", "held"), default="all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    db = lancedb.connect(str(ROOT / "data/lancedb"))
    cards = db.open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()
    ctype = {c["name"]: c["type"] for c in cards}
    rows, phantoms, n_games, minutes = [], [], 0, 0.0
    for f in sorted((CACHE / "_truth").glob("bot_*.json")):
        g = f.stem
        pr = json.loads(f.read_text())
        rf = CACHE / f"{g}.json"
        if pr["agree"] < 0.6 or not rf.exists() or (a.split != "all" and split_of(g) != a.split):
            continue
        run = json.loads(rf.read_text())
        if not run["clock"]:
            continue
        n_games += 1
        lo, hi = run["clock"][0][0], run["clock"][-1][0]
        minutes += (hi - lo) / 60
        series, _ = run_game(run, pr, cards)
        first = {}  # track id -> (t, card, owner, tti, land) at its first slot appearance as theirs
        for t, row in series:
            for tid, card, owner, tti, land in row:
                if owner == "opp" and tid not in first:
                    first[tid] = (t, card, tti, land)
        used = set()
        for p in pr["truth"]:
            if ctype.get(p["card"]) != "spell":
                continue
            c = p["t"] - pr["off"]
            if not lo + 1.0 <= c <= hi - 3.0:
                continue
            hits = sorted((v[0], tid) for tid, v in first.items() if v[1] == p["card"] and tid not in used
                          and c - 0.5 <= v[0] <= c + 3.0)
            r = {"game": g, "card": p["card"], "cast": round(c, 2), "found": bool(hits)}
            if hits:
                tid = hits[0][1]
                used.add(tid)
                t1, _, tti, land = first[tid]
                flight = edist(THEIR_KING, land) / FILE_SPEED.get(p["card"], 10.0) if land else None
                r.update(delay=round(t1 - c, 2), tti_pred=tti, flight_est=round(flight, 2) if flight else None,
                         lead=round(c + flight - t1, 2) if flight else None)
            rows.append(r)
        for tid, v in first.items():
            if tid not in used and lo + 1.0 <= v[0] <= hi - 3.0:
                phantoms.append({"game": g, "t": round(v[0], 2), "card": v[1]})
    print(f"{n_games} games ({minutes:.0f} min), split {a.split}; real opponent spells {len(rows)}")
    by = defaultdict(list)
    for r in rows:
        by[r["card"]].append(r)
    for card, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        f = [r for r in rs if r["found"]]
        line = f"  {card:16s} {len(f)}/{len(rs)} found ({len(f) / len(rs):.0%})"
        if f:
            d = sorted(r["delay"] for r in f)
            ahead = [r for r in f if r["lead"] is not None and r["lead"] > 0]
            leads = sorted(r["lead"] for r in f if r["lead"] is not None)
            line += (f"; cast -> slot median {stats.median(d):+.2f} s (IQR {d[len(d) // 4]:+.2f}..{d[3 * len(d) // 4]:+.2f});"
                     f" predicted time to impact then median {stats.median(r['tti_pred'] for r in f):.2f} s;"
                     f" before the estimated impact {len(ahead)}/{len(rs)} ({len(ahead) / len(rs):.0%}),"
                     f" lead median {stats.median(leads):+.2f} s")
        print(line)
    print(f"phantom spell slots of theirs: {len(phantoms)} ({len(phantoms) / max(n_games, 1):.1f} a game), "
          f"by card: {Counter(p['card'] for p in phantoms).most_common(8)}")
    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "phantoms": phantoms}, default=str))


if __name__ == "__main__":
    main()
