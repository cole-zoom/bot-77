"""Training rows from the new tables (2026-10-22): `moments` + `plays` -> the rows each model trains on, built with the
same feature code the live bot uses. scripts/check_moments.py checks A's rows against the old builder.
"""
from __future__ import annotations

from collections import defaultdict

from bot77.features.hand import candidates

# The bot logs session time; its screen recording starts with the game but ~1-2 s after the logged start (measured on 4
# games by reading the clock in the recording, 2026-10-22). Video time of a bot play = t - game start - BOT_VIDEO_LAG_S.
BOT_VIDEO_LAG_S = 1.5
HUMAN_REPLAYS = "source IN ('replay_tv', 'replay_bl')"  # replay_bot's bottom player is the bot: never imitation data


def play_history(db) -> dict[tuple[str, str], list[dict]]:
    """(game_id, side) -> that player's plays from the hand, in order ({"t", "card"}; abilities as "__ability")."""
    out = defaultdict(list)
    for p in db.open_table("plays").search().where("source = 'hand'").select(["game_id", "side", "t", "card", "kind"]) \
            .limit(10**7).to_arrow().to_pylist():
        out[(p["game_id"], p["side"])].append({"t": p["t"], "card": "__ability" if p["kind"] == "ability" else p["card"]})
    for v in out.values():
        v.sort(key=lambda p: p["t"])
    return out


def card_rows(db, cards: dict[int, dict], where: str = HUMAN_REPLAYS) -> list[dict]:
    """A's rows: one per (play moment, card in hand), "chosen" = the card played. Same rows as bot77.moves.decision_rows
    on events + hud_states (checked: identical on all 96,568 rows)."""
    names = {c["name"]: i for i, c in cards.items()}
    hist = play_history(db)
    moms = db.open_table("moments").search().where(f"kind = 'play' AND {where}").select(
        ["game_id", "side", "split", "league", "t", "t_match", "decision", "hand", "next", "elixir", "multiplier"]) \
        .limit(10**7).to_arrow().to_pylist()
    rows = []
    for m in moms:
        own = hist[(m["game_id"], m["side"])]
        opp = hist[(m["game_id"], "top" if m["side"] == "bottom" else "bottom")]
        k = m["decision"]
        last = {}
        for p in own[:k]:
            if p["card"] and names.get(p["card"]) is not None:
                last[names[p["card"]]] = p["t"]
        h = {"hand": m["hand"], "next": m["next"], "elixir": m["elixir"], "t_match": m["t_match"], "multiplier": m["multiplier"]}
        cid = names[own[k]["card"]]
        for r in candidates(h, m["t"], own[:k], opp, last, cards, names):
            rows.append({**r, "decision": k, "t": m["t"], "chosen": int(r["cand_card"] == cid), "match_id": m["game_id"],
                         "side": m["side"], "split": m["split"], "league": m["league"] if m["league"] is not None else -1,
                         "game": f"{m['game_id']}:{m['side']}"})
    return rows


def timing_rows(db, cards: dict[int, dict], where: str = HUMAN_REPLAYS) -> list[dict]:
    """A v3's rows (card + timing), all in one list (small selections; training streams timing_moments)."""
    return [r for rows in timing_moments(db, cards, where) for r in rows]


def timing_moments(db, cards: dict[int, dict], where: str = HUMAN_REPLAYS):
    """A v3's rows, one moment at a time: for every play and could-have-played (idle) moment, the 4 cards in hand plus a
    "wait" candidate, with hand / history / board / timing features. chosen = the card played (play moments) or wait
    (idle moments)."""
    import bisect

    from bot77.features.board import add_board_features
    from bot77.features.hand import timing_context, wait_row
    names = {c["name"]: i for i, c in cards.items()}
    hist = play_history(db)
    played = {l["moment_id"]: l["action"] for l in db.open_table("labels").search().where(
        "source = 'replay' AND action != 'wait'").select(["moment_id", "action"]).limit(10**7).to_arrow().to_pylist()}
    moms = db.open_table("moments").search().where(where).select(
        ["moment_id", "game_id", "side", "split", "league", "t", "t_match", "kind", "hand", "next", "elixir", "multiplier",
         "units", "towers"]).limit(10**7).to_arrow().to_pylist()
    times = {k: [p["t"] for p in v] for k, v in hist.items()}
    for m in moms:
        own_all = hist.get((m["game_id"], m["side"]), [])  # a side can have waits but no plays read
        own = own_all[:bisect.bisect_left(times.get((m["game_id"], m["side"]), []), m["t"])]
        opp = hist.get((m["game_id"], "top" if m["side"] == "bottom" else "bottom"), [])
        last = {}
        for p in own:
            if p["card"] and names.get(p["card"]) is not None:
                last[names[p["card"]]] = p["t"]
        h = {"hand": m["hand"], "next": m["next"], "elixir": m["elixir"], "t_match": m["t_match"], "multiplier": m["multiplier"]}
        cand = candidates(h, m["t"], own, opp, last, cards, names)
        units = [{"card": u["card"], "team": u["team"], "x": u["x"], "y": u["y"]} for u in m["units"]]
        add_board_features(cand, units, m["towers"] or {}, cards)
        for r in cand:
            r["cand_is_wait"] = 0.0
        cand.append(wait_row(cand[0]))
        tc = timing_context(h, cards)
        target = names.get(played.get(m["moment_id"])) if m["kind"] == "play" else None
        for r in cand:
            r.update(tc)
            r.update({"chosen": int((r["cand_is_wait"] == 1.0) if m["kind"] == "idle" else (r["cand_card"] == target)),
                      "moment_id": m["moment_id"], "kind": m["kind"], "split": m["split"],
                      "league": m["league"] if m["league"] is not None else -1, "t": m["t"],
                      "game": f"{m['game_id']}:{m['side']}", "decision": m["moment_id"]})
        yield cand
