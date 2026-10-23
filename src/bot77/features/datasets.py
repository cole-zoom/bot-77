"""Training rows from the new tables (2026-10-22): `moments` + `plays` -> the rows each model trains on, built with the
same feature code the live bot uses. scripts/check_moments.py checks A's rows against the old builder.
"""
from __future__ import annotations

from collections import defaultdict

from bot77.features.hand import candidates

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
