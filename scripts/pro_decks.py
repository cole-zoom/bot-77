"""The pros' decks, as engine ids, for the reinforcement learning against pro clones (option 3 step 2, 2026-10-27).

  uv run --no-sync python scripts/pro_decks.py   -> data/cache/rl/pro_decks.json   (gitignored)

One deck per player per human replay in the train / dev splits (the ones the imitation agent learned from; held-back,
test and held-out stay out), through the same mapping the imitation data used (bot77.sim.scenario._engine_deck: an
Evolution or Hero where the player played that form). Kept: 8 cards the engine has, in an order it accepts
(bot77.sim.engine.legal_deck). The frozen clone opponent has no ability heads, so a Hero plays as its plain card and a
Champion as its body without the ability (more than half the pros' decks hold one: leaving them out would skew the
pool). Duplicate decks are kept once, with how many players used them (the trainer samples by that count).
"""
import collections
import json
from pathlib import Path

from bot77.sim import cards as sc
from bot77.sim import store
from bot77.sim.engine import legal_deck
from bot77.sim.scenario import _engine_deck, _rows, side_decks

OUT = Path("data/cache/rl/pro_decks.json")


def main():
    ms = store.read("matches", "source IN ('replay_tv', 'replay_bl')")
    games = sorted(m["match_id"] for m in ms if m["pov_side"] == "bottom" and m["split"] in ("train", "dev"))
    info = sc.engine_cards()
    counts: collections.Counter = collections.Counter()
    why = collections.Counter()
    for gid in games:
        plays = _rows("plays", f"game_id = '{gid}' AND source = 'hand'", ["side", "card_id", "form", "kind"])
        for side, deck in side_decks(gid).items():
            ids, _, missing = _engine_deck(deck, [p for p in plays if p["side"] == side])
            if missing or len(ids) != 8:
                why["not 8 engine cards"] += 1
                continue
            plain = [sc.engine_id(info[c]["name"].removeprefix("Hero ")) if info[c]["is_hero"] else c for c in ids]
            if None in plain or len(set(plain)) != 8:
                why["hero without a plain card"] += 1
                continue
            ids = plain
            why["champion (no ability)"] += any(info[c]["is_champion"] for c in ids)
            try:
                counts[tuple(sorted(legal_deck(ids)))] += 1
            except Exception:  # noqa: BLE001 -- no slot order the engine accepts
                why["no legal order"] += 1
    decks = [{"cards": legal_deck(list(d)), "names": [info[c]["name"] for c in legal_deck(list(d))], "players": n}
             for d, n in counts.most_common()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(decks, indent=1))
    print(f"{len(games)} games -> {sum(counts.values())} player decks kept, {len(decks)} distinct; notes: {dict(why)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
