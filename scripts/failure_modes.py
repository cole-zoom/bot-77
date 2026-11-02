"""The bot's failure modes, from Cole's review of 7 losses (2026-11-02, data/reviews/cole/loss_review_2026-11-02.json:
31 moments, each a "should have" and a note). Cole: "can we 'classify' my answers in that review as 'failure modes'
and then I can apply them to the mined moments, or mine for those moments?"

  uv run --no-sync python scripts/failure_modes.py   # his notes by mode, and which the miner finds

Each mode has a detector in scripts/mine_moments.py (the `kind` it writes) and a number key on the review page
(scripts/build_moment_review.py). COLE maps each of his notes to its mode(s), the first the main one.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# key: (name on the page, what it means, in Cole's words where he gave them)
MODES = {
    "missed_spell": ("Missed spell", "spam or a bunch on our side that one Arrows / Fireball wins elixir on, and we "
                     "don't cast it (\"textbook arrows\"; \"Fireball is best counter\" to Barbarians)"),
    "pointless_push": ("Pointless push", "a push or support with no elixir lead, into a stacked lane or their counter, "
                       "or backing a weak push: \"WAIT\", or start the Giant in the back"),
    "spent_counter": ("Spent the counter", "used the answer to their win condition on something else, so it wasn't "
                      "in hand when the win condition came (\"save our counters to their win conditions\")"),
    "wasted_spell": ("Wasted spell", "a spell for less than its cost, on a dying push, or one that misses "
                     "(\"Why fireball? we just lost 4 elixir\")"),
    "too_early": ("Didn't wait", "defended before their troop crossed the river, so it gets value from its side "
                  "(\"wait till the musky crosses\")"),
    "bad_placement": ("Placement", "a defender dropped on top of the attacker (no time to deploy), in the wrong lane, "
                      "or too late"),
    "no_defence": ("Didn't defend / kite", "a push on our tower and we neither defend nor pull it (\"Perfect time to "
                   "kite\"; Battle Ram to distract the Mini P.E.K.K.A)"),
    "missed_finish": ("Didn't finish", "their tower low enough for a spell or a troop to take, and we don't (\"if "
                      "we're going to tower trade, we gotta finish the job\")"),
}

# Cole's 31 notes (game, t in game seconds) -> modes. Freezes at BRit 135.6 and Solande 138.1 were the live reader
# locking a card out (decisions.md 2026-11-02), not a decision.
COLE = {
    ("bot_2026-10-31_012724", 70.6): ["pointless_push", "bad_placement"],  # Giant into their Mini P.E.K.K.A; Bomber wrong lane
    ("bot_2026-10-31_012724", 71.5): ["spent_counter"],  # Mega Minion on the Valkyrie, last counter to their Battle Ram
    ("bot_2026-10-31_012724", 81.0): ["bad_placement"],  # Mini P.E.K.K.A right on top of theirs
    ("bot_2026-10-31_012724", 140.6): ["wasted_spell"],  # Arrows missed
    ("bot_2026-10-31_013343", 46.1): ["pointless_push", "spent_counter"],  # push without a lead; hold the Mega Minion
    ("bot_2026-10-31_013343", 72.4): ["pointless_push"],  # Mini P.E.K.K.A into a Tombstone
    ("bot_2026-10-31_013343", 116.6): ["missed_spell"],  # Barbarians let go
    ("bot_2026-10-31_014314", 23.8): ["wasted_spell"],  # Fireball for nothing
    ("bot_2026-10-31_014314", 87.8): ["wasted_spell", "no_defence"],  # Arrows where a kite was free
    ("bot_2026-10-31_014314", 124.5): ["spent_counter", "too_early"],  # best counter in the other lane; wait for the cross
    ("bot_2026-10-31_014314", 134.2): ["pointless_push"],  # Battle Ram for no reason
    ("bot_2026-10-31_014314", 138.1): ["no_defence"],  # desperation Bomber
    ("bot_2026-10-31_015151", 26.0): ["pointless_push"],  # Giant into a stacked lane
    ("bot_2026-10-31_015151", 46.7): ["too_early"],  # wait until things cross
    ("bot_2026-10-31_015151", 88.4): ["pointless_push"],  # Giant at the bridge without a lead
    ("bot_2026-10-31_015151", 105.5): ["pointless_push"],  # pressed at 4 elixir
    ("bot_2026-10-31_015151", 132.4): ["no_defence"],  # kite, or Fireball
    ("bot_2026-10-31_015151", 135.6): ["no_defence"],  # "we freeze": the reader bug
    ("bot_2026-10-31_025936", 22.2): ["missed_finish"],  # tower trade not finished
    ("bot_2026-10-31_025936", 62.9): ["pointless_push"],  # random play
    ("bot_2026-10-31_025936", 93.3): ["pointless_push"],  # backing a weak push
    ("bot_2026-10-31_025936", 121.7): ["missed_finish", "pointless_push"],  # Fireball the tower; Bombers wasted
    ("bot_2026-10-31_025936", 141.1): ["no_defence"],  # Battle Ram to distract the Mini P.E.K.K.A
    ("bot_2026-10-31_030246", 101.2): ["missed_spell"],
    ("bot_2026-10-31_030246", 119.2): ["missed_spell"],
    ("bot_2026-10-31_030246", 127.4): ["missed_spell"],
    ("bot_2026-10-31_030246", 131.8): ["missed_spell"],
    ("bot_2026-10-31_030246", 148.3): ["missed_spell"],
    ("bot_2026-10-31_030246", 158.5): ["wasted_spell"],  # Arrows on a dying push, the threat on the other lane
    ("bot_2026-10-31_030556", 12.1): ["missed_spell"],  # spam: Arrows + Bomber low
    ("bot_2026-10-31_030556", 18.3): ["missed_spell", "bad_placement"],  # Bomber late and in a bad spot
}
MATCH_S = 5.0  # a mined moment within this of one of Cole's counts as found


def main():
    mined = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/reviews/mined_2026-10-31/moments.json"
    review = json.loads((ROOT / "data/reviews/cole/loss_review_2026-11-02.json").read_text())
    notes = {(g["gid"], m["t"]): m for g in review["games"] for m in g.get("moments", [])}
    assert set(notes) == set(COLE), set(notes) ^ set(COLE)
    moments = json.loads(mined.read_text())["moments"] if mined.exists() else []
    print(f"Cole's {len(COLE)} notes by main mode (and the miner's hits within {MATCH_S:.0f} s):")
    for key, (name, _) in MODES.items():
        mine = [k for k, v in COLE.items() if v[0] == key]
        also = [k for k, v in COLE.items() if key in v[1:]]
        found = [k for k in mine + also if any(m["game"] == k[0] and m["kind"] == key and abs(m["t"] - k[1]) <= MATCH_S
                                              for m in moments)]
        print(f"  {name:22s} {len(mine):2d} main, {len(also)} also; miner finds {len(found)} of {len(mine) + len(also)}; "
              f"{sum(m['kind'] == key for m in moments)} mined in all")


if __name__ == "__main__":
    main()
