"""Smoke test for the scripted deck opponents (bot77.rl.decks): full games against a random legal player on our deck.
  uv run --no-sync python scripts/deck_bot_smoke.py [--deck hog26] [--games 3]
Prints each game's result, the opponent's plays per rule and any dropped plays."""
import argparse
import collections

import numpy as np

import clash_royale_env as cre
import bot77.rl.decks  # noqa: F401
from bot77.rl.deck_bots import DECKS, OUR_DECK, card_id


def game(deck_key: str, seed: int, verbose=False, passive=False):
    """passive: our side never plays (the heavy decks' opening wait)."""
    rng = np.random.default_rng(seed)
    ours = [card_id(n) for n in OUR_DECK]
    bot = DECKS[deck_key](seed)
    none = cre.TowerTroopType.NONE
    g = cre.ClashRoyaleEnv(ours, bot.deck_ids, 3600, none, none)
    g.seed(seed)
    g.reset()
    bot.bind(g)
    plays = 0
    refused = []
    while not g.is_game_over() and g.get_current_tick() < 3600:
        k0, x0, y0 = 4, 0.0, 0.0
        if not passive and (g.get_elixir() >= 9 or rng.random() < 0.02):  # a random affordable card at a random legal tile
            hand = list(g.get_hand())
            ok = [i for i, c in enumerate(hand) if g.get_elixir() >= bot.cost.get(bot.our.get(c, ""), 99)]
            if ok:
                i = int(rng.choice(ok))
                for _ in range(30):
                    x, y = float(rng.integers(0, 18)), float(rng.integers(0, 15))
                    if hand[i] in (card_id("Fireball"), card_id("Arrows")):
                        y = float(rng.integers(15, 33))
                    if g.is_valid_placement(hand[i], x, y, 0):
                        k0, x0, y0 = i, x, y
                        break
        mv = bot.tick()
        k1, x1, y1 = mv if mv else (4, 0.0, 0.0)
        hand0, sp0, osp0 = list(g.get_hand()), g.get_elixir_spent(0), g.get_elixir_spent(1)
        dbg = (round(g.get_elixir_for_team(1), 2), list(g.get_hand_for_team(1)), k1, (x1, y1), k0, (x0, y0)) if mv else None
        g.step_self_play(k0, x0, y0, k1, x1, y1, 1)
        if k1 < 4 and g.get_elixir_spent(1) <= osp0 + 0.5:
            refused.append((round(g.get_current_tick() / 10, 1), bot.mine[-1], dbg,
                            [(u.team, u.name, round(u.x, 1), round(u.y, 1)) for u in g.get_units() if not u.tower]))
        if k0 < 4 and g.get_elixir_spent(0) > sp0 + 0.5:
            bot.note_play(hand0[k0], x0, y0)
            plays += 1
        if verbose:
            for e in bot.events:
                print(f"  {e['t']:6.1f} {e['rule']:8s} {e['text']}")
        bot.events.clear()
    tw = [[max(0, g.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)]
    return {"ticks": g.get_current_tick(), "towers_bot": tw[0], "towers_opp": tw[1], "bot_plays": plays,
            "opp_plays": len(bot.mine), "first_opp_play_s": bot.mine[0][0] if bot.mine else None, "refused": refused, "fired": bot.fired,
            "opp_cards": collections.Counter(c for _, c, _, _ in bot.mine)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="hog26")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    for s in range(a.games):
        print(s, game(a.deck, s, a.v))
