"""The scripted deck opponents (bot77.rl.decks, docs/decks/): each plays a whole game against a random player on our
deck without an exception, plays only its own 8 cards, and defends (it beats the random player in most games)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bot77.rl.decks  # noqa: E402,F401
import deck_bot_smoke  # noqa: E402
from bot77.rl.deck_bots import DECKS  # noqa: E402


@pytest.mark.parametrize("key", sorted(DECKS))
def test_a_deck_bot_plays_whole_games_with_its_own_cards(key):
    wins = 0
    for seed in range(3):
        r = deck_bot_smoke.game(key, seed)
        assert r["opp_plays"] >= 5, r
        assert set(r["opp_cards"]) <= set(DECKS[key].DECK), r["opp_cards"]
        wins += r["towers_bot"][0] <= 0 or sum(r["towers_bot"]) < sum(r["towers_opp"])
    assert wins >= 2, f"{key} lost to a random player in {3 - wins} of 3 games"


def test_a_heavy_deck_waits_for_the_bot_until_0_30():
    for key, cls in DECKS.items():
        if cls.OPEN_WAIT_S <= 0:
            continue
        r = deck_bot_smoke.game(key, 0, passive=True)
        assert r["first_opp_play_s"] >= cls.OPEN_WAIT_S - 0.05, (key, r["first_opp_play_s"])
