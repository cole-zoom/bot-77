"""The live reader's learned deck at 0.1 s decisions (2026-11-02): a card sliding into its slot must not reset the deck,
and a relearned deck must not lock out the card at the back of the cycle. run40v4 (80 games): 52 had a slot blank for
79 s on average, because the limits were 15 and 90 reads, set at a read a second."""
import lancedb
import numpy as np
import pytest

from bot77.layout import Layout
from bot77.live.state import LiveReader
from bot77.readers.clock import ClockRead
from bot77.readers.hand import SlotRead

DECK = [26000003, 26000036, 26000018, 26000039, 26000013, 27000001, 28000000, 28000001]  # Giant first
GIANT = DECK[0]


@pytest.fixture(scope="module")
def cards():
    return lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()


def reader(cards):
    r = LiveReader(Layout.load("bluestacks_live"), cards)
    r.fast = True
    r.deck, r.reader = list(DECK), r.full.restrict(set(DECK))
    r._read_clock = lambda frame, t: ClockRead("regulation", int(180 - t), 1, int(t))
    return r


def hand(*slots):
    """A fresh hand read: card ids, or 'u' for an unreadable slot."""
    d = {f"slot{k + 1}": SlotRead("unknown") if c == "u" else SlotRead("card", c) for k, c in enumerate(slots[:4])}
    d["next"] = SlotRead("card", slots[4])
    return d


def run(r, reads, t0, t1, dt=0.1):
    """Feed one read per dt; a new dict only when `reads(t)` changes, as fast mode hands back its cached read."""
    t, last, key = t0, None, None
    frame = np.zeros((1920, 1080, 3), np.uint8)
    while t < t1 - 1e-9:
        k = reads(t)
        if k != key:
            last, key = hand(*k), k
        r._read_hand = lambda f, h=last: h
        st = r.read(frame, t)
        t += dt
    return st


def test_card_sliding_in_keeps_the_deck(cards):
    r = reader(cards)
    # Giant just played; its slot unreadable for 2 s as the next card slides in (the old limit: 15 reads = 1.5 s)
    run(r, lambda t: (28000001, 26000036, "u", 28000000, 27000001), 10.0, 12.0)
    assert r.deck == DECK


def test_relearned_deck_keeps_the_card_at_the_back(cards):
    r = reader(cards)
    run(r, lambda t: (28000001, 26000036, "u", 28000000, 27000001), 10.0, 26.0)  # 16 s unreadable: a reset
    assert r.reader is r.full
    # then 10 s of reads showing only 7 cards (the Giant at the back of the cycle): the deck we had comes back
    cycle = [(28000001, 26000013, 26000018, 28000000, 27000001), (28000001, 26000013, 26000039, 28000000, 26000036),
             (28000001, 26000036, 27000001, 28000000, 26000018)]
    run(r, lambda t: cycle[int(t) % 3], 26.0, 36.0)
    assert sorted(r.deck) == sorted(DECK) and GIANT in {t.card_id for t in r.reader.templates}


def test_seven_card_deck_needs_90_seconds(cards):
    r = reader(cards)
    r.deck, r.reader, r.last_deck = [], r.full, []  # a first game: nothing learned yet
    cycle = [(28000001, 26000013, 26000018, 28000000, 27000001), (28000001, 26000013, 26000039, 28000000, 26000036),
             (28000001, 26000036, 27000001, 28000000, 26000018)]
    run(r, lambda t: cycle[int(t) % 3], 0.0, 30.0)  # 300 reads, 30 s: the old code locked 7 cards in after 90 reads
    assert r.deck == []


def test_a_slot_just_played_is_not_playable_until_the_play_shows(cards):
    """run20cyc game 1: the tracker kept the Goblin Hut in its slot while the next card slid in, and the bot tapped it
    three times in 2 s. After a tap the slot is blank until the tracker sees that slot's play, or 2.5 s pass."""
    r = LiveReader(Layout.load("bluestacks_live"), cards, deck=DECK)
    r.fast = True
    r._read_clock = lambda frame, t: ClockRead("regulation", int(180 - t), 1, int(t))
    import bot77.live.state as S
    el = {"v": 8.0}
    S.read_elixir = lambda frame, layout: type("E", (), {"elixir": el["v"]})()
    hand0 = (DECK[0], DECK[1], DECK[2], DECK[3], DECK[4])
    run(r, lambda t: hand0, 0.0, 1.0)
    assert r.tracker.started
    r.played(0, 1.0)
    el["v"] = 3.0  # the Giant (5) left the bar
    st = run(r, lambda t: ("u", DECK[1], DECK[2], DECK[3], DECK[5]), 1.0, 2.0)  # slot 1 sliding: unreadable
    assert st.hand[0] is None  # in transit: not playable
    st = run(r, lambda t: (DECK[4], DECK[1], DECK[2], DECK[3], DECK[5]), 2.0, 2.5)  # the Bomber (Next) settles
    assert st.hand[0] == DECK[4]
    r.played(1, 2.5)  # a tap that doesn't land: nothing changes on screen
    st = run(r, lambda t: (DECK[4], DECK[1], DECK[2], DECK[3], DECK[5]), 2.5, 4.0)
    assert st.hand[1] is None
    st = run(r, lambda t: (DECK[4], DECK[1], DECK[2], DECK[3], DECK[5]), 4.0, 5.2)
    assert st.hand[1] == DECK[1]  # after 2.5 s it's playable again
