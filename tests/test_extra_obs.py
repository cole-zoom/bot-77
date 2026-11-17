"""The new input block (bot77.live.extra_obs) on a scripted board: their cycle, elixir and a Fireball in flight."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "clash-royale-env"))

from bot77.live import extra_obs as E  # noqa: E402
from bot77.sim import cards as sc  # noqa: E402

THEIRS = ["Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Skeletons", "Ice Spirit", "Knight", "Archers"]
OURS = ["Giant", "Battle Ram", "Mini P.E.K.K.A", "Mega Minion", "Bomber", "Goblin Hut", "Fireball", "Arrows"]
TYPES = {"Cannon": "building", "Goblin Hut": "building", "Fireball": "spell", "Arrows": "spell"}
CARDS = [{"name": n, "type": TYPES.get(n, "troop"), "elixir": 3} for n in THEIRS + OURS]


def run(t_end=95.0):
    opp = E.OppInputs(CARDS, OURS)
    t = 0.0
    for k, card in enumerate(THEIRS + THEIRS[:2]):  # their 8, then two again (each came back after 4 others)
        t0 = 3.0 + 9.5 * k
        while t < t0 + 1.0:
            us = [(card, "enemy", 9.0, 8.0, 0.9)] if t >= t0 else []
            opp.update_units(t, us, t)
            t = round(t + 0.1, 2)
    while t < t_end:  # their Fireball, thrown from their King, seen twice on its way down
        us = [("Fireball", "enemy", 8.6, 6.0, 0.8)] if abs(t - 93.0) < 0.05 else \
             [("Fireball", "enemy", 8.0, 9.0, 0.8)] if abs(t - 93.3) < 0.05 else []
        us = us + [("Giant", "friendly", 7.5, 15.0, 0.9)]
        opp.update_units(t, us, t)
        t = round(t + 0.1, 2)
    return opp, t


def test_cycle_hand_and_next():
    opp, t = run(92.0)
    assert [p["card"] for p in opp.plays] == THEIRS + THEIRS[:2]
    b = opp.block(t)
    assert b.shape == (E.EXTRA_DIM,) and b[6] == 1.0 and b[4] == 1.0 and b[2] == 1.0
    hold = {sc.our_name(int(e)) for e in np.nonzero(b[E.CAN_HOLD:E.CAN_HOLD + E.NCARD])[0]}
    # last 4 different plays: Musketeer, Hog Rider, Archers, Knight -> hand: the other four
    assert hold == {"Cannon", "Ice Golem", "Skeletons", "Ice Spirit"}
    back = [sc.our_name(int(e)) for e in np.nonzero(b[E.COMES_BACK:E.COMES_BACK + E.NCARD])[0]]
    assert back == ["Knight"]
    assert 0 < b[0] < 1.0


def test_fireball_in_flight():
    opp, t = run(93.4)
    b = opp.block(t)
    s = b[E.SPELLS:E.SPELLS + E.SLOT_DIM]
    assert s[0] == 1.0 and s[1] == 1.0  # present, Fireball
    assert s[12] == 1.0  # theirs
    assert s[18] > 0  # still flying
    assert s[17] < s[15]  # lands lower in the engine frame (toward us) than where it is now
    assert E.decode_block(b)["spells_in_flight"][0]["type"] == "Fireball"


def test_fit_obs():
    a = np.ones((3, E.BASE_DIM), np.float32)
    assert E.fit_obs(a, E.BASE_DIM + E.EXTRA_DIM)[:, E.BASE_DIM:].sum() == 0
    assert E.fit_obs(E.fit_obs(a, E.BASE_DIM + 5), E.BASE_DIM).shape == a.shape
