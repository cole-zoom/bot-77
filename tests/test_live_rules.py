"""Playbook rules (config/playbook_rules.yaml) on synthetic states: each test is a mistake from Cole's reviews."""
import lancedb
import pytest

from bot77.live.policy import Policy
from bot77.live.state import LiveState, Unit


@pytest.fixture(scope="module")
def cards():
    return lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()


@pytest.fixture
def pol(cards):
    return Policy(cards, playbook=False)


@pytest.fixture(scope="module")
def ids(cards):
    return {c["name"]: c["id"] for c in cards}


def st(ids, hand, elixir, units=(), t=50.0):
    return LiveState(t, [ids[n] for n in hand], None, elixir, 120, 60, 1, "regulation", list(units))


def u(card, x, y, team="enemy"):
    return Unit(card, card, team, x, y, 0.9)


def test_no_idle_plays(pol, ids):
    assert pol.decide(st(ids, ["Giant", "Knight", "Archers", "Minions"], 8.0), []) is None  # "whats the point?"


def test_giant_goes_in_the_back_never_the_bridge(pol, ids):
    act = pol.decide(st(ids, ["Giant", "Knight", "Archers", "Minions"], 9.8), [])
    assert act.card == "Giant" and act.y >= 24.5  # behind the king tower


def test_wait_until_the_attacker_crosses(pol, ids):
    assert pol.decide(st(ids, ["Knight", "Mini P.E.K.K.A", "Archers", "Minions"], 6, [u("Knight", 4, 12)]), []) is None


def test_archers_alone_are_not_worth_elixir(pol, ids):
    assert pol.decide(st(ids, ["Knight", "Mini P.E.K.K.A", "Archers", "Minions"], 6, [u("Archers", 4, 18)]), []) is None


def test_mini_pekka_on_the_giant_directly(pol, ids):
    act = pol.decide(st(ids, ["Giant", "Mini P.E.K.K.A", "Arrows", "Musketeer"], 6, [u("Giant", 4, 18)]), [])
    assert act.card == "Mini P.E.K.K.A" and abs(act.x - 4) < 0.5 and abs(act.y - 18) < 0.5


def test_never_arrows_or_giant_on_a_giant(pol, ids):
    act = pol.decide(st(ids, ["Giant", "Fireball", "Arrows", "Musketeer"], 6, [u("Giant", 4, 18)]), [])
    assert act.card == "Musketeer"
    assert (act.x, act.y) == (7.0, 21.5)  # ranged: in front of the king tower, away from the Giant


def test_no_arrows_on_a_lone_knight(pol, ids):
    act = pol.decide(st(ids, ["Arrows", "Fireball", "Giant", "Archers"], 3.5, [u("Knight", 14, 18)]), [])
    assert act is None or act.card != "Arrows"


def test_arrows_on_the_goblins(pol, ids):
    gobs = [u("Spear Goblins", 3, 17), u("Spear Goblins", 3.6, 17.3), u("Spear Goblins", 4, 17.1)]
    act = pol.decide(st(ids, ["Arrows", "Giant", "Knight", "Musketeer"], 4, gobs), [])
    assert act.card == "Arrows" and abs(act.x - 3.5) < 1


def test_ground_troops_dont_answer_air(pol, ids):
    act = pol.decide(st(ids, ["Knight", "Mini P.E.K.K.A", "Archers", "Giant"], 6, [u("Minions", 4, 18)]), [])
    assert act.card == "Archers"


def test_one_answer_per_attacker(pol, ids):
    s = st(ids, ["Minions", "Knight", "Archers", "Giant"], 8, [u("Valkyrie", 4, 18)])
    act = pol.decide(s, [])
    assert act.card == "Minions"
    pol.record_own(s.t, act.card)
    later = st(ids, ["Knight", "Archers", "Giant", "Arrows"], 7, [u("Valkyrie", 4, 19)], t=s.t + 2)
    assert pol.decide(later, []) is None  # the Minions are on it


def test_support_the_giant(pol, ids):
    pol.record_own(40.0, "Giant")
    act = pol.decide(st(ids, ["Musketeer", "Knight", "Arrows", "Fireball"], 5, [u("Giant", 4, 19, "friendly")]), [])
    assert act.card == "Musketeer" and act.reason.startswith("support") and act.y > 19
