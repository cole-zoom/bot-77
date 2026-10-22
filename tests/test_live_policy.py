"""B v0 policy rules and tap geometry on synthetic states (no recordings, no BlueStacks)."""
import lancedb
import pytest

from bot77.layout import Layout
from bot77.live.device import FakeDevice, deploy, slot_centre
from bot77.live.policy import BRIDGE, CENTRE, ENEMY_TOWER, Action, Policy
from bot77.live.state import LiveReader, LiveState, Unit


@pytest.fixture(scope="module")
def cards():
    return lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000).to_arrow().to_pylist()


@pytest.fixture(scope="module")
def policy(cards):
    return Policy(cards)


def state(elixir, units=(), hand=None, t=50.0):
    return LiveState(t, hand or [26000021, 27000006, 28000015, 26000010], 26000084, elixir, 100, 80, 1, "regulation", list(units))


def enemy(card, x, y):
    return Unit(card, card.lower().replace(" ", "-"), "enemy", x, y, 0.9)


def test_win_condition_goes_to_bridge(policy):
    st = state(9.8)
    assert policy.place("Hog Rider", "troop", [], st) in BRIDGE.values()


def test_building_centre_without_threat(policy):
    assert policy.place("Tesla", "building", [], state(9.8)) == CENTRE


def test_spell_defends_only_on_our_half(policy):
    far = [enemy("Goblins", 3, 6), enemy("Goblins", 3.5, 6.2)]  # a group still on their side
    threat = [enemy("Knight", 12, 19)]
    st = state(5, far + threat)
    spot = policy.place("Barbarian Barrel", "spell", st.threats(), st)
    assert spot is not None and spot[1] > 15  # aimed at the threat, not the far group
    assert policy.place("Earthquake", "spell", [enemy("Knight", 12, 19)], state(5, [])) is not None


def test_spell_without_target(policy):
    st = state(9, [])  # nothing on the field
    assert policy.place("The Log", "spell", [], st) is None  # a small spell waits for a target
    assert policy.place("Earthquake", "spell", [], st) in ENEMY_TOWER.values()  # a building spell hits a tower


def test_mighty_miner_defends_in_front(policy):
    st = state(6, [enemy("Hog Rider", 4, 19)])
    x, y = policy.place("Mighty Miner", "troop", st.threats(), st)
    assert y > 15 and abs(x - 4) < 3


def test_miner_targets_tower_on_offence(policy):
    assert policy.place("Mighty Miner", "troop", [], state(9.5)) in ENEMY_TOWER.values()


def test_hog_counter_push_needs_elixir(policy):
    st = state(4, [enemy("Knight", 4, 19)])
    assert policy.place("Hog Rider", "troop", st.threats(), st) is None
    st = state(8, [enemy("Knight", 4, 19)])
    assert policy.place("Hog Rider", "troop", st.threats(), st) == BRIDGE["right"]


def test_decide_respects_elixir_and_gap(policy):
    assert policy.decide(state(3.0), []) is None  # nothing to defend, low elixir: wait
    act = policy.decide(state(9.8, t=60.0), [])
    assert isinstance(act, Action) and act.reason == "full elixir"
    policy.record_own(60.0, act.card)
    assert policy.decide(state(9.8, t=60.5), []) is None  # min gap between plays


def test_deploy_taps_land_on_slot_and_spot(cards):
    layout = Layout.load("cole_iphone_mirroring")
    reader = LiveReader(layout, cards)
    dev = FakeDevice.__new__(FakeDevice); dev.taps = []; dev.now = lambda: 0.0
    act = Action(2, "Hog Rider", 3.5, 16.2, "test", [0, 0, 1, 0])
    sx, sy, tx, ty = deploy(dev, layout, reader, act, (2340, 1080, 3))
    x0, y0, x1, y1 = layout.box("slot3", 1080, 2340)
    assert x0 <= sx <= x1 and y0 <= sy <= y1
    tile, ox, oy = reader.tile_px(1080)
    assert abs((tx - ox) / tile - 3.5) < 1e-6 and abs((ty - oy) / tile - 16.2) < 1e-6
    ax0, ay0, ax1, ay1 = layout.katacr_arena
    assert ax0 <= tx <= ax1 and ay0 <= ty <= ay1
