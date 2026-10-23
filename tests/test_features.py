"""The feature registry (bot77.features.registry) matches what the feature code produces: nothing unlisted, nothing
stale."""
import lancedb
import pytest

from bot77.features import (BY_NAME, FEATURES, RateTables, add_board_features, candidates, spot_features, spots,
                            summary_features)


@pytest.fixture(scope="module")
def cards():
    rows = lancedb.connect("data/lancedb").open_table("cards").search().where("kind = 'deck_card'").limit(1000) \
        .select(["id", "name", "elixir", "type"]).to_arrow().to_pylist()
    return {c["id"]: c for c in rows}


def produced(cards):
    names = {c["name"]: i for i, c in cards.items()}
    hand = [names[n] for n in ("Giant", "Knight", "Arrows", "Musketeer")]
    h = {"hand": hand, "next": names["Minions"], "elixir": 6.0, "t_match": 60.0, "multiplier": 1}
    rows = candidates(h, 60.0, [{"t": 50.0, "card": "Knight"}], [{"t": 55.0, "card": "Hog Rider"}], {}, cards, names)
    units = [{"card": "Hog Rider", "team": "opp", "x": 4.0, "y": 18.0}, {"card": "Giant", "team": "own", "x": 14.0, "y": 20.0}]
    add_board_features(rows, units, {"own_L": 1}, cards)
    for r in rows:
        r["chosen"] = 0
    RateTables.fit(rows).apply(rows)
    el = {c["name"]: c["elixir"] or 0 for c in cards.values()}
    summ = summary_features(units, {}, el, names)
    sp = [{**summ, **spot_features(s, "Knight", "troop", units, el, {}), "p_card": 0}
          for s in spots("Knight", "troop", units, [(9.0, 20.0)], el)]
    return (set(rows[0]) | set(sp[0])) - {"chosen"}  # the label, not a feature


def test_every_feature_is_registered(cards):
    missing = produced(cards) - set(BY_NAME)
    assert not missing, f"features the code makes that the registry doesn't list: {sorted(missing)}"


def test_no_stale_registry_entries(cards):
    stale = set(BY_NAME) - produced(cards)
    assert not stale, f"registry entries no code makes: {sorted(stale)}"


def test_names_unique():
    assert len(FEATURES) == len(BY_NAME)
