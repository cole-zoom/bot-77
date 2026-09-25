import json
from pathlib import Path

import pytest

from bot77.cards import build_rows, load_curated, starting_level

SNAPSHOT = max(Path("data/cards/api_snapshots").glob("*.json"))
CURATED = Path("data/cards/curated.yaml")


@pytest.fixture(scope="module")
def rows():
    api = json.loads(SNAPSHOT.read_text())
    curated = load_curated(CURATED, {c["name"] for c in api["items"]})
    report = build_rows(api, curated, snapshot=SNAPSHOT.stem)
    return {r["name"]: r for r in report.rows}, report.warnings


@pytest.mark.parametrize("api_max, start", [(16, 1), (14, 3), (11, 6), (8, 9), (6, 11)])
def test_starting_level_by_rarity(api_max, start):
    assert starting_level(api_max) == start


def test_no_structural_warnings(rows):
    _, warnings = rows
    assert [w for w in warnings if "not filled in" not in w] == []


def test_evo_hero_flags(rows):
    by_name, _ = rows
    assert (by_name["P.E.K.K.A"]["has_evo"], by_name["P.E.K.K.A"]["has_hero"]) == (True, False)
    assert (by_name["Giant"]["has_evo"], by_name["Giant"]["has_hero"]) == (False, True)
    assert (by_name["Knight"]["has_evo"], by_name["Knight"]["has_hero"]) == (True, True)


def test_deck_cards(rows):
    by_name, _ = rows
    expected = {
        "Hog Rider": ("troop", 4),
        "Earthquake": ("spell", 3),
        "Tesla": ("building", 4),
        "Cannon": ("building", 3),
        "Firecracker": ("troop", 3),
        "Skeletons": ("troop", 1),
        "Electro Spirit": ("troop", 1),
        "Mighty Miner": ("troop", 4),
        "The Log": ("spell", 2),
        "Barbarian Barrel": ("spell", 2),
    }
    for name, (type_, elixir) in expected.items():
        assert (by_name[name]["type"], by_name[name]["elixir"]) == (type_, elixir), name
    assert by_name["Mighty Miner"]["is_champion"]
    assert by_name["Tesla"]["has_evo"] and by_name["Firecracker"]["has_evo"]


def test_special_cases(rows):
    by_name, _ = rows
    assert by_name["Mirror"]["elixir"] is None and by_name["Mirror"]["elixir_variable"]
    assert by_name["Heal Spirit"]["type"] == "troop"
    assert by_name["Boss Bandit"]["ability_uses_per_deploy"] == 2
    assert by_name["Boss Bandit"]["ability_cooldown_s"] == 3
    assert by_name["Mighty Miner"]["ability_uses_per_deploy"] == 1


def test_deck_special_forms(rows):
    by_name, _ = rows
    mm = by_name["Mighty Miner"]
    assert (mm["ability_name"], mm["ability_cost"]) == ("Explosive Escape", 1)
    assert (by_name["Tesla"]["evo_cycles"], by_name["Tesla"]["evo_ability_name"]) == (2, "Pulsating Voltage")
    assert (by_name["Cannon"]["evo_cycles"], by_name["Firecracker"]["evo_cycles"]) == (2, 2)
    assert by_name["Skeleton Barrel"]["evo_stat_boost"] == "+61% Death Damage; +25% Hitpoints"
    assert by_name["Knight"]["evo_cycles"] == 2 and by_name["Knight"]["ability_name"] == "Triumphant Taunt"


def test_curated_coverage(rows):
    """Every evo / hero / champion has its curated facts, except known gaps."""
    _, warnings = rows
    assert [w for w in warnings if "not filled in" in w] == ["Ice Wizard: ability name/cost not filled in"]


def test_tower_troops(rows):
    by_name, _ = rows
    princess = by_name["Tower Princess"]
    assert princess["kind"] == "tower_troop" and princess["type"] is None
    assert sum(r["kind"] == "tower_troop" for r in by_name.values()) == 4


def test_curated_rejects_unknown_card(tmp_path):
    bad = tmp_path / "curated.yaml"
    bad.write_text("Not A Card:\n  evo: {cycles: 2}\n")
    with pytest.raises(ValueError, match="Not A Card"):
        load_curated(bad, {"Knight"})
