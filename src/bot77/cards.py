"""Card metadata: merge an official API snapshot with hand-curated facts, fetch card art,
and write everything to the LanceDB `cards` table."""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import yaml

MAX_LEVEL = 16

# The API card id's leading digits encode the card type.
TYPE_BY_ID_PREFIX = {"26": "troop", "27": "building", "28": "spell"}

# iconUrls key -> art variant name
ART_VARIANTS = {"medium": "normal", "evolutionMedium": "evo", "heroMedium": "hero"}

CURATED_KEYS = {"evo", "hero_ability", "champion_ability", "type", "elixir"}
EVO_KEYS = {"cycles", "stat_boost", "ability_name", "ability_description"}
ABILITY_KEYS = {"name", "cost", "description", "uses_per_deploy", "cooldown_s"}

SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("name", pa.string()),
        ("kind", pa.string()),  # "deck_card" | "tower_troop"
        ("type", pa.string()),  # troop | building | spell; null for tower troops
        ("rarity", pa.string()),
        ("elixir", pa.int32()),  # null when variable (Mirror) or for tower troops
        ("elixir_variable", pa.bool_()),
        ("starting_level", pa.int32()),
        ("api_max_level", pa.int32()),
        ("api_max_evolution_level", pa.int32()),
        ("is_champion", pa.bool_()),
        ("has_evo", pa.bool_()),
        ("has_hero", pa.bool_()),
        ("evo_cycles", pa.int32()),
        ("evo_stat_boost", pa.string()),
        ("evo_ability_name", pa.string()),
        ("evo_ability_description", pa.string()),
        ("ability_name", pa.string()),  # hero or champion ability
        ("ability_cost", pa.int32()),
        ("ability_description", pa.string()),
        ("ability_uses_per_deploy", pa.int32()),
        ("ability_cooldown_s", pa.float32()),
        ("icon_url_normal", pa.string()),
        ("icon_url_evo", pa.string()),
        ("icon_url_hero", pa.string()),
        ("art_normal", pa.binary()),
        ("art_evo", pa.binary()),
        ("art_hero", pa.binary()),
        ("snapshot", pa.string()),
    ]
)


@dataclass
class IngestReport:
    rows: list[dict]
    warnings: list[str] = field(default_factory=list)


def starting_level(api_max_level: int) -> int:
    """The API's maxLevel counts upgrades, not the level cap: commons start at 1,
    rares 3, epics 6, legendaries 9, champions 11 — all cap at 16."""
    return MAX_LEVEL - api_max_level + 1


def load_curated(path: Path, api_names: set[str]) -> dict:
    curated = yaml.safe_load(path.read_text()) or {}
    problems = []
    for name, entry in curated.items():
        if name not in api_names:
            problems.append(f"curated card not in API snapshot: {name!r}")
            continue
        for key, value in entry.items():
            if key not in CURATED_KEYS:
                problems.append(f"{name}: unknown key {key!r}")
            elif key == "evo" and set(value) - EVO_KEYS:
                problems.append(f"{name}.evo: unknown keys {set(value) - EVO_KEYS}")
            elif key.endswith("_ability") and set(value) - ABILITY_KEYS:
                problems.append(f"{name}.{key}: unknown keys {set(value) - ABILITY_KEYS}")
    if problems:
        raise ValueError("curated.yaml problems:\n  " + "\n  ".join(problems))
    return curated


def build_rows(api: dict, curated: dict, snapshot: str) -> IngestReport:
    report = IngestReport(rows=[])
    for card in api["items"]:
        report.rows.append(_deck_card_row(card, curated.get(card["name"], {}), snapshot, report))
    for troop in api.get("supportItems", []):
        report.rows.append(_tower_troop_row(troop, snapshot))
    return report


def _deck_card_row(card: dict, cur: dict, snapshot: str, report: IngestReport) -> dict:
    name, icons = card["name"], card["iconUrls"]
    is_champion = card["rarity"] == "champion"
    has_evo, has_hero = "evolutionMedium" in icons, "heroMedium" in icons

    # maxEvolutionLevel is a bitmask-like flag: 1 = evo, 2 = hero, 3 = both.
    flag = card.get("maxEvolutionLevel")
    expected = {None: (False, False), 1: (True, False), 2: (False, True), 3: (True, True)}
    if expected.get(flag) != (has_evo, has_hero):
        report.warnings.append(f"{name}: maxEvolutionLevel={flag} disagrees with icon urls")

    elixir_variable = cur.get("elixir") == "variable"
    if "elixirCost" not in card and not elixir_variable:
        report.warnings.append(f"{name}: no elixirCost in API and not marked variable")

    ability = cur.get("champion_ability") if is_champion else cur.get("hero_ability")
    ability = ability or {}
    has_ability = is_champion or has_hero
    if has_ability and (ability.get("name") is None or ability.get("cost") is None):
        report.warnings.append(f"{name}: ability name/cost not filled in")
    evo = cur.get("evo") or {}
    if has_evo and evo.get("cycles") is None:
        report.warnings.append(f"{name}: evo cycles not filled in")

    return {
        "id": card["id"],
        "name": name,
        "kind": "deck_card",
        "type": cur.get("type") or TYPE_BY_ID_PREFIX[str(card["id"])[:2]],
        "rarity": card["rarity"],
        "elixir": None if elixir_variable else card.get("elixirCost"),
        "elixir_variable": elixir_variable,
        "starting_level": starting_level(card["maxLevel"]),
        "api_max_level": card["maxLevel"],
        "api_max_evolution_level": flag,
        "is_champion": is_champion,
        "has_evo": has_evo,
        "has_hero": has_hero,
        "evo_cycles": evo.get("cycles"),
        "evo_stat_boost": evo.get("stat_boost"),
        "evo_ability_name": evo.get("ability_name"),
        "evo_ability_description": evo.get("ability_description"),
        "ability_name": ability.get("name"),
        "ability_cost": ability.get("cost"),
        "ability_description": ability.get("description"),
        "ability_uses_per_deploy": ability.get("uses_per_deploy", 1) if has_ability else None,
        "ability_cooldown_s": ability.get("cooldown_s"),
        **_icon_urls(icons),
        "snapshot": snapshot,
    }


def _tower_troop_row(troop: dict, snapshot: str) -> dict:
    return {
        "id": troop["id"],
        "name": troop["name"],
        "kind": "tower_troop",
        "type": None,
        "rarity": troop["rarity"],
        "elixir": None,
        "elixir_variable": False,
        "starting_level": starting_level(troop["maxLevel"]),
        "api_max_level": troop["maxLevel"],
        "api_max_evolution_level": None,
        "is_champion": False,
        "has_evo": False,
        "has_hero": False,
        "evo_cycles": None,
        "evo_stat_boost": None,
        "evo_ability_name": None,
        "evo_ability_description": None,
        "ability_name": None,
        "ability_cost": None,
        "ability_description": None,
        "ability_uses_per_deploy": None,
        "ability_cooldown_s": None,
        **_icon_urls(troop["iconUrls"]),
        "snapshot": snapshot,
    }


def _icon_urls(icons: dict) -> dict:
    return {f"icon_url_{variant}": icons.get(key) for key, variant in ART_VARIANTS.items()}


def fetch_art(rows: list[dict], art_dir: Path) -> None:
    """Download every art variant (cached on disk) and attach the PNG bytes to each row."""
    art_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (row, variant, row[f"icon_url_{variant}"], art_dir / f"{row['id']}_{variant}.png")
        for row in rows
        for variant in ART_VARIANTS.values()
        if row[f"icon_url_{variant}"]
    ]

    def fetch(job):
        row, variant, url, path = job
        if not path.exists():
            with urllib.request.urlopen(url, timeout=30) as resp:
                path.write_bytes(resp.read())
        return row, variant, path.read_bytes()

    for row in rows:
        for variant in ART_VARIANTS.values():
            row[f"art_{variant}"] = None
    with ThreadPoolExecutor(max_workers=16) as pool:
        for row, variant, data in pool.map(fetch, jobs):
            row[f"art_{variant}"] = data


def ingest(api_json: Path, curated_yaml: Path, db_dir: Path, art_dir: Path) -> IngestReport:
    import lancedb

    api = json.loads(api_json.read_text())
    curated = load_curated(curated_yaml, {c["name"] for c in api["items"]})
    report = build_rows(api, curated, snapshot=api_json.stem)
    fetch_art(report.rows, art_dir)

    db = lancedb.connect(db_dir)
    db.create_table("cards", pa.Table.from_pylist(report.rows, schema=SCHEMA), mode="overwrite")
    return report
