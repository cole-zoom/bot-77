"""Per-unit mechanics for the simulator engine from the game client's own logic tables -> the fork's
include/core/ClientMechanics.h (2026-10-29, sim fidelity P2: projectiles, bodies, towers, King activation).

  uv run --no-sync python scripts/sim_client_mechanics.py [--out path/to/ClientMechanics.h]

Source: data/external/cr_client/logic (the installed client's tables, gitignored; Cole's 2026-10-28 exception for the
stat audit). Card -> unit as scripts/audit_client_stats.py and scripts/sim_load_times_client.py do it; Evolutions and
Heroes use their base card's unit. Values are written in the client's own units (speeds in its tiles-per-minute
unit, radii in thousandths of a tile, times in ms); the engine converts them (CardRegistry.h clientProjectileSpeed:
the same conversion as a troop's Speed).

Per card / spawned unit:
  - the unit's projectile: Speed and Homing (absent = false: Bomber, Bowler, Mortar, Executioner... aim at where the
    target stood at launch);
  - the unit's CollisionRadius and Mass.
Towers (buildings.toml / characters/princesstower.toml, king_tower.toml): projectile Speed, CollisionRadius, and the
King's activation (ACTION.WaitForKingTowerActivation -> OnActivateAction ActionDuration). Globals (globals.csv):
EXTRA_SIGHT_RANGE_TO_CROWN_TOWERS.
"""
import argparse
import csv
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_client_stats import CLIENT_NAME, FIRST_LEVEL, LOGIC, UNIT_OF, UNIT_OVERRIDE, norm, units  # noqa: E402

from bot77.sim import cards as sc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="third_party/clash-royale-env/include/core/ClientMechanics.h")
a = ap.parse_args()

chars, projs = units()
cards = {}
for r in csv.DictReader(open(LOGIC / "spells_characters.csv")):
    if r["Name"] and r["SummonCharacter"] and r["Rarity"] in FIRST_LEVEL:
        cards[norm(r["Name"])] = r
for f in ("spells_buildings.csv", "spells_hero_form.csv"):
    for r in csv.DictReader(open(LOGIC / f)):
        if r.get("Name") and r.get("SummonCharacter") and r.get("Rarity") in FIRST_LEVEL:
            cards.setdefault(norm(r["Name"]), r)
for r in csv.DictReader(open(LOGIC / "spells_characters.csv")):
    if r["Name"] and not r["SummonCharacter"] and r["Rarity"] in FIRST_LEVEL and r["Name"] in UNIT_OF:
        cards[norm(r["Name"])] = {**r, "SummonCharacter": UNIT_OF[r["Name"]]}


def proj_of(unit: dict) -> tuple[int, bool] | None:
    p = projs.get(unit.get("Projectile") or "")
    if not p or not p.get("Speed"):
        return None
    return int(p["Speed"]), bool(p.get("Homing", False))


rows, missing, by_name = [], [], {}
info = sc.engine_cards()
for eid, c in sorted(info.items()):
    if c["is_spell"]:
        continue
    name = c["name"]
    base = name[5:] if c["is_hero"] and name.startswith("Hero ") else name
    card = cards.get(norm(base)) or cards.get(norm(CLIENT_NAME.get(base, "")))
    unit_name = UNIT_OVERRIDE.get(base, card["SummonCharacter"]) if card else None
    unit = chars.get(unit_name) if unit_name else None
    if unit is None:
        missing.append(f"{eid} {name}")
        continue
    pr = proj_of(unit)
    rad = int(unit.get("CollisionRadius", -1))
    mass = int(unit.get("Mass", -1))
    tag = " (Evolution: base unit)" if c["evolution"] else (" (Hero: base unit)" if c["is_hero"] else "")
    rows.append((eid, pr, rad, mass, f"{name}: {unit_name}{tag}" + (f", {unit['Projectile']}" if pr else "")))
    if not c["evolution"] and not c["is_hero"]:
        by_name.setdefault(name, (pr, rad, mass))

# bodies a card spawns carry the playable card of the same name's values (as LoadTimes.h); the client's spawned
# units that have no playable card of their name, by the engine's name for them
SPAWNED = {"Spear Goblins": "SpearGoblin", "Barbarians": "Barbarian", "Skeletons": "Skeleton", "Goblins": "Goblin",
           "Goblin Brawler": "GoblinBrawler", "Bats": "Bat"}
for ename, cname in SPAWNED.items():
    u = chars.get(cname)
    if u and ename not in by_name:
        by_name[ename] = (proj_of(u), int(u.get("CollisionRadius", -1)), int(u.get("Mass", -1)))

pt, kt = chars["PrincessTower"], chars["KingTower"]
pt_speed, kt_speed = proj_of(pt)[0], proj_of(kt)[0]
king = tomllib.loads((LOGIC / "characters" / "king_tower.toml").read_text())
act = king["ACTION"]["WaitForKingTowerActivation"]["OnActivateAction"]["ActionDuration"]
glob = {r["Name"]: r for r in csv.DictReader(open(LOGIC / "globals.csv")) if r.get("Name")}
crown_sight = int(glob["EXTRA_SIGHT_RANGE_TO_CROWN_TOWERS"]["NumberValue"])


def sw(fn: str, ret: str, cases: list[tuple[int, str, str]], default: str) -> str:
    body = "\n".join(f"        case {eid}: return {v};  // {cmt}" for eid, v, cmt in cases)
    return f"""inline {ret} {fn}(int cardId) {{
    switch (cardId) {{
{body}
        default: return {default};
    }}
}}"""


speed_cases = [(eid, str(pr[0]), cmt) for eid, pr, rad, mass, cmt in rows if pr]
homing_cases = [(eid, "true" if pr[1] else "false", cmt) for eid, pr, rad, mass, cmt in rows if pr]
rad_cases = [(eid, str(rad), cmt) for eid, pr, rad, mass, cmt in rows if rad > 0]
mass_cases = [(eid, str(mass), cmt) for eid, pr, rad, mass, cmt in rows if mass > 0]


def by_name_fn(fn: str, ret: str, idx: int, fmt, default: str) -> str:
    lines = []
    for n, v in sorted(by_name.items()):
        x = v[idx]
        if x is None or (isinstance(x, int) and x <= 0):
            continue
        lines.append(f'    if (name == "{n}") return {fmt(x)};')
    return f"inline {ret} {fn}(const std::string& name) {{\n" + "\n".join(lines) + f"\n    return {default};\n}}"


Path(a.out).write_text(f"""#pragma once
#include <string>

// Per-unit mechanics from the game client's logic tables, in the client's own units. Generated by bot-77's
// scripts/sim_client_mechanics.py (characters/*.toml, buildings.toml, projectiles.toml, globals.csv; 2026-10-29).
// Evolutions and Heroes use their base card's unit. Not matched (no entry, engine defaults): {', '.join(missing) or 'none'}.
// Used by CombatEntity / Projectile / Tower behind CombatEntity's client-mechanics switches.

// Crown towers: projectile Speed (TowerPrincessProjectile / KingProjectile), CollisionRadius (PrincessTower /
// KingTower), and the King's wake-up: ActionDuration of WaitForKingTowerActivation's OnActivateAction (the King
// shoots this long after the trigger).
inline constexpr int CLIENT_PRINCESS_TOWER_PROJECTILE_SPEED = {pt_speed};
inline constexpr int CLIENT_KING_TOWER_PROJECTILE_SPEED = {kt_speed};
inline constexpr int CLIENT_PRINCESS_TOWER_COLLISION_RADIUS = {int(pt['CollisionRadius'])};
inline constexpr int CLIENT_KING_TOWER_COLLISION_RADIUS = {int(kt['CollisionRadius'])};
inline constexpr int CLIENT_KING_ACTIVATION_MS = {int(act)};
// globals.csv: a troop sees a crown tower this much further than its SightRange.
inline constexpr int CLIENT_EXTRA_SIGHT_RANGE_TO_CROWN_TOWERS = {crown_sight};

// The unit's projectile Speed (client units; CardRegistry.h clientProjectileSpeed converts). -1 = no projectile.
{sw("clientProjectileSpeedForCard", "int", speed_cases, "-1")}

// Whether that projectile homes (the client's Homing; absent = false: it lands where the target stood at launch).
{sw("clientProjectileHomingForCard", "bool", homing_cases, "true")}

// The unit's CollisionRadius, thousandths of a tile. -1 = none.
{sw("clientCollisionRadiusForCard", "int", rad_cases, "-1")}

// The unit's Mass. -1 = none (buildings).
{sw("clientMassForCard", "int", mass_cases, "-1")}

// Bodies a card spawns carry no card id: the playable card of the same name's, else the client's spawned unit.
{by_name_fn("clientProjectileSpeedForUnitName", "int", 0, lambda x: x[0], "-1")}
{by_name_fn("clientProjectileHomingForUnitName", "bool", 0, lambda x: "true" if x[1] else "false", "true")}
{by_name_fn("clientCollisionRadiusForUnitName", "int", 1, str, "-1")}
{by_name_fn("clientMassForUnitName", "int", 2, str, "-1")}
""")
print(f"{len(rows)} cards -> {a.out}; towers {pt_speed}/{kt_speed}, radii {pt['CollisionRadius']}/{kt['CollisionRadius']}, "
      f"King activation {act} ms, crown sight +{crown_sight}; not matched: {missing}")
