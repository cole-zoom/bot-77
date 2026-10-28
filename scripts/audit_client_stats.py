"""Every engine card against the game's own stat files, at level 11 (2026-10-28, Cole: "audit this with the real game").

  uv run --no-sync python scripts/audit_client_stats.py [--card "Suspicious Bush"]

Source: the installed client's logic tables, pulled from BlueStacks over ADB (split_install_time_asset_pack.apk ->
assets/csv_logic, LZMA with Supercell's 4-byte size header) into data/external/cr_client/logic (gitignored: game files
never go in the repo). Cole lifted the "no client files" ground rule for this audit (decisions.md, 2026-10-28).

Card -> unit: spells_characters.csv (SummonCharacter, Rarity, ManaCost); unit stats: characters/*.toml,
characters_base.toml, buildings.toml (+ their PROJECTILE sections). A unit's stats are at the first level of the
rarity its own section names (most say Common, whatever the card's rarity): level 11 = x PowerLevelMultiplier
(rarities.csv) for the upgrades from that first level (Common 1, Rare 3, Epic 6, Legendary 9, Champion 11), as the
engine's cards are at level 11.

Compared: hitpoints, damage (the unit's or its projectile's), hit speed, range, sight, speed, deploy time, load time,
splash radius (AreaDamageRadius or the projectile's Radius), cost. Engine values via clash_royale_env.get_card_stats.
"""
import argparse
import csv
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGIC = ROOT / "data/external/cr_client/logic"
FIRST_LEVEL = {"Common": 1, "Rare": 3, "Epic": 6, "Legendary": 9, "Champion": 11}
# engine card name -> the client's internal card name (spells_*.csv Name), where they differ
CLIENT_NAME = {
    "Archers": "Archer", "Elite Barbarians": "AngryBarbarians", "Dart Goblin": "BlowdartGoblin",
    "Lumberjack": "RageBarbarian", "Executioner": "AxeMan", "Ice Golem": "IceGolemite", "Royal Ghost": "Ghost",
    "Bandit": "Assassin", "Night Witch": "DarkWitch", "Sparky": "ZapMachine", "Magic Archer": "EliteArcher",
    "Flying Machine": "DartBarrell", "Mother Witch": "WitchMother", "Cannon Cart": "MovingCannon",
    "Furnace": "FirespiritHut", "Ice Spirit": "IceSpirits", "Fire Spirit": "FireSpirits", "Heal Spirit": "Heal_hero",
    "Guards": "SkeletonWarriors", "Zappies": "MiniSparkys", "Rune Giant": "GiantBuffer",
    "Skeleton Barrel": "SkeletonBalloon", "Spirit Empress": "MergeMaiden_Normal",
}
# client cards with an empty SummonCharacter -> the CHARACTER section that is the card's unit
UNIT_OF = {"IceWizard": "IceWizard", "ElectroWizard": "ElectroWizard", "ThreeMusketeers": "ThreeMusketeer_Rework"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def multipliers() -> dict[str, list[int]]:
    rows = list(csv.reader(open(LOGIC / "rarities.csv")))
    i = rows[0].index("PowerLevelMultiplier")
    tab, cur = {}, None
    for r in rows[2:]:
        if r[0]:
            cur = r[0]
            tab[cur] = []
        if cur and len(r) > i and r[i]:
            tab[cur].append(int(r[i]))
    return tab


def level11(rarity: str, tab) -> float:
    ups = 11 - FIRST_LEVEL.get(rarity, 1)
    return 1.0 if ups <= 0 else tab[rarity][ups - 1] / 100.0


def units() -> tuple[dict, dict]:
    """{name: CHARACTER / BUILDING section}, {name: PROJECTILE section} from every logic TOML."""
    chars, projs = {}, {}
    files = list((LOGIC / "characters").glob("*.toml")) + [LOGIC / "characters_base.toml", LOGIC / "buildings.toml",
                                                          LOGIC / "projectiles.toml"]
    for f in files:
        try:
            d = tomllib.loads(f.read_text(errors="replace"))
        except Exception:  # noqa: BLE001 -- a few files use syntax tomllib rejects; skip them, reported below
            print(f"  (could not parse {f.name})", file=sys.stderr)
            continue
        for kind, sec in d.items():
            if not isinstance(sec, dict):
                continue
            for name, body in sec.items():
                if not isinstance(body, dict):
                    continue
                if kind in ("CHARACTER", "BUILDING"):
                    chars[name] = body
                elif kind == "PROJECTILE":
                    projs[name] = body
    return chars, projs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=None)
    ap.add_argument("--tol", type=float, default=0.06, help="relative difference to report")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
    import clash_royale_env as cre

    from bot77.sim import cards as sc
    tab = multipliers()
    chars, projs = units()
    cards = {}
    for r in csv.DictReader(open(LOGIC / "spells_characters.csv")):
        if r["Name"] and r["SummonCharacter"] and r["Rarity"] in FIRST_LEVEL:
            cards[norm(r["Name"])] = r
    for f in ("spells_buildings.csv", "spells_hero_form.csv"):
        for r in csv.DictReader(open(LOGIC / f)):
            if r.get("Name") and r.get("SummonCharacter") and r.get("Rarity") in FIRST_LEVEL:
                cards.setdefault(norm(r["Name"]), r)  # hero rows only fill cards missing above (Heal Spirit)
    # cards whose client row has no SummonCharacter: the unit section named after the card
    for r in csv.DictReader(open(LOGIC / "spells_characters.csv")):
        if r["Name"] and not r["SummonCharacter"] and r["Rarity"] in FIRST_LEVEL and r["Name"] in UNIT_OF:
            cards[norm(r["Name"])] = {**r, "SummonCharacter": UNIT_OF[r["Name"]]}
    report, missing = [], []
    for eid, info in sorted(sc.engine_cards().items()):
        if info["evolution"] or info["is_hero"] or info["is_spell"]:
            continue
        name = info["name"]
        if a.card and norm(a.card) != norm(name):
            continue
        card = cards.get(norm(name)) or cards.get(norm(CLIENT_NAME.get(name, "")))
        if card is None:
            missing.append(name)
            continue
        unit = chars.get(card["SummonCharacter"])
        if unit is None:
            missing.append(f"{name} (unit {card['SummonCharacter']})")
            continue
        # the unit's own Rarity field scales it: the client stores units at their level-1 value under the Common
        # table even for Rare / Epic cards (Mini P.E.K.K.A: 543 x 2.56 = 1390 at level 11)
        m = level11(unit.get("Rarity", card["Rarity"]), tab)
        proj = projs.get(unit.get("Projectile", ""), {})
        game = {
            "hp": round(unit.get("Hitpoints", 0) * m),
            "damage": round((unit.get("Damage") or proj.get("Damage") or 0) * m),
            "hit_speed_s": unit.get("HitSpeed", 0) / 1000,
            "range": unit.get("Range", 0) / 1000,
            "sight": unit.get("SightRange", 0) / 1000,
            "speed": unit.get("Speed", 0),
            "deploy_s": unit.get("DeployTime", 1000) / 1000,
            "splash": max(unit.get("AreaDamageRadius", 0), proj.get("Radius", 0) if proj.get("AoeToGround") or proj.get("AoeToAir") else 0) / 1000,
            "cost": int(card["ManaCost"] or 0),
            "kamikaze": bool(unit.get("Kamikaze")),
        }
        e = dict(cre.get_card_stats(eid))
        eng = {
            "hp": e.get("hp"), "damage": e.get("damage"), "hit_speed_s": e.get("attack_cooldown", 0) / 10,
            "range": round(e.get("attack_range", 0), 2), "sight": round(e.get("sight_range", 0), 2),
            # engine tiles a tick -> the game's speed units (60 = Medium = 1.325 tiles a second, the engine's tier)
            "speed": round(e.get("speed", 0) * 10 / (1.325 / 60), 0),
            # a deploy delay (X-Bow, Mortar) runs alongside the deploy: the unit acts at the later of the two
            "deploy_s": max(e.get("deploy_ticks", 10), e.get("deploy_delay_ticks", 0)) / 10, "splash": round(e.get("splash_radius", 0), 2),
            "cost": int(e.get("cost", 0)),
        }
        diffs = []
        for k in ("hp", "damage", "hit_speed_s", "range", "sight", "speed", "deploy_s", "splash", "cost"):
            g, v = game[k], eng[k]
            if g == 0 and v == 0:
                continue
            if abs(g - v) > max(a.tol * max(abs(g), abs(v)), 0.05 if k not in ("hp", "damage", "speed") else 0):
                diffs.append(f"{k} game {g} engine {v}")
        if game["kamikaze"] != bool(e.get("die_after_first_hit")):
            diffs.append(f"kamikaze game {game['kamikaze']} engine {bool(e.get('die_after_first_hit'))}")
        report.append((name, card["Rarity"], m, diffs))
    bad = [r for r in report if r[3]]
    print(f"{len(report)} engine cards matched to the game files; {len(bad)} with differences "
          f"(> {a.tol:.0%} or noted); not matched: {missing}")
    for name, rar, m, diffs in bad:
        print(f"- {name} ({rar}, x{m:.2f}): " + "; ".join(diffs))


if __name__ == "__main__":
    main()
