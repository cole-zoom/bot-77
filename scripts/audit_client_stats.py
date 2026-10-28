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

Spells (SPELLS below): damage and Crown Tower damage per hit, radius, hit count, cost, from the spell's projectile,
area object (AEO) or damage-over-time buff.

Compared for units: hitpoints, damage (the unit's or its projectile's), hit speed, range, sight, speed, deploy time, load time,
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
# engine card -> the client unit that is the card on the board, where SummonCharacter names something else
UNIT_OVERRIDE = {"Goblin Drill": "GoblinDrill"}  # SummonCharacter is the burrowing GoblinDrillDig


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


def sections() -> dict[str, dict]:
    """{kind: {name: body}} for every section of every logic TOML (AEO, BUFF, PROJECTILE, DAMAGE_TYPE, ...)."""
    out: dict[str, dict] = {}
    for f in list((LOGIC / "characters").glob("*.toml")) + [LOGIC / "projectiles.toml"]:
        try:
            d = tomllib.loads(f.read_text(errors="replace"))
        except Exception:  # noqa: BLE001 -- the same unparseable few units() reports
            continue
        for kind, sec in d.items():
            if isinstance(sec, dict):
                for name, body in sec.items():
                    if isinstance(body, dict):
                        out.setdefault(kind, {})[name] = body
    return out


# engine spell -> where the client keeps it: a projectile (Damage, Radius, CrownTowerDamagePercent), an area
# object with its own Damage {BaseDamage, TowerDamage}, or an area object applying a damage-over-time buff
# (DamagePerSecond every HitFrequency for the area's LifeDuration). `card` is the spells_other.csv row (cast
# radius, waves). Rollers: the engine's spell_radius is not their corridor, so radius isn't compared.
SPELLS = {
    "Arrows": {"card": "Arrows", "proj": "ArrowsSpell"},
    "Fireball": {"card": "Fireball", "proj": "FireballSpell"},
    "Rocket": {"card": "Rocket", "proj": "RocketSpell"},
    "Giant Snowball": {"card": "Snowball", "proj": "SnowballSpell"},
    "The Log": {"card": "Log", "proj": "LogProjectileRolling", "roller": True},
    "Barbarian Barrel": {"card": "BarbLog", "proj": "BarbLogProjectileRolling", "roller": True},
    "Lightning": {"card": "Lightning", "proj": "LighningSpell", "aeo": "Lightning"},
    "Zap": {"card": "Zap", "aeo": "Zap"},
    "Freeze": {"card": "Freeze", "aeo": "Freeze"},
    "Rage": {"card": "Rage", "aeo": "RageDamage"},
    "Poison": {"card": "Poison", "aeo": "Poison", "buff": "Poison"},
    "Tornado": {"card": "Tornado", "aeo": "Tornado", "buff": "Tornado"},
    "Earthquake": {"card": "Earthquake", "aeo": "Earthquake", "buff": "Earthquake"},
    "Goblin Curse": {"card": "GoblinCurse", "aeo": "GoblinCurseBase", "buff": "GoblinCurseDamage"},
    "Royal Delivery": {"card": "RoyalDelivery", "proj": "RoyalDeliveryProjectile"},
}


def spell_game(spec, secs, tab, card) -> dict:
    """The client's level-11 figures for one spell: damage and Crown Tower damage per hit, radius, hits, cost."""
    def mult(sec):
        return level11(sec.get("Rarity", "Common"), tab)

    def tower(dmg, sec, m):
        if "CrownTowerDamagePerHit" in sec:
            return round(sec["CrownTowerDamagePerHit"] * m)
        return round(dmg * (1 + sec.get("CrownTowerDamagePercent", 0) / 100))

    proj = secs.get("PROJECTILE", {}).get(spec.get("proj", ""), {})
    aeo = secs.get("AEO", {}).get(spec.get("aeo", ""), {})
    buff = secs.get("BUFF", {}).get(spec.get("buff", ""), {})
    radius = int(card.get("Radius") or 0) / 1000
    if buff:
        m = mult(buff)
        freq = buff.get("HitFrequency", 1000)
        dmg = round(buff.get("DamagePerSecond", 0) * freq / 1000 * m)
        hits = max(1, round(aeo.get("LifeDuration", freq) / freq))
        ct = tower(dmg, buff, m)
        radius = radius or aeo.get("Radius", 0) / 1000
    elif proj:
        m = mult(proj)
        dmg = round(proj.get("Damage", 0) * m)
        ct = tower(dmg, proj, m)
        hits = int(card.get("ProjectileWaves") or 1)
        radius = radius or proj.get("Radius", 0) / 1000
    else:
        d = aeo.get("Damage")
        if isinstance(d, str):  # a named DAMAGE_TYPE (Rage)
            d = secs.get("DAMAGE_TYPE", {}).get(d, {})
        d = d or {}
        m = mult(aeo)
        dmg = round(d.get("BaseDamage", 0) * m)
        ct = round(d.get("TowerDamage", d.get("BaseDamage", 0)) * m)
        hits = 1
        radius = radius or aeo.get("Radius", 0) / 1000
    return {"damage": dmg, "tower": ct, "hits": hits, "radius": radius, "cost": int(card.get("ManaCost") or 0)}


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
        unit = chars.get(UNIT_OVERRIDE.get(name, card["SummonCharacter"]))
        if unit is None:
            missing.append(f"{name} (unit {card['SummonCharacter']})")
            continue
        # the unit's own Rarity field scales it: the client stores units at their level-1 value under the Common
        # table even for Rare / Epic cards (Mini P.E.K.K.A: 543 x 2.56 = 1390 at level 11)
        m = level11(unit.get("Rarity", card["Rarity"]), tab)
        proj = projs.get(unit.get("Projectile", ""), {})
        if not proj.get("Damage"):  # Princess: the damage is on her first-shot projectile, the rest decoration
            proj = projs.get(unit.get("CustomFirstProjectile", ""), proj)
        if not proj.get("Damage") and proj.get("SpawnProjectile"):  # Firecracker: the shell bursts into sparks
            proj = projs.get(proj["SpawnProjectile"], proj)
        seq = unit.get("AttackSequenceList")
        # the hit damage: a ramp's last stage (Inferno, Mighty Miner; engine `damage` is the ramped max), the
        # unit's own Damage, an attack sequence's first hit (Berserker), or its projectile's
        ramp = unit.get("VariableDamage3") if unit.get("VariableDamageTime1") else None  # Monk's are a combo, not a ramp
        hit = (ramp or unit.get("Damage")
               or (seq[0].get("Damage") if isinstance(seq, list) and seq and isinstance(seq[0], dict) else 0)
               or proj.get("Damage") or 0)
        game = {
            "hp": round(unit.get("Hitpoints", 0) * m),
            "damage": round(hit * m),
            "hit_speed_s": unit.get("HitSpeed", 0) / 1000,
            "range": unit.get("Range", 0) / 1000,
            "sight": unit.get("SightRange", 0) / 1000,
            "speed": unit.get("Speed", 0),
            "deploy_s": unit.get("DeployTime", 1000) / 1000,
            "splash": max(unit.get("AreaDamageRadius", 0), proj.get("Radius", 0) if proj.get("AoeToGround") or proj.get("AoeToAir") else 0) / 1000,
            "cost": int(card["ManaCost"] or 0),
            "kamikaze": bool(unit.get("Kamikaze")),
            "lifetime_s": unit.get("LifeTime", 0) / 1000,
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
            "lifetime_s": e.get("lifetime_ticks", 0) / 10 if game["lifetime_s"] else 0,
        }
        skip = set()
        if game["kamikaze"]:  # one hit and gone: the hit speed never comes round
            skip.add("hit_speed_s")
        if not game["damage"] and not eng["damage"]:  # a building that never attacks: sight / range / hit speed unused
            skip |= {"hit_speed_s", "range", "sight"}
        diffs = []
        for k in ("hp", "damage", "hit_speed_s", "range", "sight", "speed", "deploy_s", "splash", "cost", "lifetime_s"):
            if k in skip:
                continue
            g, v = game[k], eng[k]
            if g == 0 and v == 0:
                continue
            if abs(g - v) > max(a.tol * max(abs(g), abs(v)), 0.05 if k not in ("hp", "damage", "speed") else 0):
                diffs.append(f"{k} game {g} engine {v}")
        if game["kamikaze"] != bool(e.get("die_after_first_hit")):
            diffs.append(f"kamikaze game {game['kamikaze']} engine {bool(e.get('die_after_first_hit'))}")
        report.append((name, card["Rarity"], m, diffs))
    if not a.card or any(norm(a.card) == norm(k) for k in SPELLS):
        secs = sections()
        spell_rows = {r["Name"]: r for r in csv.DictReader(open(LOGIC / "spells_other.csv")) if r["Name"]}
        by_name = {info["name"]: eid for eid, info in sc.engine_cards().items()
                   if info["is_spell"] and not info["evolution"]}
        for name, spec in SPELLS.items():
            if (a.card and norm(a.card) != norm(name)) or name not in by_name:
                continue
            game = spell_game(spec, secs, tab, spell_rows[spec["card"]])
            e = dict(cre.get_card_stats(by_name[name]))
            ct = e.get("spell_crown_tower_damage", -1)
            eng = {"damage": e.get("damage"), "tower": e.get("damage") if ct < 0 else ct, "hits": e.get("spell_hits"),
                   "radius": round(e.get("spell_radius", 0), 2), "cost": int(e.get("cost", 0))}
            diffs = []
            for k in ("damage", "tower", "hits", "radius", "cost"):
                if k == "radius" and spec.get("roller"):
                    continue
                g, v = game[k], eng[k]
                if abs(g - v) > max(a.tol * max(abs(g), abs(v)), 0.05 if k == "radius" else 0):
                    diffs.append(f"{k} game {g} engine {v}")
            report.append((name, "spell", 1.0, diffs))
    bad = [r for r in report if r[3]]
    print(f"{len(report)} engine cards (troops, buildings, spells) matched to the game files; {len(bad)} with differences "
          f"(> {a.tol:.0%} or noted); not matched: {missing}")
    for name, rar, m, diffs in bad:
        print(f"- {name} ({rar}, x{m:.2f}): " + "; ".join(diffs))


if __name__ == "__main__":
    main()
