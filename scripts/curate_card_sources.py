"""Parse the pasted evo / hero / champion tables in data/cards/sources into data/cards/curated.yaml.

Re-run after editing a source file: `uv run python scripts/curate_card_sources.py`, then `uv run bot77 cards ingest`."""
import json, re
from pathlib import Path
import yaml

SRC = Path("data/cards/sources")
api = {c["name"]: c for c in json.loads(max(Path("data/cards/api_snapshots").glob("*.json")).read_text())["items"]}
def api_name(n):
    n = n.strip()
    return n if n in api else n.rstrip(".") if n.rstrip(".") in api else None

cur, problems = {}, []

# --- evolutions ---
lines = SRC.joinpath("evolutions.txt").read_text().splitlines()
i = 0
while i < len(lines):
    m = re.match(r"^\S+EvoShard\s+(\d+)\s+(\d+)\s+(\d+)\s+(.*)$", lines[i])
    if not m:
        i += 1; continue
    name = api_name(lines[i-1]); cost, cycles, total, rest = int(m[1]), int(m[2]), int(m[3]), m[4]
    parts = re.split(r"\s{2,}", rest, maxsplit=1)
    if len(parts) == 1:  # stat boost wraps onto the next line
        i += 1; more = re.split(r"\s{2,}", lines[i], maxsplit=1)
        parts = [parts[0] + "; " + more[0], more[1]]
    stat, ability = parts
    aname, adesc = ability.split(" - ", 1)
    if name is None: problems.append(f"evo: unknown card {lines[i-1]!r}")
    else:
        if api[name].get("elixirCost") != cost: problems.append(f"evo {name}: cost {cost} != API {api[name].get('elixirCost')}")
        if cost * (cycles + 1) != total: problems.append(f"evo {name}: total {total} != cost*(cycles+1)")
        cur.setdefault(name, {})["evo"] = {"cycles": cycles, "stat_boost": None if stat == "Identical Stats" else stat,
                                            "ability_name": aname.strip(), "ability_description": adesc.strip()}
    i += 1

# --- heroes ---
blocks = re.findall(r"^(.+)\n\S+HeroShard\n(\d+)\s+(\d+)\s+(\d+)\s+(.+?) -\n(.+)$", SRC.joinpath("heroes.txt").read_text(), re.M)
for raw, cost, acost, total, aname, desc in blocks:
    name = api_name(raw)
    if name is None: problems.append(f"hero: unknown card {raw!r}"); continue
    if api[name].get("elixirCost") != int(cost): problems.append(f"hero {name}: cost {cost} != API")
    if int(cost) + int(acost) != int(total): problems.append(f"hero {name}: total mismatch")
    cur.setdefault(name, {})["hero_ability"] = {"name": aname.strip(), "cost": int(acost), "description": desc.strip()}

# --- champions ---
CHAMP_ABILITY = {"Golden Knight": "Dashing Dash", "Skeleton King": "Soul Summoning", "Boss Bandit": "Getaway Grenade",
                 "Archer Queen": "Cloaking Cape", "Mighty Miner": "Explosive Escape", "Goblinstein": "Lightning Link",
                 "Little Prince": "Royal Rescue", "Monk": "Pensive Protection"}
text = SRC.joinpath("champions.txt").read_text()
for name, aname in CHAMP_ABILITY.items():
    para = re.search(rf"^{name}\n\S+Card\n.*?^(For (\d+) Elixir, you can use .+)$", text, re.M | re.S)
    if not para or aname not in para[1]: problems.append(f"champion {name}: ability paragraph not found"); continue
    entry = {"name": aname, "cost": int(para[2]), "description": para[1].strip()}
    if name == "Boss Bandit": entry |= {"uses_per_deploy": 2, "cooldown_s": 3}
    cur.setdefault(name, {})["champion_ability"] = entry

cur.setdefault("Heal Spirit", {})["type"] = "troop"
cur.setdefault("Mirror", {})["elixir"] = "variable"

# coverage vs API
for n, c in api.items():
    ic = c["iconUrls"]
    if "evolutionMedium" in ic and "evo" not in cur.get(n, {}): problems.append(f"MISSING evo data: {n}")
    if "heroMedium" in ic and "hero_ability" not in cur.get(n, {}): problems.append(f"MISSING hero data: {n}")
for n, e in cur.items():
    if "evo" in e and "evolutionMedium" not in api[n]["iconUrls"]: problems.append(f"evo data but API has no evo art: {n}")
    if "hero_ability" in e and "heroMedium" not in api[n]["iconUrls"]: problems.append(f"hero data but API has no hero art: {n}")

header = """# Hand-maintained card facts that the Clash Royale API doesn't provide.
# Merged onto the API snapshot by `bot77 cards ingest`. Keys are exact API card names.
# Sources (Cole, 2026-09-24): data/cards/sources/{evolutions,heroes,champions}.txt
#
# evo:               cycles to charge, stat boost vs base card, special ability
# hero_ability /
# champion_ability:  name, elixir cost, description; uses_per_deploy (default 1), cooldown_s
# type:              override the troop/building/spell type derived from the card id
#                    (Heal Spirit: was the Heal spell, id never updated)
# elixir:            "variable" = Mirror (mirrored card's cost + 1)
"""
body = yaml.safe_dump(dict(sorted(cur.items())), sort_keys=False, allow_unicode=True, width=10_000)
Path("data/cards/curated.yaml").write_text(header + "\n" + body)
print(f"evos {sum('evo' in e for e in cur.values())}, heroes {sum('hero_ability' in e for e in cur.values())}, champions {sum('champion_ability' in e for e in cur.values())}")
print("\n".join(problems) or "no problems")
