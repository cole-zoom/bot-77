"""Prepare KataCR's generator [Wu et al. 2025, §3.1, Alg. 1] (MIT, third_party/katacr) for synth_v2 (decisions.md,
2026-10-10): every class of config/classes_v2.json, and a sprite folder with KataCR's segments plus our sprites.

  python3 scripts/katacr/setup_synth_v2.py <KataCR repo> <dataset dir> [sprite_review_v2.json]

1. label_list.py: appends classes_v2[155:] to unit_list (so generator ids == classes_v2 ids) and puts each new class
   in the generator's ground / flying / spell lists: an evo or hero form copies its base card's lists (KataCR's own
   choices, e.g. Balloon is "ground"), new cards are set below.
2. constant.py: path_dataset -> <dataset dir>.
3. <dataset dir>/images/segment: KataCR's segments (copied) + approved sprites (data/slices/approved) + auto-cut
   sprites (data/slices/auto_v2) minus the ones Cole rejected. File names follow the generator's
   `<class>_<team>_<n>.png` (team is irrelevant for us: the detector has no faction output).
Idempotent: re-running re-copies the sprites and skips patches already applied.
"""
import json, re, shutil, sys
from pathlib import Path

repo, ds = Path(sys.argv[1]), Path(sys.argv[2])
review = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None
BOT = Path(__file__).resolve().parents[2]
classes = json.loads((BOT / "config/classes_v2.json").read_text())

NEW = {  # cards with no KataCR base (lists from the card: flying troop / ground troop / spell)
    "minion-giant": ["ground"], "goblinstein-doctor": ["ground"], "goblinstein-monster": ["ground"],
    "berserker": ["ground"], "boss-bandit": ["ground"], "goblin-curse": ["ground", "spell"],
    "goblin-demolisher": ["ground"], "goblin-machine": ["ground"], "ronin": ["ground"], "rune-giant": ["ground"],
    "spirit-empress-air": ["flying"], "spirit-empress-ground": ["ground"], "suspicious-bush": ["ground"],
    "tomb-queen": ["ground"], "vines": ["ground", "spell"], "void": ["ground", "spell"],
    "royal-hogs": ["ground"], "dark-prince-hero-dismounted": ["ground"],
}
BASE_ALIAS = {"royal-hogs": "hog", "skeletons": "skeleton", "bats": "bat"}

# 1. label_list.py
p = repo / "katacr/constants/label_list.py"; s = p.read_text()
ns = {}
exec(compile(s.split("if __name__")[0], str(p), "exec"), ns)
unit_list = ns["unit_list"]
assert classes[:len(unit_list)] == unit_list[:len(classes)], "KataCR unit_list must be a prefix of classes_v2"
add = classes[len(unit_list):]
lists = {"ground": "ground_unit_list", "flying": "flying_unit_list", "spell": "spell_unit_list"}
member = {k: set(ns[v]) for k, v in lists.items()}
cats = {}
for c in add:
    base = re.sub(r"-(evolution|hero)(-dismounted)?$", "", c)
    base = BASE_ALIAS.get(base, base)
    if c in NEW:
        cats[c] = NEW[c]
    elif base in NEW:
        cats[c] = NEW[base]
    else:
        cats[c] = [k for k in lists if base in member[k]] or ["ground"]
if add:
    i = s.index("]\n", s.index("unit_list = ["))
    s = s[:i] + "".join(f"  '{c}',  # bot77 (classes_v2)\n" for c in add) + s[i:]
    for k, var in lists.items():
        names = [c for c in add if k in cats[c]]
        if names:
            j = s.index("]\n", s.index(f"\n{var} = [") + 1)
            s = s[:j] + "".join(f"  '{c}',  # bot77\n" for c in names) + s[j:]
    p.write_text(s)
print(f"label_list: +{len(add)} classes", {k: sum(k in v for v in cats.values()) for k in lists})

# 2. constant.py
p = repo / "katacr/build_dataset/constant.py"
p.write_text(re.sub(r'^path_dataset = Path\(".*"\)$', f'path_dataset = Path("{ds.resolve()}")', p.read_text(), flags=re.M))

# 3. segment folder
seg = ds / "images/segment"
if seg.exists():
    shutil.rmtree(seg)
shutil.copytree(BOT / "data/external/katacr_dataset/images/segment", seg, symlinks=False)
rejected = set()
if review:
    for r in review["rows"]:
        rejected |= set(r["bad"])
n = {}
for name, man in (("approved", "data/slices/approved/manifest.json"), ("auto", "data/slices/auto_v2/manifest.json")):
    for m in json.loads((BOT / man).read_text()):
        if Path(m["file"]).name in rejected:
            continue
        c = m["class"]
        d = seg / c; d.mkdir(exist_ok=True)
        n[c] = n.get(c, 0) + 1
        shutil.copy(BOT / m["file"], d / f"{c}_1_bot77{name}_{n[c]:04d}.png")
print(f"sprites added: {sum(n.values())} in {len(n)} classes; rejected by Cole: {len(rejected)}")
print("classes in classes_v2 with no sprites at all:",
      [c for c in classes if not (seg / c).exists() and c not in ("padding_0", "selected", "text")][:80])
