"""Data inventory (training-architecture phase 0, 2026-10-22): every LanceDB table, data folder, model and ground-truth
group with its size, where it lives (laptop / LaCie), what it is, and which code writes / reads it.
-> docs/data-inventory.md (regenerate any time; descriptions live in DESCRIBE below).

  uv run python scripts/inventory.py
"""
import datetime, os, re, shutil, subprocess
from collections import Counter
from pathlib import Path

import lancedb

ROOT = Path(".")
LACIE = Path("/Volumes/LaCie")
CODE = [p for d in ("src", "scripts") for p in (ROOT / d).rglob("*") if p.suffix in (".py", ".sh") and p.name != "inventory.py"]
CODE_TEXT = {p: p.read_text(errors="ignore") for p in CODE}

# layer: source / observation / ground truth / training data / bot / analysis / model input
DESCRIBE = {
    "detections": ("observation", "every detector run in one table (run_id): boxes, class, card, team, arena tiles"),
    "runs": ("observation", "what each detector run is and what it's used for"),
    "plays": ("observation / ground truth", "every play: from the hand (exact), detected by C, or made by the bot"),
    "moments": ("training data", "one player at one point in time with everything they saw: plays + could-have-played waits + bot plays"),
    "labels": ("training data", "what was / should have been done at a moment: replay, bot, review_cole, review_claude"),
    "reviews": ("ground truth", "each exported bot-play review (who, when, approved by, raw export)"),
    "move_samples_m": ("training data (A)", "A's rows rebuilt from moments + plays (identical to move_samples; phase 2 check)"),
    "media": ("source", "every video / recording file: where it really lives (laptop / LaCie), size, available now"),
    "videos": ("source", "one row per video file: creator, layout, mode, path (laptop or LaCie symlink), size"),
    "matches": ("source", "one row per match in a video: start / end, opponent, deck. Split / source / league not here yet"),
    "cards": ("reference", "the card catalogue: names, elixir, type, rarity, art (from the Clash API)"),
    "hud_states": ("observation", "what the HUD showed, every frame read: hand, next, elixir, clock, multiplier"),
    "events": ("observation / ground truth", "plays and abilities read from the hand (exact for the player whose hand is shown; replay answer keys)"),
    "unit_detections": ("observation", "detector output (older default run) used by mining, slices, autolabel"),
    "unit_detections_v5l_e15": ("observation", "detector V5 on all 524 keyed replays, every 0.25 s: the board for A / placement"),
    "unit_detections_v6l_best": ("observation", "detector V6 (live bot's detector) on the bot's replays"),
    "unit_detections_v7l_best": ("observation", "detector V7 on a check set (30-ish games)"),
    "unit_detections_v8l_best": ("observation", "detector V8 on a check set"),
    "unit_detections_mine4": ("observation", "detections from mining run 4 (RunPod), source of autolabels"),
    "opponent_events": ("observation", "C's detected opponent plays (default output table)"),
    "unit_crops": ("training data (C)", "unit image crops + embeddings: train the card classifier, feed mining"),
    "real_crop_index": ("training data (C)", "index of real crops used by the crop classifier"),
    "slice_index": ("training data (C)", "index of video slices for labelling"),
    "mined_candidates": ("training data (C)", "mined failure candidates for review / labelling"),
    "play_outcomes": ("analysis (C)", "per detected play: real / fake / missed vs the key; trains the fake filter"),
    "fakes_now": ("analysis (C)", "current fake plays snapshot (fake-filter work)"),
    "pov_fakes": ("analysis (C)", "fake plays in Cole's POV games"),
    "live_outcomes": ("analysis (C)", "play outcomes on Cole's live (POV) games"),
    "move_samples": ("training data (A)", "A's samples: one row per (decision, hand card), both players of every keyed replay"),
    "board_states": ("training data (A)", "the board at each of A's decisions (JSON strings for now)"),
    "placements": ("training data (placement)", "every play with its spot and the board before it"),
    "interactions": ("training data (playbook)", "play -> the other player's answer within 8 s, with positions"),
    "playbook": ("model input", "interactions aggregated per opponent card: common answers and spots"),
    "bot_actions": ("bot", "every bot play: card, spot, reason, hand, elixir, threats, full board, model spot"),
    "bot_games": ("bot", "every bot game: start / end, plays, result, deck"),
}
FOLDERS = {
    "data/raw/replays": "replay recordings from the Battle Log (bot + Cole's account)",
    "data/raw/tv_royale": "TV Royale replay recordings (top ladder)",
    "data/raw/live_games": "the bot's games: screen recording + a frame at / after every play",
    "data/raw/replay_links": "symlinks to replay videos by match id",
    "data/raw/live_ui": "screens the bot didn't recognise between games",
    "data/raw/live_calib": "BlueStacks calibration frames",
    "data/raw/youtube": "YouTube download metadata",
    "data/raw": "Cole's POV recordings + YouTube videos (top level)",
    "data/external": "datasets for the detector: KataCR, synthetic and real crop sets",
    "data/upload": "bundles made for RunPod uploads (tarballs and unpacked copies)",
    "data/reviews": "generated review pages (HTML)",
    "data/ground_truth": "answer keys and Cole's reviews (JSON)",
    "data/models": "model weights",
    "data/calibration": "layout calibration images",
    "data/cards": "card art + overrides",
    "data/slices": "video slices for labelling",
    "data/logs": "run logs",
    "data/opponent": "opponent-pipeline debug images",
    "data/live": "the bot's runtime files (STOP flag, game results)",
}


def du(p: Path) -> int:
    out = subprocess.run(["du", "-sk", str(p)], capture_output=True, text=True).stdout.split()
    return int(out[0]) * 1024 if out else 0


def human(n: float) -> str:
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{u}" if u in ("B", "K") else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}P"


def refs(name: str, how: str = "any") -> list[str]:
    pat = re.compile(rf"""["'/]{re.escape(name)}["'/ .]""")
    hit = [p.name for p, t in CODE_TEXT.items() if pat.search(t)]
    if how == "write":
        w = re.compile(rf"""(create_table\(\s*["']{re.escape(name)}["']|_save\(\s*["']{re.escape(name)}["']|out_table\s*=\s*["']{re.escape(name)}["'])""")
        hit = [p.name for p, t in CODE_TEXT.items() if w.search(t)]
    return sorted(set(hit))


def links_to_lacie(p: Path) -> tuple[int, int]:
    n = on = 0
    for f in p.rglob("*"):
        if f.is_symlink():
            n += 1
            on += str(os.readlink(f)).startswith(str(LACIE))
    return n, on


lines = [f"# Data inventory", "", f"Generated {datetime.date.today()} by `scripts/inventory.py` (phase 0 of "
         "[training-architecture.md](training-architecture.md)). Regenerate after any data change.", ""]

lap = shutil.disk_usage("/")
lines += ["## Storage", "", "| Tier | Used | Free |", "|---|---|---|",
          f"| Laptop SSD | {human(lap.used)} | {human(lap.free)} |"]
if LACIE.exists():
    lc = shutil.disk_usage(LACIE)
    lines.append(f"| LaCie HDD (`/Volumes/LaCie/bot-77-data`: {human(du(LACIE / 'bot-77-data'))}) | {human(lc.used)} | {human(lc.free)} |")
else:
    lines.append("| LaCie HDD | not mounted | |")
lines.append("")

db = lancedb.connect("data/lancedb")
names = db.table_names(limit=10_000)
lines += ["## LanceDB tables (`data/lancedb`, laptop)", "",
          "| Table | Layer | What it is | Rows | Size | Versions | Written by | Read by |", "|---|---|---|---|---|---|---|---|"]
total = 0
for n in sorted(names, key=lambda n: (DESCRIBE.get(n, ("zz",))[0], n)):
    t = db.open_table(n)
    p = Path("data/lancedb") / f"{n}.lance"
    size = du(p); total += size
    vers = len(list((p / "_versions").glob("*.manifest"))) if (p / "_versions").exists() else 0
    layer, what = DESCRIBE.get(n, ("?", "**undocumented**"))
    w = ", ".join(refs(n, "write")) or "—"
    r = ", ".join(x for x in refs(n) if x not in refs(n, "write"))[:120] or "—"
    lines.append(f"| `{n}` | {layer} | {what} | {t.count_rows():,} | {human(size)} | {vers} | {w} | {r} |")
lines += ["", f"Total: {len(names)} tables, {human(total)}.", ""]

lines += ["## Data folders", "", "| Folder | What it is | On laptop | Symlinks (→ LaCie) |", "|---|---|---|---|"]
for d, what in FOLDERS.items():
    p = Path(d)
    if not p.exists():
        continue
    if d == "data/raw":
        size = sum(du(f) for f in p.iterdir() if f.is_file() and not f.is_symlink())
        n = sum(1 for f in p.iterdir() if f.is_symlink()); on = sum(1 for f in p.iterdir() if f.is_symlink() and str(os.readlink(f)).startswith(str(LACIE)))
        lines.append(f"| `data/raw/*` (files at the top) | {what} | {human(size)} | {n} ({on}) |")
        continue
    n, on = links_to_lacie(p)
    lines.append(f"| `{d}` | {what} | {human(du(p))} | {n} ({on}) |")
lines.append("")

lines += ["## Models (`data/models`)", "", "| Model | Size | Referenced in code |", "|---|---|---|"]
for m in sorted(Path("data/models").iterdir()):
    r = refs(m.name)
    lines.append(f"| `{m.name}` | {human(du(m))} | {', '.join(r[:5]) or '— (not referenced: old version?)'} |")
lines.append("")

gt = Counter(re.sub(r"_\d{4}-\d\d-\d\d.*|_[A-Za-z0-9_-]{11}_m.*", "", f.name) for f in Path("data/ground_truth").glob("*.json"))
lines += ["## Ground truth (`data/ground_truth`)", "", "| Group | Files |", "|---|---|"]
lines += [f"| `{k}` | {v} |" for k, v in gt.most_common()]
lines += [f"| `reviews/` | {len(list(Path('data/ground_truth/reviews').glob('*.json')))} |", ""]

lines += ["## Config (`config/`)", "", "| File | Referenced in code |", "|---|---|"]
for c in sorted(Path("config").iterdir()):
    lines.append(f"| `{c.name}` | {', '.join(refs(c.name)[:4]) or '—'} |")
lines.append("")
Path("docs/data-inventory.md").write_text("\n".join(lines) + "\n")
print("docs/data-inventory.md:", len(names), "tables")
