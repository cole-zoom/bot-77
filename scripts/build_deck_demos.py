"""Cole's games on our deck against the scripted deck bots (scripts/deck_play.py, "vs <deck> (you play the bot)") as
imitation data, in the imitation shards' format, so scripts/train_bc.py can mix them with the other demos (2026-11-06;
Cole: "this is also for the model to do immitation learning").

  uv run --no-sync python scripts/build_deck_demos.py [--since "2026-11-06 16:00"] [--weight 3.0]
  -> data/cache/bc/deck_t01/cole_<deck>_<seed>.npz   (gitignored)

Each game is replayed from its seed exactly as the page played it: the engine as deck_play.py builds it, the deck bot
ticked every 0.1 s and told each of Cole's plays that landed, his plays (and refusals) at their logged ticks. A game is
kept only if every one of his plays lands, or is refused, as the log says; one that doesn't replay (an older engine) is
reported and skipped. The deck bot is the one the page ran: its module as committed at the game's time (bot_class), since
its rules change after Cole's games. Per tick: the observation team 0 decides on (float16), the hand slot played (4 = wait) and its
board cell (the fork's binning, -1 for a wait), the fields of data/cache/bc/defence_t01.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_defence_demos import cell_from_xy  # noqa: E402

LOG = ROOT / "data/live/deck_play.jsonl"
OUT = ROOT / "data/cache/bc/deck_t01"


_CLASSES: dict = {}


def bot_class(deck_cls, when: str, logged_rev: str | None = None):
    """The deck bot's class as the page ran it: at the commit the game logged (`code.rev`, since 2026-11-06), else as
    committed at `when`; the current class when that is the same version."""
    path = Path(sys.modules[deck_cls.__module__].__file__).relative_to(ROOT)
    rev = subprocess.run(["git", "log", "-1", "--format=%H", *([logged_rev] if logged_rev else [f"--until={when}"]),
                          "--", str(path)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "log", "-1", "--format=%H", "--", str(path)], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    if not rev or rev == head:
        return deck_cls
    if (path, rev) not in _CLASSES:
        src = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        f = Path(tempfile.mkdtemp()) / path.name
        f.write_text(src.replace("@register\n", ""))  # an old copy must not replace the live deck in DECKS
        spec = importlib.util.spec_from_file_location(f"{deck_cls.__module__}_{rev[:7]}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CLASSES[(path, rev)] = getattr(mod, deck_cls.__name__)
    return _CLASSES[(path, rev)]


def replay(g, cre, DECKS, ours):
    """(obs, card, cell, mismatches) for one logged game."""
    bot = bot_class(DECKS[g["deck"]], g["when"], g.get("code", {}).get("rev"))(g["seed"])
    none = cre.TowerTroopType.NONE
    env = cre.ClashRoyaleEnv(ours, bot.deck_ids, 3600, none, none)
    env.seed(g["seed"])
    env.reset()
    bot.bind(env)
    by_n = {round(p["t"] / 0.1) - 1: p for p in g["plays"]}
    obs, card, cell, bad = [], [], [], 0
    for n in range(int(round(g["result"]["clock"] / 0.1))):
        p = by_n.get(n)
        hand = list(env.get_hand())
        k = hand.index(p["card_id"]) if p and p["card_id"] in hand else 4
        x, y = (float(p["x"]), float(p["y"])) if p else (0.0, 0.0)
        obs.append(np.asarray(env.get_observation_for_team(0), np.float16))
        spent = env.get_elixir_spent(0)
        mv = bot.tick()
        bot.events.clear()
        k1, x1, y1 = mv if mv else (4, 0.0, 0.0)
        env.step_self_play(k, x, y, k1, x1, y1, 1)
        landed = bool(p) and env.get_elixir_spent(0) > spent + 0.5
        if p:
            bad += landed != p["landed"]
        if landed:
            bot.note_play(hand[k], x, y)
        # a refused play is a wait in the demo (he couldn't make it there)
        card.append(k if landed else 4)
        cell.append(cell_from_xy(x, y) if landed else -1)
    return np.stack(obs), np.array(card, np.int64), np.array(cell, np.int64), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-06 16:00", help="games logged at or after this (today's engine)")
    ap.add_argument("--weight", type=float, default=3.0, help="per-row weight, as build_defence_demos' --cole-weight")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import clash_royale_env as cre
    import bot77.rl.decks  # noqa: F401
    from bot77.rl.deck_bots import DECKS, OUR_DECK, card_id
    cre.set_client_collision(True)  # as deck_play.py
    ours = [card_id(n) for n in OUR_DECK]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = rows = 0
    for line in open(LOG):
        g = json.loads(line)
        if g.get("mode") == "you_play_the_deck" or str(g.get("when", "")) < a.since or g.get("deck") not in DECKS:
            continue
        obs, card, cell, bad = replay(g, cre, DECKS, ours)
        res = g["result"]
        tag = f"{g['deck']} {g['seed']} ({g['when']}): {res['crowns'][0]}-{res['crowns'][1]}, {int((card < 4).sum())} plays"
        if bad:
            print(f"skip {tag}: {bad} plays don't replay as logged")
            continue
        np.savez_compressed(out / f"cole_{g['deck']}_{g['seed']}.npz", obs=obs, card=card, cell=cell,
                            t=np.arange(len(card), dtype=np.float32) / 10,
                            weight=np.full(len(card), a.weight, np.float32), split=np.array("train"),
                            side=np.array(f"cole_{g['deck']}"), result=np.array(f"{res['result']:.3f}"))
        kept += 1
        rows += len(card)
        print(f"kept {tag}, {len(card)} rows")
    print(f"done: {kept} games, {rows} rows -> {out}")


if __name__ == "__main__":
    main()
