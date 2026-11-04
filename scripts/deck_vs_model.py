"""A checkpoint against the scripted deck bots (bot77.rl.decks), whole games in the sim (2026-11-04, Cole: "do they all
beat the current model rn?").

  uv run --no-sync --with gymnasium==1.3.0 python scripts/deck_vs_model.py --ckpt data/models/bc_school_v5/epoch03.pth [--n 20]

Our side is the checkpoint on our deck, deciding every 0.1 s and sampling as live (match_eval.py's loop); team 1 is the
deck bot, ticked every 0.1 s as on the play page and told each of our plays that lands. Seeds 30_000+.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_eval import DECK, FORK  # noqa: E402

_S: dict = {}


def setup(ckpt):
    os.environ.update({"CLASH_DECK": DECK, "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "0",
                       "RL_CLIENT_COLLISION": "1"})
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import python_ai  # noqa: F401
    import torch
    torch.set_num_threads(1)
    import clash_royale_env as cre
    import bot77.rl.decks  # noqa: F401
    from bot77.rl.deck_bots import DECKS
    from bot77.rl.clone_env import CloneEnv
    from python_ai.models.net import MicroRoyaleNet
    net = MicroRoyaleNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    net.eval()

    class DeckEnv(CloneEnv):
        """CloneEnv with a deck bot as team 1."""
        bot = None

        def _sample_pfsp_opponent(self):
            self.opponent_kind = "neural"
            self.opp_deck = list(self.bot.deck_ids)
            none = cre.TowerTroopType.NONE
            self.game = cre.ClashRoyaleEnv(self.deck, self.opp_deck, self.max_ticks, none, none)
            self.game.seed(int(self.bot_seed))
            self.bot.bind(self.game)
            self.opponent_checkpoint_path = None

        def _opponent_action(self):
            mv = self.bot.tick()
            return (*(mv or (4, 0.0, 0.0)), False, False)

        def step(self, action, skip_frames=None):
            k = int(action["card_index"])
            hand = list(self.game.get_hand())
            spent = float(self.game.get_elixir_spent(0))
            out = super().step(action, skip_frames)
            if k < 4 and k < len(hand) and float(self.game.get_elixir_spent(0)) > spent + 0.5:
                self.bot.note_play(hand[k], float(action["target_x"][0]), float(action["target_y"][0]))
            return out

    _S.update(torch=torch, net=net, DeckEnv=DeckEnv, DECKS=DECKS)


def game(args):
    ckpt, key, seed = args
    if not _S:
        setup(ckpt)
    torch, net = _S["torch"], _S["net"]
    from torch.distributions import Categorical
    from python_ai.models.policy_io import LSTM_HIDDEN
    env = _S["DeckEnv"]({"scenario_seed": 0, "decision_ticks": 1, "scenarios_enabled": False})
    env.bot = _S["DECKS"][key](seed)
    env.bot_seed = seed
    env.rng = np.random.default_rng(seed)
    env.scenario_rng = env.rng
    torch.manual_seed(seed)
    obs, _ = env.reset()
    h = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
    while True:
        o = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            m = net.affordability_mask(o)
            f, e, sp, hi = net.extract_features_hires(o)
            lg, _, _, _, h = net.step_lstm_and_card(f, h, m)
            k = Categorical(logits=lg).sample()
            x, y = net.cell_to_xy(Categorical(logits=net.placement_given_card(h[0], e, k, o, sp, hires_map=hi)).sample())
        obs, r, term, trunc, info = env.step({"card_index": int(k.item()), "target_x": np.array([float(x.item())], np.float32),
                                              "target_y": np.array([float(y.item())], np.float32),
                                              "activate_ability_slot1": 0, "activate_ability_slot2": 0})
        if term or trunc:
            g = env.game
            alive = [[g.get_tower_hp(t, s) > 0 for s in range(3)] for t in (0, 1)]
            return {"deck": key, "seed": seed, "result": float(r), "clock": round(g.get_current_tick() * 0.1, 1),
                    "crowns_model": 3 - sum(alive[1]), "crowns_bot": 3 - sum(alive[0]),
                    "bot_plays": len(env.bot.mine), "fired": dict(env.bot.fired)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/bc_school_v5/epoch03.pth")
    ap.add_argument("--decks", default="", help="comma-separated keys (default: all)")
    ap.add_argument("--n", type=int, default=20, help="games per deck")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ckpt = str((ROOT / a.ckpt).resolve())
    sys.path.insert(0, str(ROOT / "src"))
    import bot77.rl.decks  # noqa: F401
    from bot77.rl.deck_bots import DECKS
    keys = [k for k in a.decks.split(",") if k] or sorted(DECKS)
    with Pool(a.workers) as pool:
        rows = pool.map(game, [(ckpt, k, 30_000 + i) for k in keys for i in range(a.n)])
    for k in keys:
        rs = [r for r in rows if r["deck"] == k]
        res = np.array([r["result"] for r in rs])
        print(f"{DECKS[k].NAME:24s} bot {int((res < -0.5).sum()):2d} - model {int((res > 0.5).sum()):2d}"
              f" (draws {int((abs(res) <= 0.5).sum())}); crowns bot {np.mean([r['crowns_bot'] for r in rs]):.2f},"
              f" model {np.mean([r['crowns_model'] for r in rs]):.2f}; {np.mean([r['clock'] for r in rs]):.0f} s")
    if a.json:
        out = ROOT / a.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
