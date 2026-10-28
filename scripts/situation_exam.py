"""The situation exam (2026-10-27, with Cole): every defence template on fixed held-out seeds, scored the way training
scores it (bot77.rl.situations), for a checkpoint and two reference players.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/situation_exam.py [--ckpt PATH ...] [--n 30]

Players: "nothing" (never plays), "rush" (plays its most expensive affordable card at the bridge of the other lane: the
exploit the scoring must not reward), "defend" (a plain scripted defence: the yardstick), and each --ckpt (sampled
like training). A template is fit to train on only if
a sensible defence beats both references (Cole's check); the trend of a checkpoint's scores across rounds is the exam.
Seeds 10_000+ (training draws from the run's own seeds, so these states are held out by construction).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"
DECK = "2,81,5,43,9,95,7,3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    os.environ.update({"CLASH_DECK": os.environ.get("CLASH_DECK", DECK),
                       "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    ckpts = [str((ROOT / c).resolve()) for c in a.ckpt]
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import python_ai  # noqa: F401
    import torch
    from torch.distributions import Categorical
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.models.policy_io import LSTM_HIDDEN

    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    torch.set_num_threads(1)

    def net_player(path):
        net = MicroRoyaleNet()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=False)
        net.eval()
        state = {}

        def act(env, obs, first):
            if first:
                state["h"] = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
            o = torch.tensor(obs).unsqueeze(0)
            with torch.no_grad():
                m = net.affordability_mask(o)
                f, e, sp, hi = net.extract_features_hires(o)
                lg, _, _, _, state["h"] = net.step_lstm_and_card(f, state["h"], m)
                k = Categorical(logits=lg).sample()
                cell = Categorical(logits=net.placement_given_card(state["h"][0], e, k, o, sp, hires_map=hi)).sample()
                x, y = net.cell_to_xy(cell)
            return int(k.item()), float(x.item()), float(y.item())
        return act

    def nothing(env, obs, first):
        return 4, 0.0, 0.0

    def rush(env, obs, first):
        hand, el = list(env.game.get_hand()), env.game.get_elixir()
        ok = [(sit.COST.get(c, 9), i) for i, c in enumerate(hand) if sit.COST.get(c, 9) <= el and c not in (sit.FIREBALL, sit.ARROWS)]
        if not ok:
            return 4, 0.0, 0.0
        threat_x = env.rush_lane
        x = sit.LANE_X[1] if threat_x < 8.5 else sit.LANE_X[0]
        return max(ok)[1], x, float(env.MAX_Y)  # the bridge end of our half, in the other lane

    def defend(env, obs, first):
        """A plain sensible defence (the yardstick, not a tuned one): the dealt answer once, in front of the threatened
        tower (Goblin Hut in the middle, to pull); on a supported push, then the support answer on the support (Fireball
        where it stands, a troop behind the tower), each card only when affordable and only once."""
        if first:
            env.defend_played = set()
        spec, hand, el = env.situation.spec if env.situation else None, list(env.game.get_hand()), env.game.get_elixir()
        if spec is None:
            return 4, 0.0, 0.0
        lane_x = spec["units"][0][1]
        for group in ("answers", "answers2"):
            if group in env.defend_played or not spec.get(group):
                continue
            for i, c in enumerate(hand):
                if c in spec[group] and sit.COST[c] <= el:
                    env.defend_played.add(group)
                    if c == sit.GOBLIN_HUT:
                        return i, 8.5, 9.0
                    if c in (sit.FIREBALL, sit.ARROWS):
                        sup = [u for u in env.game.get_units() if u.team == 1 and u.card_id == spec["units"][-1][0]]
                        if not sup:
                            continue
                        return i, float(sup[0].x), float(sup[0].y)
                    return i, lane_x, 9.0
        return 4, 0.0, 0.0

    players = [("nothing", nothing), ("rush", rush), ("defend", defend)] + [(Path(c).parent.name + "/" + Path(c).stem, net_player(c)) for c in ckpts]
    tpls = [sit.hog_rider, sit.battle_ram, sit.melee, sit.supported_push]
    print(f"{'template':16s} " + " ".join(f"{p[0][:24]:>34s}" for p in players))
    for tpl in tpls:
        row = {}
        for pname, act in players:
            env = CloneEnv({"scenario_seed": 0})
            env.forced_template = tpl
            res = []
            for i in range(a.n):
                env.rng = np.random.default_rng(10_000 + i)  # the same states for every player
                env.scenario_rng = env.rng
                obs, _ = env.reset()
                env.rush_lane = env.situation.spec["units"][0][1]
                first = True
                while True:
                    k, x, y = act(env, obs, first)
                    first = False
                    obs, r, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                          "target_y": np.array([y], np.float32),
                                                          "activate_ability_slot1": 0, "activate_ability_slot2": 0})
                    if info.get("rl_situation_end") or term or trunc:
                        res.append(env.last_situation)
                        break
            sc = np.array([r["score"] for r in res])
            row[pname] = (sc.mean(), np.median([r["tower_hp_lost"] for r in res]), np.mean([r["elixir_spent"] for r in res]),
                          np.mean([r["unresolved"] for r in res]))
        print(f"{tpl.__name__:16s} " + " ".join(f"{v[0]:+.2f} (hp {v[1]:5.0f}, el {v[2]:3.1f}, ur {v[3]:.0%})" for v in row.values()))
    print("score = -(tower HP lost / princess max) - 0.04 * elixir spent [+ survivors] - 0.3 if unresolved; hp = median tower HP"
          " lost, el = mean elixir spent, ur = share unresolved at the timeout")


if __name__ == "__main__":
    main()
