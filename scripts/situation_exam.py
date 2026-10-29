"""The situation exam (2026-10-27, with Cole): every defence template on fixed held-out seeds, scored the way training
scores it (bot77.rl.situations), for a checkpoint and two reference players.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/situation_exam.py [--ckpt PATH ...] [--n 30]

Players: "nothing" (never plays), "rush" (plays its most expensive affordable card at the bridge of the other lane: the
exploit the scoring must not reward), "defend" (a plain scripted defence: the yardstick), and each --ckpt (sampled
like training). A template is fit to train on only if
a sensible defence beats both references (Cole's check); the trend of a checkpoint's scores across rounds is the exam.
Seeds 10_000+ (training draws from the run's own seeds, so these states are held out by construction). Torch is seeded
with the same seed before each start (2026-10-28): the networks sample (ours and the clone opponent), and unseeded the
same checkpoint's template means moved by up to 0.36 between runs at 20 seeds. --ticks (2026-10-28): before it the
exam ran every player once a second (CloneEnv's default), so v3 / v4, trained at 0.1 s, were examined off their pace.
"""
import argparse
import json
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
    ap.add_argument("--only", nargs="*", default=None, help="template names (default: all)")
    ap.add_argument("--ticks", type=int, default=1,
                    help="engine ticks between our decisions (1 = every 0.1 s, as v3 / v4 train; 10 = once a second)")
    ap.add_argument("--greedy", action="store_true",
                    help="the networks take their most likely card (or wait) and cell instead of sampling; on an "
                         "uncalibrated imitation checkpoint this plays when its calibrated play chance passes 1/(1+play weight)")
    ap.add_argument("--reference", action="store_true",
                    help="also score every player on Cole's reference starts (data/live/cole_reference.jsonl)")
    ap.add_argument("--no-spell-guard", action="store_true",
                    help="turn CloneEnv's spell guard off (bot77.rl.guards; on by default from 2026-10-29): the guard-off record")
    ap.add_argument("--reference-only", action="store_true", help="skip the templates (with --reference: only Cole's starts)")
    ap.add_argument("--json", default=None, help="also write every episode's result here (jsonl)")
    a = ap.parse_args()
    os.environ.update({"CLASH_DECK": os.environ.get("CLASH_DECK", DECK),
                       "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1",
                       "RL_SPELL_GUARD": "0" if a.no_spell_guard else "1"})
    jout = open(a.json, "w") if a.json else None
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
                k = lg.argmax(-1) if a.greedy else Categorical(logits=lg).sample()
                pl = net.placement_given_card(state["h"][0], e, k, o, sp, hires_map=hi)
                cell = pl.argmax(-1) if a.greedy else Categorical(logits=pl).sample()
                x, y = net.cell_to_xy(cell)
            return int(k.item()), float(x.item()), float(y.item())
        return act

    def lane_of(spec):  # the threat's lane (1 / 9 start empty: their planned lane)
        return spec["units"][0][1] if spec["units"] else spec.get("lane", sit.LANE_X[0])

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
        return sit.scripted_defence(env)

    players = [("nothing", nothing), ("rush", rush), ("defend", defend)] + [(Path(c).parent.name + "/" + Path(c).stem, net_player(c)) for c in ckpts]
    tpls = [] if a.reference_only else [t for t in sit.TEMPLATES if not a.only or t.__name__ in a.only]
    print(f"{'template':16s} " + " ".join(f"{p[0][:24]:>34s}" for p in players))
    for tpl in tpls:
        row = {}
        for pname, act in players:
            env = CloneEnv({"scenario_seed": 0, "decision_ticks": a.ticks})
            env.forced_template = tpl
            res = []
            for i in range(a.n):
                env.rng = np.random.default_rng(10_000 + i)  # the same states for every player
                env.scenario_rng = env.rng
                torch.manual_seed(10_000 + i)
                obs, _ = env.reset()
                env.rush_lane = lane_of(env.situation.spec)
                first = True
                while True:
                    k, x, y = act(env, obs, first)
                    first = False
                    obs, r, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                          "target_y": np.array([y], np.float32),
                                                          "activate_ability_slot1": 0, "activate_ability_slot2": 0})
                    if info.get("rl_situation_end") or term or trunc:
                        res.append(env.last_situation)
                        if jout:
                            jout.write(json.dumps({"template": tpl.__name__, "seed": 10_000 + i, "player": pname,
                                                   **env.last_situation}, default=str) + "\n")
                        break
            sc = np.array([r["score"] for r in res])
            row[pname] = (sc.mean(), np.median([r["tower_hp_lost"] for r in res]), np.mean([r["elixir_spent"] for r in res]),
                          np.mean([r["unresolved"] for r in res]), np.mean([r.get("spell_vetoes", 0) for r in res]))
        print(f"{tpl.__name__:16s} " + " ".join(f"{v[0]:+.2f} (hp {v[1]:5.0f}, el {v[2]:3.1f}, ur {v[3]:.0%})" for v in row.values()))
        if not a.no_spell_guard:
            print(f"{'  spell vetoes':16s} " + " ".join(f"{v[4]:34.2f}" for v in row.values()))
    if a.reference:
        # the second yardstick: the exact starts Cole beat the scripted defence on, against his own score there
        refs = [json.loads(l) for l in open(ROOT / "data/live/cole_reference.jsonl")]
        tot = {p: [] for p, _ in players}
        for r in refs:
            for pname, act in players:
                env = CloneEnv({"scenario_seed": 0, "decision_ticks": a.ticks})
                env.forced_seed = (sit.BY_NAME[r["situation"]], int(r["seed"]))
                torch.manual_seed(int(r["seed"]))
                obs, _ = env.reset()
                env.rush_lane = lane_of(env.situation.spec)
                first = True
                while True:
                    k, x, y = act(env, obs, first)
                    first = False
                    obs, _, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                          "target_y": np.array([y], np.float32),
                                                          "activate_ability_slot1": 0, "activate_ability_slot2": 0})
                    if info.get("rl_situation_end") or term or trunc:
                        tot[pname].append(env.last_situation["score"])
                        break
        cole = [r["result"]["you"]["score"] for r in refs]
        print(f"\nCole's {len(refs)} reference starts (his score then {np.mean(cole):+.2f}): " +
              ", ".join(f"{p} {np.mean(v):+.2f} (beats Cole on {sum(x > c for x, c in zip(v, cole))})" for p, v in tot.items()))
    print("score (v2) = -(tower HP lost / princess max) - 0.04 * elixir spent + 0.04 * our board left - 0.08 * theirs on our side + 0.15 * tower dealt / princess max"
          " - 0.04 * elixir leaked - 0.04 * wasted spell elixir (P2); tempo kinds per bot77.rl.situations (P3); hp = median tower HP"
          " lost, el = mean elixir spent, ur = share unresolved at the timeout")


if __name__ == "__main__":
    main()
