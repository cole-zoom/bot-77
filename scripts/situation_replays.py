"""Record defence situations frame by frame for viewing (2026-10-28, for Cole to see the scripted setups before
training on them): each template on a few seeds, played by the scripted defence and by a checkpoint, one frame per
decision (1 s): every unit (card, team, position, HP), tower HP, both elixir bars, our play.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/situation_replays.py --ckpt data/models/rl_clones_v2/final.pth \
      --out <dir>/situations.json [--seeds 2]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--review", action="store_true",
                    help="Cole's 1 / 8 / 9 with the scripted opponent (scripted defence and a bridge rush on our side)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    os.environ.update({"CLASH_DECK": "2,81,5,43,9,95,7,3", "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    ckpt = str((ROOT / a.ckpt).resolve())
    sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
    os.chdir(ROOT / "third_party" / "clash-royale-env")
    import python_ai  # noqa: F401
    import torch
    from torch.distributions import Categorical
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.models.policy_io import LSTM_HIDDEN

    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    import situation_exam as ex  # noqa: F401  (the scripted defence lives there; re-declared below to stay standalone)
    torch.set_num_threads(1)
    net = MicroRoyaleNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    net.eval()
    h = {}

    def model(env, obs, first):
        if first:
            h["s"] = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
        o = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            m = net.affordability_mask(o)
            f, e, sp, hi = net.extract_features_hires(o)
            lg, _, _, _, h["s"] = net.step_lstm_and_card(f, h["s"], m)
            k = Categorical(logits=lg).sample()
            x, y = net.cell_to_xy(Categorical(logits=net.placement_given_card(h["s"][0], e, k, o, sp, hires_map=hi)).sample())
        return int(k.item()), float(x.item()), float(y.item())

    def defend(env, obs, first):
        return sit.scripted_defence(env)

    def rush(env, obs, first):
        # the failure Cole's 1 and 9 catch: everything affordable at the bridge
        hand, el = list(env.game.get_hand()), env.game.get_elixir()
        ok = [(sit.COST.get(c, 9), i) for i, c in enumerate(hand) if sit.COST.get(c, 9) <= el and c not in (sit.FIREBALL, sit.ARROWS)]
        if not ok:
            return 4, 0.0, 0.0
        lane = env.situation.spec.get("lane", sit.LANE_X[0]) if env.situation else sit.LANE_X[0]
        return max(ok)[1], float(lane), float(env.MAX_Y)

    names = {}

    def frame(env, play=None):
        units = []
        for u in env.game.get_units():
            names[u.card_id] = u.name
            units.append({"id": u.id, "name": u.name, "team": u.team, "x": round(u.x, 2), "y": round(u.y, 2),
                          "hp": round(u.hp), "flying": bool(u.flying), "building": bool(u.building), "tower": bool(u.tower)})
        opp = env.situation.opp_state.get("plays", []) if env.situation else []
        new = opp[frame.seen:] if opp else []
        frame.seen = len(opp)
        return {"opp_play": [{"card": names.get(c, card_name.get(c, str(c))), "x": x, "y": y} for _, c, x, y in new],
                "units": units, "elixir": [round(env.game.get_elixir_for_team(0), 1), round(env.game.get_elixir_for_team(1), 1)],
                "towers": [[max(0, env.game.get_tower_hp(t, s)) for s in range(3)] for t in (0, 1)], "play": play}

    card_name = {sit.MINI_PEKKA: "Mini P.E.K.K.A", sit.GOBLIN_HUT: "Goblin Hut", sit.MEGA_MINION: "Mega Minion",
                 sit.BOMBER: "Bomber", sit.FIREBALL: "Fireball", sit.ARROWS: "Arrows", sit.GIANT: "Giant",
                 sit.BATTLE_RAM: "Battle Ram"}
    frame.seen = 0
    from bot77.sim import cards as sc
    names.update({i: c["name"] for i, c in sc.engine_cards().items()})
    episodes = []
    tpls = sit.REVIEW_TEMPLATES if a.review else [sit.hog_rider, sit.battle_ram, sit.melee, sit.supported_push]
    players = [("Scripted defence", defend), ("Bridge rush", rush)] if a.review else [("Scripted defence", defend), ("Round-2 model", model)]
    shown: dict[str, int] = {}
    for tpl in tpls:
        for s in range(a.seeds * (30 if a.review else 1)):
            if a.review:  # each variant (the opening's four plans, the two pushes) a.seeds times
                probe = CloneEnv({"scenario_seed": 0})
                probe.forced_template = tpl
                probe.rng = np.random.default_rng(20_000 + s)
                probe.scenario_rng = probe.rng
                probe.reset()
                vname = probe.situation.spec["name"]
                if shown.get(vname, 0) >= a.seeds:
                    continue
                shown[vname] = shown.get(vname, 0) + 1
            for pname, act in players:
                frame.seen = 0
                env = CloneEnv({"scenario_seed": 0})
                env.forced_template = tpl
                env.rng = np.random.default_rng(20_000 + s)
                env.scenario_rng = env.rng
                torch.manual_seed(s)
                obs, _ = env.reset()
                spec = env.situation.spec
                frames = [frame(env)]
                hand0 = [card_name.get(c, str(c)) for c in env.game.get_hand()]
                first = True
                while True:
                    hand = list(env.game.get_hand())
                    k, x, y = act(env, obs, first)
                    first = False
                    play = {"card": card_name.get(hand[k], str(hand[k])), "x": round(x, 1), "y": round(y, 1)} if k < 4 else None
                    obs, r, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                                          "target_y": np.array([y], np.float32),
                                                          "activate_ability_slot1": 0, "activate_ability_slot2": 0})
                    frames.append(frame(env, play))
                    if info.get("rl_situation_end") or term or trunc:
                        break
                episodes.append({"template": spec["name"], "seed": s, "player": pname, "hand": hand0,
                                 "start_elixir": [round(spec["own_elixir"], 1), round(spec["opp_elixir"], 1)],
                                 "result": env.last_situation, "frames": frames})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(episodes))
    print(f"{len(episodes)} episodes -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
