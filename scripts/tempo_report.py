"""The tempo report (docs/plan-elixir-spells.md P4, 2026-10-29): does a checkpoint know when to spend, beyond the exam
mean. On the exam's held-out seeds (10_000+), sampled like the exam (torch seeded per start), spell guard on:

1. per tempo pair, both halves' mean scores and the share of starts where it did the right thing:
   - press halves: a play within PRESS_S (3 s) of the start;
   - chill halves: before their first troop reaches our side, no play below 9 elixir and at most one play.
2. the play rate by state, over every template: plays per 10 s while calm (no enemy troop on our side, lead < +3),
   under a threat (an enemy troop on our side, y < 19), and ahead by 3+ (calm, our elixir - theirs >= 3). One flat
   rate means it learned a knob, not a state.
3. elixir leaked and spell-guard vetoes (spells it tried on nothing) per template.

  uv run --no-sync --with gymnasium==1.3.0 python scripts/tempo_report.py --ckpt A_cal50.pth B_cal50.pth [--n 50]
  (add "script" as a ckpt to report the scripted policy)
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "third_party" / "clash-royale-env"
PRESS_S = 3.0
_S: dict = {}


def setup():
    os.environ.update({"CLASH_DECK": "2,81,5,43,9,95,7,3", "RL_CLONE_CKPT": str(ROOT / "data/models/bc_v2/frozen.pth"),
                       "RL_PRO_DECKS": str(ROOT / "data/cache/rl/pro_decks.json"), "RL_SITUATION_SHARE": "1"})
    sys.path.insert(0, str(FORK))
    os.chdir(FORK)
    import python_ai  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from bot77.rl import situations as sit
    from bot77.rl.clone_env import CloneEnv
    _S.update(torch=torch, sit=sit, CloneEnv=CloneEnv, nets={})


def player(ckpt):
    torch, sit = _S["torch"], _S["sit"]
    if ckpt == "script":
        return lambda env, obs, first: sit.scripted_defence(env)
    from torch.distributions import Categorical
    from python_ai.models.net import MicroRoyaleNet
    from python_ai.models.policy_io import LSTM_HIDDEN
    if ckpt not in _S["nets"]:
        net = MicroRoyaleNet()
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
        net.eval()
        _S["nets"][ckpt] = net
    net, st = _S["nets"][ckpt], {}

    def act(env, obs, first):
        if first:
            st["h"] = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
        o = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            m = net.affordability_mask(o)
            f, e, sp, hi = net.extract_features_hires(o)
            lg, _, _, _, st["h"] = net.step_lstm_and_card(f, st["h"], m)
            k = Categorical(logits=lg).sample()
            cell = Categorical(logits=net.placement_given_card(st["h"][0], e, k, o, sp, hires_map=hi)).sample()
            x, y = net.cell_to_xy(cell)
        return int(k.item()), float(x.item()), float(y.item())
    return act


def run(args):
    ckpt, tpl, seed = args
    if not _S:
        setup()
    torch, sit = _S["torch"], _S["sit"]
    env = _S["CloneEnv"]({"scenario_seed": 0, "decision_ticks": 1})
    env.forced_template = {t.__name__: t for t in sit.TEMPLATES}[tpl]
    env.rng = np.random.default_rng(seed)
    env.scenario_rng = env.rng
    torch.manual_seed(seed)
    obs, _ = env.reset()
    act = player(ckpt)
    first, steps, plays = True, [], []
    while True:
        g = env.game
        threat = any(u.team == 1 and not u.tower and not u.building and u.y < 19.0 for u in g.get_units())
        el = float(g.get_elixir())
        lead = el - float(g.get_elixir_for_team(1))
        state = "threat" if threat else ("ahead3" if lead >= 3.0 else "calm")
        t = env.situation.seconds
        k, x, y = act(env, obs, first)
        first = False
        spent0 = float(g.get_elixir_spent(0))
        card = int(g.get_hand()[k]) if k < 4 else None
        obs, _, term, trunc, info = env.step({"card_index": k, "target_x": np.array([x], np.float32),
                                              "target_y": np.array([y], np.float32),
                                              "activate_ability_slot1": 0, "activate_ability_slot2": 0})
        played = float(env.game.get_elixir_spent(0)) > spent0 + 1e-6
        steps.append(state)
        if played:
            plays.append((round(t, 1), card, round(el, 1), state))
        if info.get("rl_situation_end") or term or trunc:
            r = env.last_situation
            return {"ckpt": ckpt, "template": tpl, "seed": seed, "score": r["score"], "leaked": r["elixir_leaked"],
                    "vetoes": r.get("spell_vetoes", 0), "plays": plays, "steps": steps}


def right_thing(r, half):
    if half == "press":
        return any(p[0] <= PRESS_S for p in r["plays"])
    first_threat = next((i / 10 for i, s in enumerate(r["steps"]) if s == "threat"), 1e9)
    before = [p for p in r["plays"] if p[0] < first_threat]
    return len(before) <= 1 and all(p[2] >= 9.0 for p in before)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = str(Path(a.out).resolve()) if a.out else None  # before setup() changes directory
    setup()
    sit = _S["sit"]
    ckpts = [c if c == "script" else str((ROOT / c).resolve()) for c in a.ckpt]
    tpls = [t.__name__ for t in sit.TEMPLATES]
    jobs = [(c, t, 10_000 + i) for c in ckpts for t in tpls for i in range(a.n)]
    with Pool(a.workers) as pool:
        res = pool.map(run, jobs, chunksize=2)
    if out:
        with open(out, "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
    name = {c: (c if c == "script" else Path(c).parent.name + "/" + Path(c).stem) for c in ckpts}
    print("1. tempo pairs (exam seeds): mean score / share doing the right thing (press: a play within 3 s; "
          "chill: before their first troop crosses, at most one play and none below 9 elixir)")
    in_tpl = set(tpls)
    for ch, pr in sit.TEMPO_PAIRS:
        if ch.__name__ not in in_tpl:
            continue
        for c in ckpts:
            cells = []
            for t, half in ((ch.__name__, "chill"), (pr.__name__, "press")):
                rs = [r for r in res if r["ckpt"] == c and r["template"] == t]
                cells.append(f"{t} {np.mean([r['score'] for r in rs]):+.2f} / {np.mean([right_thing(r, half) for r in rs]):.0%}")
            print(f"  {name[c]:36s} " + "   ".join(cells))
    for c in ckpts:
        rs = [r for r in res if r["ckpt"] == c and r["template"] == "spell_finish"]
        if rs:
            fin = np.mean([any(p[1] in (sit.FIREBALL, sit.ARROWS) and p[0] <= PRESS_S for p in r["plays"]) for r in rs])
            print(f"  {name[c]:36s} spell_finish {np.mean([r['score'] for r in rs]):+.2f} / spell within 3 s {fin:.0%}")
    print("\n2. plays per 10 s by state, all templates (time in state, s)")
    for c in ckpts:
        rs = [r for r in res if r["ckpt"] == c]
        out = []
        for s in ("calm", "threat", "ahead3"):
            secs = sum(r["steps"].count(s) for r in rs) / 10
            n = sum(1 for r in rs for p in r["plays"] if p[3] == s)
            out.append(f"{s} {10 * n / max(secs, 1e-9):.2f} ({secs:.0f})")
        print(f"  {name[c]:36s} " + "   ".join(out))
    print("\n3. elixir leaked / spell vetoes a situation, per template")
    print(f"  {'template':20s}" + "".join(f"{name[c][-24:]:>26s}" for c in ckpts))
    for t in tpls:
        print(f"  {t:20s}" + "".join(
            f"{np.mean([r['leaked'] for r in res if r['ckpt'] == c and r['template'] == t]):12.2f} / "
            f"{np.mean([r['vetoes'] for r in res if r['ckpt'] == c and r['template'] == t]):9.2f}  " for c in ckpts))


if __name__ == "__main__":
    main()
