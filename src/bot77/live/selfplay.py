"""The self-play agent in the real game (2026-10-24 overnight test; docs/simulator.md S6).

The agent was trained in the simulator (the ClashRoyaleEnv fork's PPO stack, scripts/selfplay_train.sh). Here, once a
second (its training cadence: 10 engine ticks), the live screen read (bot77.live.state) is mirrored into an engine
(bot77.sim: clock, our hand and elixir, every detected unit, the opponent's plays for its card-cycle inputs), the
engine's observation goes through the trained net, and its card + cell become a tap. The net's LSTM memory is kept
across decisions and reset each game.

Not mirrored: tower HP (the live bot doesn't read it: towers count as full), units' HP (full), the opponent's elixir
(estimated: 5 + regeneration - what we saw them spend).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from bot77.features.frames import FEET_DY
from bot77.live.policy import Action
from bot77.live.state import LiveState
from bot77.sim import cards as sc
from bot77.sim.engine import Sim  # noqa: F401  (type)
from bot77.sim.geometry import engine_to_ours
from bot77.sim.mirror import mirror
from bot77.sim.scenario import REAL_ELIXIR_S, real_multiplier

FORK = Path(__file__).resolve().parents[3] / "third_party" / "clash-royale-env"
DECK = ["Giant", "Mini P.E.K.K.A", "Musketeer", "Goblin Hut", "Fireball", "Arrows", "Skeletons", "Minions"]
DECISION_S = 1.0


class SelfPlayPolicy:
    def __init__(self, cards: list[dict], ckpt: str, sample: bool = True):
        if str(FORK) not in sys.path:
            sys.path.insert(0, str(FORK))
        import torch
        from python_ai.models.policy_io import LSTM_HIDDEN, load_net
        self.torch = torch
        self.net = load_net(ckpt, "cpu", verbose=False)
        self.hidden = LSTM_HIDDEN
        self.sample = sample
        self.cards = {c["id"]: c for c in cards}
        self.deck = [sc.engine_id(n) for n in DECK]
        self.reset()

    def reset(self) -> None:
        self.hx = self.torch.zeros(1, self.hidden)
        self.cx = self.torch.zeros(1, self.hidden)
        self.last_t = -1e9
        self.noted = 0
        self.log: list[dict] = []

    def record_own(self, t: float, card: str) -> None:  # the loop's hook (Policy has it too)
        pass

    def _mirror(self, st: LiveState, opp_plays: list[dict]) -> Sim | None:
        """The screen read -> an engine, through bot77.sim.mirror (the same function the imitation data was built
        with). The live bot doesn't read tower HP (towers count as full, except one missing from the detections)
        or the opponent's elixir (estimated)."""
        hand = [sc.engine_id(self.cards[c]["name"]) if c in self.cards else None for c in st.hand]
        if None in hand or st.elixir is None or st.elapsed_s is None:
            return None
        opp_deck = list(dict.fromkeys(e for e in (sc.engine_id(p["card"]) for p in opp_plays) if e is not None))
        spent = sum(sc.cost(e) for e in (sc.engine_id(p["card"]) for p in opp_plays) if e is not None)
        regen = sum(real_multiplier(t) for t in range(int(st.elapsed_s))) / REAL_ELIXIR_S
        units = [{"card": u.card, "cls": u.cls, "team": "own" if u.team == "friendly" else "opp", "x": u.x,
                  "y": u.y + FEET_DY} for u in st.units
                 if u.card and u.team in ("friendly", "enemy") and "tower" not in u.cls and "bar" not in u.cls]
        opp_played = [e for e in (sc.engine_id(p["card"]) for p in opp_plays) if e is not None]
        return mirror(self.deck, opp_deck[:8], float(st.elapsed_s), hand, st.elixir, 5.0 + regen - spent, units,
                      None, opp_played)

    def decide(self, st: LiveState, opp_plays: list[dict]) -> Action | None:
        if st.t - self.last_t < DECISION_S:
            return None
        self.last_t = st.t
        sim = self._mirror(st, opp_plays)
        if sim is None:
            return None
        torch = self.torch
        obs = torch.tensor(np.asarray(sim.env.get_observation_for_team(0), dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            mask = self.net.affordability_mask(obs)
            feats, emb, spatial, hires = self.net.extract_features_hires(obs)
            logits, _, _, value, (self.hx, self.cx) = self.net.step_lstm_and_card(feats, (self.hx, self.cx), mask)
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample() if self.sample else logits.argmax(-1)
            slot = int(idx.item())
            self.log.append({"t": st.t, "value": float(value.item()), "slot": slot})
            if slot >= 4:  # the no-op arm: wait
                return None
            pl = self.net.placement_given_card(self.hx, emb, idx, obs, spatial, hires_map=hires)
            cell = torch.distributions.Categorical(logits=pl).sample() if self.sample else pl.argmax(-1)
            ex, ey = self.net.cell_to_xy(cell)
        x, y = engine_to_ours(float(ex.item()), float(ey.item()))
        card = self.cards[st.hand[slot]]["name"]
        probs = torch.softmax(logits, -1)[0].tolist()
        return Action(slot, card, x, y, "self-play agent", [round(p, 4) for p in probs[:4]])
