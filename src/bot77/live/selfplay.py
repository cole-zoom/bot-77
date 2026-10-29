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
from bot77.rl import guards
from bot77.sim import cards as sc
from bot77.sim.engine import Sim  # noqa: F401  (type)
from bot77.sim.geometry import engine_to_ours
from bot77.sim.mirror import mirror
from bot77.sim.scenario import REAL_ELIXIR_S, real_multiplier

FORK = Path(__file__).resolve().parents[3] / "third_party" / "clash-royale-env"
DECISION_S = 1.0
# the full-bar guard (plan P1, live only, temporary): at >= FULL_BAR elixir for FULL_BAR_S seconds, nothing of theirs
# on our side, and the model still choosing wait, play its most likely affordable card. Never in training or the exam
FULL_BAR, FULL_BAR_S = 9.8, 3.0


class SelfPlayPolicy:
    def __init__(self, cards: list[dict], ckpt: str, sample: bool = True, decision_s: float = DECISION_S,
                 full_bar_guard: bool = False):
        if str(FORK) not in sys.path:
            sys.path.insert(0, str(FORK))
        import torch
        from python_ai.models.policy_io import LSTM_HIDDEN, load_net
        self.torch = torch
        self.net = load_net(ckpt, "cpu", verbose=False)
        self.hidden = LSTM_HIDDEN
        self.sample = sample
        self.decision_s = decision_s  # 1.0 for the 1 s models, 0.1 for the 0.1 s ones (2026-10-28)
        self.cards = {c["id"]: c for c in cards}
        self.full_bar_guard = full_bar_guard
        self.reset()

    def reset(self) -> None:
        # our deck, learned from the cards seen in hand this game (the deck changes between sessions; nothing here
        # assumes one). Until all 8 are seen, Sim pads it with filler cards that are never in the hand.
        self.deck: list[int] = []
        self.hx = self.torch.zeros(1, self.hidden)
        self.cx = self.torch.zeros(1, self.hidden)
        self.last_t = -1e9
        self.noted = 0
        self.log: list[dict] = []
        self.full_since: float | None = None

    def record_own(self, t: float, card: str) -> None:  # the loop's hook (Policy has it too)
        pass

    def _mirror(self, st: LiveState, opp_plays: list[dict]) -> Sim | None:
        """The screen read -> an engine, through bot77.sim.mirror (the same function the imitation data was built
        with). The live bot doesn't read tower HP (towers count as full, except one missing from the detections)
        or the opponent's elixir (estimated)."""
        hand = [sc.engine_id(self.cards[c]["name"]) if c in self.cards else None for c in st.hand]
        if st.elixir is None or st.elapsed_s is None or all(h is None for h in hand):
            return None
        for e in hand + ([sc.engine_id(self.cards[st.next]["name"])] if st.next in self.cards else []):
            if e is not None and e not in self.deck:
                self.deck.append(e)
        # an unreadable slot (a card the hand reader doesn't know yet): a stand-in from the deck, never played. Kept
        # out of self.deck: stand-ins added there pushed the deck past 8, the engine kept the first 8, a real card
        # fell out, set_hand failed and the agent stopped deciding for the rest of the game (2026-10-27)
        self.blind = [i for i, h in enumerate(hand) if h is None]
        for i in self.blind:
            hand[i] = next((d for d in self.deck if d not in hand), None) or next(
                e for e in sc.engine_cards() if not sc.engine_cards()[e]["evolution"] and e not in hand
                and not sc.engine_cards()[e]["is_hero"] and not sc.engine_cards()[e]["is_champion"])
        # the engine's deck: the hand first (it must hold them), then the rest seen, 8 at most (a misread can add a 9th)
        deck = list(dict.fromkeys(hand + self.deck))[:8]
        opp_deck = list(dict.fromkeys(e for e in (sc.engine_id(p["card"]) for p in opp_plays) if e is not None))
        spent = sum(sc.cost(e) for e in (sc.engine_id(p["card"]) for p in opp_plays) if e is not None)
        regen = sum(real_multiplier(t) for t in range(int(st.elapsed_s))) / REAL_ELIXIR_S
        units = [{"card": u.card, "cls": u.cls, "team": "own" if u.team == "friendly" else "opp", "x": u.x,
                  "y": u.y + FEET_DY} for u in st.units
                 if u.card and u.team in ("friendly", "enemy") and "tower" not in u.cls and "bar" not in u.cls]
        opp_played = [e for e in (sc.engine_id(p["card"]) for p in opp_plays) if e is not None]
        return mirror(deck, opp_deck[:8], float(st.elapsed_s), hand, st.elixir, 5.0 + regen - spent, units,
                      None, opp_played)

    def decide(self, st: LiveState, opp_plays: list[dict]) -> Action | None:
        if st.t - self.last_t < self.decision_s - 1e-3:
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
            probs = torch.softmax(logits, -1)[0].tolist()
            entry = {"t": round(st.t, 1), "clock": st.elapsed_s, "elixir": st.elixir, "value": float(value.item()),
                     "slot": slot, "p": [round(p, 3) for p in probs], "blind": list(self.blind),
                     "hand": list(st.hand)}
            self.log.append(entry)
            units = sim.env.get_units()
            # the full bar: how long we've sat at it
            self.full_since = (self.full_since if self.full_since is not None else st.t) if st.elixir >= FULL_BAR else None
            reason = "self-play agent"
            if slot >= 4 or slot in getattr(self, "blind", []):  # the no-op arm (or an unreadable card): wait
                calm = not any(u.team == 1 and not u.tower and u.y < 17.5 for u in units)
                if not (self.full_bar_guard and calm and self.full_since is not None
                        and st.t - self.full_since >= FULL_BAR_S - 1e-6):
                    return None
                # the full-bar guard: its most likely affordable, readable card (the spell guard still applies)
                ok = mask[0, :4].tolist()
                order = [i for i in sorted(range(4), key=lambda i: -probs[i]) if ok[i] and i not in self.blind]
                for i in order:
                    idx = torch.tensor([i])
                    pl = self.net.placement_given_card(self.hx, emb, idx, obs, spatial, hires_map=hires)
                    cell = torch.distributions.Categorical(logits=pl).sample() if self.sample else pl.argmax(-1)
                    ex, ey = self.net.cell_to_xy(cell)
                    if guards.spell_vetoed(self._hand_engine(st)[i],
                                           float(ex.item()), float(ey.item()), units):
                        continue
                    slot, reason = i, "full-bar guard"
                    entry["guard"] = "full_bar"
                    entry["slot_guard"] = i
                    break
                else:
                    return None
            else:
                pl = self.net.placement_given_card(self.hx, emb, idx, obs, spatial, hires_map=hires)
                cell = torch.distributions.Categorical(logits=pl).sample() if self.sample else pl.argmax(-1)
                ex, ey = self.net.cell_to_xy(cell)
                # the spell guard: a Fireball / Arrows that touches nothing of theirs is a wait
                if guards.spell_vetoed(self._hand_engine(st)[slot], float(ex.item()), float(ey.item()), units):
                    entry["guard"] = "spell_veto"
                    return None
        x, y = engine_to_ours(float(ex.item()), float(ey.item()))
        card = self.cards[st.hand[slot]]["name"]
        return Action(slot, card, x, y, reason, [round(p, 4) for p in probs[:4]])

    def _hand_engine(self, st: LiveState) -> list[int]:
        """The hand as engine ids (-1 for an unreadable slot)."""
        out = []
        for c in st.hand:
            e = sc.engine_id(self.cards[c]["name"]) if c in self.cards else None
            out.append(-1 if e is None else e)
        return out
