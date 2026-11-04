"""A checkpoint playing team 0 in an engine the caller steps (2026-11-04, scripts/deck_play.py's "you play the deck"
mode: Cole pilots a deck bot's deck against the model).

Decides every tick it's asked (0.1 s), samples as live (bot77.live.selfplay.SelfPlayPolicy) and as the sim exams do
(scripts/match_eval.py), with the pocket in the placement mask and the spell guard (a Fireball / Arrows that would touch
nothing of theirs is a wait), as in training and live. No wait nudge: the live model's run20v5raw setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from bot77.rl import guards, pocket

FORK = Path(__file__).resolve().parents[3] / "third_party" / "clash-royale-env"
LIVE_CKPT = Path(__file__).resolve().parents[3] / "data/models/bc_school_v5/epoch03.pth"  # run20v5raw (2026-11-03)


class ModelPlayer:
    def __init__(self, ckpt: str | Path = LIVE_CKPT, seed: int = 0):
        if str(FORK) not in sys.path:
            sys.path.insert(0, str(FORK))
        import torch
        from python_ai.models.policy_io import LSTM_HIDDEN, load_net
        pocket.install()
        self.torch = torch
        self.net = load_net(str(ckpt), "cpu", verbose=False)
        self.net.eval()
        self.gen = torch.Generator().manual_seed(int(seed))
        self.h = (torch.zeros(1, LSTM_HIDDEN), torch.zeros(1, LSTM_HIDDEN))
        self.ckpt = Path(ckpt).name

    def _sample(self, logits):
        p = self.torch.softmax(logits, -1)
        return self.torch.multinomial(p, 1, generator=self.gen).view(-1)

    def act(self, game):
        """This tick's move for team 0: (hand index, x, y) on the board, or None to wait."""
        torch = self.torch
        o = torch.tensor(np.asarray(game.get_observation_for_team(0), dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            m = self.net.affordability_mask(o)
            f, e, sp, hi = self.net.extract_features_hires(o)
            lg, _, _, _, self.h = self.net.step_lstm_and_card(f, self.h, m)
            k = self._sample(lg)
            if int(k.item()) >= 4:
                return None
            cell = self._sample(self.net.placement_given_card(self.h[0], e, k, o, sp, hires_map=hi))
            x, y = self.net.cell_to_xy(cell)
        k, x, y = int(k.item()), float(x.item()), float(y.item())
        hand = list(game.get_hand())
        if k >= len(hand) or guards.spell_vetoed(hand[k], x, y, game.get_units()):
            return None
        return k, x, y
