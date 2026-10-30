"""The pocket in the network's placement mask (2026-10-30, Cole: "Once you take a tower, you can place higher up. we
should be doing this for like plays in the pocket").

The engine opens the pocket (Fidelity::pocketDeploy: with an enemy Princess Tower down, that lane's half of their
side, 4 tiles past the river). The fork's MicroRoyaleNet.placement_mask caches legality with a delta for our own
towers only (its tripwire tests, test_placement_legality_is_static, say so), so the network could never pick a pocket
tile. python_ai/ is read-only here; install() wraps the method instead: on top of the fork's mask, the cells the
engine opens for the chosen card when an enemy Princess is down (read from the observation's enemy-building channel at
that tower's centre, as the fork reads ours). The cells come from the engine (is_valid_placement after destroy_tower),
not a copy of its rule. Deterministic in obs, so the PPO ratio's rollout and update masks agree.
"""
from __future__ import annotations

import numpy as np

_TABLE: dict = {}
_CH_ENEMY_BUILDINGS = 7  # observation channels 4-7 are the opponent's; 7 its buildings (MicroRoyaleNet's comment)


def pocket_cells(num_card_ids: int) -> tuple[np.ndarray, list[int]]:
    """(2, num_card_ids + 1, H * W) bool: cells each card may use once the enemy's left / right Princess is down
    that it may not before (the last row: the no-op, nothing); and the flat observation indices of those towers'
    centres in the enemy-building channel."""
    if num_card_ids not in _TABLE:
        import clash_royale_env as E
        H, W = E.ClashRoyaleEnv.BOARD_HEIGHT, E.ClashRoyaleEnv.BOARD_WIDTH
        deck = [0, 1, 2, 3, 4, 5, 6, 7]
        out = np.zeros((2, num_card_ids + 1, H * W), dtype=bool)
        base = E.ClashRoyaleEnv(deck, deck, 3600)
        base.reset()
        down = []
        for slot in (1, 2):
            g = E.ClashRoyaleEnv(deck, deck, 3600)
            g.reset()
            g.destroy_tower(1, slot)
            down.append(g)
        for card in range(num_card_ids):
            for y in range(H):
                for x in range(W):
                    if base.is_valid_placement(card, float(x), float(y), 0):
                        continue
                    for side, g in enumerate(down):
                        out[side, card, y * W + x] = g.is_valid_placement(card, float(x), float(y), 0)
        y = int(E.arena_princess_y(1))
        flat = [_CH_ENEMY_BUILDINGS * H * W + y * W + int(x) for x in (E.ARENA_LEFT_LANE_X, E.ARENA_RIGHT_LANE_X)]
        _TABLE[num_card_ids] = (out, flat)
    return _TABLE[num_card_ids]


def install() -> None:
    """Wrap MicroRoyaleNet.placement_mask once per process (every net built afterwards, and before, uses it)."""
    import torch
    from python_ai.models import net as N
    if getattr(N.MicroRoyaleNet.placement_mask, "_pocket", False):
        return
    fork_mask = N.MicroRoyaleNet.placement_mask

    def placement_mask(self, obs, card_idx):
        mask = fork_mask(self, obs, card_idx)
        if self.placement_cells != self.board_width * self.board_height:
            return mask  # a net that can't address the enemy half has no pocket to open
        table, flat = pocket_cells(self.num_card_ids)
        if not hasattr(self, "_pocket_table") or self._pocket_table.device != obs.device:
            self._pocket_table = torch.as_tensor(table, device=obs.device)
        batch = obs.shape[0]
        onehots = obs[:, self.spatial_size + 1 + self.hand_size:][:, :self.hand_size * self.num_card_ids]
        onehots = onehots.view(batch, self.hand_size, self.num_card_ids)
        ids = torch.where(onehots.sum(dim=-1) <= 0.0, torch.full_like(onehots.argmax(dim=-1), -1), onehots.argmax(dim=-1))
        ids = torch.cat([ids, torch.full((batch, 1), -1, device=obs.device, dtype=ids.dtype)], dim=1)
        chosen = ids.gather(1, card_idx.view(-1, 1)).squeeze(1)
        chosen = torch.where(chosen < 0, torch.full_like(chosen, self.num_card_ids), chosen)  # no-op: nothing opens
        for side in (0, 1):
            down = (obs[:, flat[side]] <= 0.0).view(-1, 1)
            mask = mask | (self._pocket_table[side, chosen] & down)
        return mask

    placement_mask._pocket = True
    N.MicroRoyaleNet.placement_mask = placement_mask
