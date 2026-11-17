"""The policy net with the new input block (2026-11-16; bot77.live.extra_obs), built on the fork's MicroRoyaleNet
without touching it (python_ai is read-only).

ExtraObsNet reads observations of BASE_DIM + extra_dim floats. The first BASE_DIM go through MicroRoyaleNet exactly
as before (every one of its reads is by a forward offset, so the block after them changes nothing there); the block
goes through a small MLP whose output is added to the LSTM's input (CNN + scalar features). The MLP's last layer
starts at zero, so a net initialised from an old checkpoint gives the old net's outputs on any input, bit for bit
(tests/test_extra_net.py), and training grows the block's influence from there. Observations narrower than
BASE_DIM + extra_dim (old data) read an all-zero block.

Checkpoints: {"model": state_dict, "extra_obs": {"dim", "hidden", "layout"}}. load_policy() takes either kind;
init_from() starts an ExtraObsNet from an old checkpoint (or a new one).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

FORK = Path(__file__).resolve().parents[3] / "third_party" / "clash-royale-env"
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))
from python_ai.models.net import MicroRoyaleNet  # noqa: E402

from bot77.live.extra_obs import BASE_DIM, EXTRA_DIM, LAYOUT  # noqa: E402

EXTRA_HIDDEN = 64


class ExtraObsNet(MicroRoyaleNet):
    def __init__(self, extra_dim: int = EXTRA_DIM, extra_hidden: int = EXTRA_HIDDEN, **kw):
        super().__init__(**kw)
        self.base_obs_dim = self.spatial_size + self.scalar_size
        assert self.base_obs_dim == BASE_DIM, (self.base_obs_dim, BASE_DIM)
        self.extra_dim, self.extra_hidden = extra_dim, extra_hidden
        self.obs_dim = self.base_obs_dim + extra_dim
        self.extra_mlp = nn.Sequential(nn.Linear(extra_dim, extra_hidden), nn.ReLU(),
                                       nn.Linear(extra_hidden, self.lstm_input_dim))
        nn.init.zeros_(self.extra_mlp[-1].weight)
        nn.init.zeros_(self.extra_mlp[-1].bias)

    def extra_part(self, obs: torch.Tensor) -> torch.Tensor:
        e = obs[:, self.base_obs_dim:self.obs_dim]
        if e.shape[1] < self.extra_dim:  # old data: no block
            e = torch.cat([e, e.new_zeros(e.shape[0], self.extra_dim - e.shape[1])], dim=1)
        return e

    def extract_features_hires(self, obs):
        combined, card_embeds, spatial_map, hires_map = super().extract_features_hires(obs[:, :self.base_obs_dim])
        return combined + self.extra_mlp(self.extra_part(obs)), card_embeds, spatial_map, hires_map

    def meta(self) -> dict:
        return {"dim": self.extra_dim, "hidden": self.extra_hidden, "layout": LAYOUT}


def obs_width(net) -> int:
    return getattr(net, "obs_dim", BASE_DIM)


def _state(ckpt):
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def load_policy(path: str, device="cpu", verbose: bool = False):
    """Any checkpoint -> an eval-mode net: an ExtraObsNet when it was saved with the block, else the fork's loader."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = _state(ckpt)
    meta = ckpt.get("extra_obs") if isinstance(ckpt, dict) else None
    if meta is None and any(k.startswith("extra_mlp.") for k in state):
        meta = {"dim": state["extra_mlp.0.weight"].shape[1], "hidden": state["extra_mlp.0.weight"].shape[0]}
    if meta is None:
        from python_ai.models.policy_io import load_net
        return load_net(path, device, verbose=verbose)
    net = ExtraObsNet(meta["dim"], meta["hidden"]).to(device)
    net.load_state_dict(state)
    if verbose:
        print(f"  loaded {path} (input block {meta})")
    return net.eval()


def init_from(path: str | None, device="cpu", extra_dim: int = EXTRA_DIM, extra_hidden: int = EXTRA_HIDDEN) -> ExtraObsNet:
    """An ExtraObsNet from an old checkpoint (the block's MLP fresh, its output zero) or from a new one."""
    net = ExtraObsNet(extra_dim, extra_hidden).to(device)
    if path:
        state = _state(torch.load(path, map_location=device, weights_only=False))
        missing, unexpected = net.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith("extra_mlp.") for k in missing), missing
    return net
