"""The pocket (Cole, 2026-10-30: "Once you take a tower, you can place higher up"): the engine opens it, and the
network's placement mask (bot77.rl.pocket over the fork's MicroRoyaleNet.placement_mask) offers exactly those tiles."""
import sys
from pathlib import Path

import numpy as np
import pytest

E = pytest.importorskip("clash_royale_env")
ROOT = Path(__file__).resolve().parents[1]
DECK = [2, 81, 5, 43, 9, 95, 7, 3]


@pytest.fixture(scope="module")
def net():
    sys.path.insert(0, str(ROOT / "third_party" / "clash-royale-env"))
    torch = pytest.importorskip("torch")
    import python_ai  # noqa: F401
    from python_ai.models.net import MicroRoyaleNet
    from bot77.rl import pocket
    pocket.install()
    return torch, MicroRoyaleNet()


@pytest.mark.parametrize("slot", [1, 2])
def test_the_network_offers_exactly_the_tiles_the_engine_takes_once_a_tower_falls(net, slot):
    torch, n = net
    g = E.ClashRoyaleEnv(DECK, DECK, 3600)
    g.reset()
    g.destroy_tower(1, slot)
    o = torch.tensor(np.asarray(g.get_observation_for_team(0), dtype=np.float32)).unsqueeze(0)
    for k, card in enumerate(g.get_hand()):
        mask = n.placement_mask(o, torch.tensor([k]))[0]
        legal = [g.is_valid_placement(card, float(x), float(y), 0) for y in range(34) for x in range(18)]
        assert [bool(v) for v in mask.tolist()] == legal, card
    pocket_rows = {(x, y) for y in range(18, 22) for x in (range(9) if slot == 1 else range(9, 18))}
    giant = n.placement_mask(o, torch.tensor([list(g.get_hand()).index(2)]))[0] if 2 in g.get_hand() else None
    if giant is not None:
        assert all(giant[y * 18 + x] for x, y in pocket_rows if g.is_valid_placement(2, float(x), float(y), 0))
