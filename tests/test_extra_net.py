"""ExtraObsNet initialised from v8 is v8: same card and placement logits, value and LSTM state, bit for bit, on real
shadow episodes with a random input block appended (2026-11-16, bot77.rl.extra_net). Needs the local checkpoint and
shadow data (data/ is not in git): skipped without them."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data/models/bc_school_v8/epoch03.pth"
SHADOW = sorted((ROOT / "data/cache/bc/shadow_t01_held").glob("*.npz"))
pytestmark = pytest.mark.skipif(not CKPT.exists() or not SHADOW, reason="needs v8 and the shadow episodes (local data)")

sys.path.insert(0, str(ROOT / "scripts"))
STEPS = 400


@pytest.fixture(scope="module")
def nets():
    from bot77.rl.extra_net import init_from, load_policy
    base = load_policy(str(CKPT))
    new = init_from(str(CKPT)).eval()
    return base, new


def episodes(n=2):
    rng = np.random.default_rng(0)
    from bot77.live.extra_obs import EXTRA_DIM
    for f in SHADOW[:n]:
        d = np.load(f)
        obs = d["obs"][:STEPS].astype(np.float32)
        block = rng.random((len(obs), EXTRA_DIM), dtype=np.float32)
        yield obs, np.concatenate([obs, block], 1), d["card"][:STEPS]


def test_same_logits_on_shadow_episodes(nets):
    from train_bc import sequence_loss
    base, new = nets
    for obs, obs_x, card in episodes():
        outs = []
        for net, o in ((base, obs), (new, obs_x)):
            ot = torch.tensor(o).unsqueeze(1)
            c = torch.tensor(card).unsqueeze(1)
            with torch.no_grad():
                _, _, (cl, pl, _, _) = sequence_loss(net, ot, c, torch.full_like(c, -1), torch.ones_like(c, dtype=torch.bool),
                                                     torch.ones_like(c, dtype=torch.float32))
            outs.append((cl, pl))
        assert torch.equal(outs[0][0], outs[1][0])
        assert torch.equal(torch.nan_to_num(outs[0][1]), torch.nan_to_num(outs[1][1]))


def test_same_live_step(nets):
    """The live path (SelfPlayPolicy.decide): features, LSTM step, card head, value, placement."""
    base, new = nets
    obs, obs_x, card = next(episodes(1))
    hb = (torch.zeros(1, 256), torch.zeros(1, 256))
    hn = (torch.zeros(1, 256), torch.zeros(1, 256))
    with torch.no_grad():
        for i in range(0, 200, 7):
            ob, on = torch.tensor(obs[i:i + 1]), torch.tensor(obs_x[i:i + 1])
            res = []
            for net, o, h in ((base, ob, hb), (new, on, hn)):
                mask = net.affordability_mask(o)
                feats, emb, spatial, hires = net.extract_features_hires(o)
                logits, _, _, value, h2 = net.step_lstm_and_card(feats, h, mask)
                pl = net.placement_given_card(h2[0], emb, torch.tensor([0]), o, spatial, hires_map=hires)
                res.append((logits, value, pl, h2))
            (lb, vb, pb, hb), (ln, vn, pn, hn) = res
            assert torch.equal(lb, ln) and torch.equal(vb, vn) and torch.equal(hb[0], hn[0])
            assert torch.equal(torch.nan_to_num(pb), torch.nan_to_num(pn))


def test_old_data_reads_a_zero_block(nets):
    _, new = nets
    obs, obs_x, _ = next(episodes(1))
    o = torch.tensor(obs[:5])
    z = torch.tensor(np.concatenate([obs[:5], np.zeros((5, new.extra_dim), np.float32)], 1))
    with torch.no_grad():
        assert torch.equal(new.extract_features_hires(o)[0], new.extract_features_hires(z)[0])


def test_the_block_can_learn():
    """Not vacuous: one optimiser step and the block moves the features."""
    from bot77.rl.extra_net import init_from
    net = init_from(str(CKPT))
    obs, obs_x, _ = next(episodes(1))
    o = torch.tensor(obs_x[:32])
    f0 = net.extract_features_hires(o)[0].detach()
    opt = torch.optim.SGD(net.extra_mlp.parameters(), lr=0.1)
    loss = net.extract_features_hires(o)[0].pow(2).mean() * -1
    loss.backward()
    opt.step()
    assert not torch.equal(f0, net.extract_features_hires(o)[0].detach())
