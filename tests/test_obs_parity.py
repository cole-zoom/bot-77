"""The live path's observation against the engine's own (scripts/obs_parity.py; after Hasty-CR's test_rl_policy.py,
MIT): a moment rebuilt through bot77.sim.mirror from a perfect read must match the running match's observation where
the mirror can know everything (clock, multiplier, tower HP, the opponent's cycle, the river), and put a one-body unit
where it is. Known, measured gaps are pinned too, so fixing one shows up here: elixir spent (the engine has no setter,
so the mirror reads 0) and squads (re-formed at their centre, refilled to full)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import obs_parity as op  # noqa: E402
from bot77.live.rebuild import SCAL0  # noqa: E402
from bot77.sim.engine import Sim  # noqa: E402
from bot77.sim.mirror import mirror, observation  # noqa: E402
from bot77.sim.scenario import _squad_size  # noqa: E402


@pytest.fixture(scope="module")
def moments():
    rng = np.random.default_rng(0)
    own, opp = op.ids(op.OWN), op.ids(op.OPPS[0])
    sim = Sim.new(own, opp)
    sim.reset()
    played, out = [], []
    while sim.t < 60:
        for team in ("own", "opp"):
            if rng.random() < 0.04:
                c = op.play_random(sim, team, rng)
                if c is not None and team == "opp":
                    played.append((c, sim.t))
        sim.step(1)
        if round(sim.t * 10) % 30:
            continue
        units, towers, _ = op.perfect_read(sim, unit_hp=True)
        reb = mirror(own, opp, sim.t, sim.hand("own"), sim.elixir("own"), sim.elixir("opp"), units, towers,
                     [(e, sim.t - t) for e, t in played], unit_hp=True)
        assert reb is not None
        out.append((sim.snapshot(), observation(sim), reb, observation(reb)))
    return out


def test_sections_the_mirror_knows_exactly_match(moments):
    for _, nat, _, reb in moments:
        diff = op.compare(nat, reb)
        for s in ("scalar time", "scalar multiplier", "scalar tower HP", "opp cycle", "board river ch8"):
            assert diff[s]["diff"] == 0, s


def test_one_body_units_land_where_they_were(moments):
    d = []
    for sim, _, reb, _ in moments:
        ru = [u for u in reb.units() if u["card_id"] >= 0 and not u["spell"]]
        for u in sim.units():
            if u["card_id"] < 0 or u["spell"] or u["attached_to"] >= 0 or _squad_size(u["card_id"]) != 1:
                continue
            same = [np.hypot(v["x"] - u["x"], v["y"] - u["y"]) for v in ru
                    if v["card_id"] == u["card_id"] and v["team"] == u["team"]]
            assert same, u["card"]
            d.append(min(same))
    # the mirror steps one tick to commit its injections, so a walking unit is ~0.1-0.2 tiles on (3 matches: median
    # 0.04, 90th percentile 0.19 tiles over 547 one-body units, buildings included)
    assert d and np.median(d) < 0.25 and np.percentile(d, 90) < 0.5


def test_known_gap_elixir_spent_reads_zero_through_the_mirror(moments):
    # sim demos and self-play see the real totals; live and the pro shards see 0 (docs: decisions.md 2026-11-18)
    _, nat, _, reb = moments[-1]
    assert nat[SCAL0 + 1] > 0 and nat[SCAL0 + 2] > 0
    assert reb[SCAL0 + 1] == 0 and reb[SCAL0 + 2] == 0
