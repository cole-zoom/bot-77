"""The models registry (bot77.models) on a throwaway LanceDB: adopting supersedes, retraining keeps adoption, missing
files fall back to the default."""
import lancedb
import pytest

import bot77.models as reg


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    db = lancedb.connect(tmp_path / "db")
    monkeypatch.setattr(reg, "_db", lambda: db)
    return tmp_path


def test_adopt_supersedes_and_retrain_keeps(tmpdb):
    a, b = tmpdb / "a.pkl", tmpdb / "b.pkl"
    a.write_text("a"); b.write_text("b")
    reg.register("card:a", "card", "a", str(a), "card", adopt=True)
    reg.register("card:b", "card", "b", str(b), "card")
    assert reg.adopted_path("card", "DEFAULT") == str(a)
    reg.adopt_model("card:b")
    rows = {r["model_id"]: r for r in reg.list_models()}
    assert rows["card:b"]["adopted"] and not rows["card:a"]["adopted"] and rows["card:a"]["status"] == "superseded"
    reg.register("card:b", "card", "b", str(b), "card", metrics={"x": 1})  # retrained in place
    assert reg.adopted("card")["model_id"] == "card:b"


def test_fallbacks(tmpdb):
    assert reg.adopted_path("card", "DEFAULT") == "DEFAULT"  # no registry yet
    reg.register("card:gone", "card", "gone", str(tmpdb / "missing.pkl"), "card", adopt=True)
    assert reg.adopted_path("card", "DEFAULT") == "DEFAULT"  # adopted file missing
