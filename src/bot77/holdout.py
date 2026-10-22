"""Every game no training builder may use (2026-10-17): the benchmark, the replay test set, and Cole's live test
recording with the replays of those same games (config/live_test_games.json: the replay shows the identical match, so
training on it would leak the test). Tuning sets (held-back replays, dev) are not listed: rules tune on them, models
don't train on them where noted in each builder."""
import json
from pathlib import Path


def held_out() -> set[str]:
    out = set(json.loads(Path("config/benchmark_holdout.json").read_text())["matches"])
    out |= set(json.loads(Path("config/replay_test_games.json").read_text())["matches"])
    live = json.loads(Path("config/live_test_games.json").read_text())
    return out | set(live["matches"]) | set(live["replays"])
