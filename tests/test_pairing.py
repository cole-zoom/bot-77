from bot77.pairing import replay_can_be_of


def test_a_replay_is_recorded_on_or_soon_after_its_game():
    assert replay_can_be_of("bot_2026-11-15_071317", "bl_2026-11-15_r61_m01")
    assert replay_can_be_of("bot_2026-11-14_233000", "bl_2026-11-15_r02_m01")
    assert not replay_can_be_of("bot_2026-10-27_205025", "bl_2026-11-15_r08_m01")  # paired by chance before
    assert not replay_can_be_of("bot_2026-11-15_071317", "bl_2026-11-03_r01_m01")  # replay before the game
    assert replay_can_be_of("live_x", "bl_2026-11-15_r08_m01")  # no date: not ruled out
