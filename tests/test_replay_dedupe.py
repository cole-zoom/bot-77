from bot77.replay_dedupe import duplicates, recorded_order, same_game

GAME = [(5.0 + 7 * i, c, s) for i, (c, s) in enumerate(
    [("Giant", "bottom"), ("Knight", "top"), ("Arrows", "bottom"), ("Hog Rider", "top"),
     ("Bomber", "bottom"), ("Fireball", "top"), ("Mega Minion", "bottom"), ("Zap", "top")])]


def test_the_same_game_twice_is_one_game_despite_detector_jitter():
    again = [(t + 0.5, c, s) for t, c, s in GAME[:-1]]  # one play missed, clock 0.5 s off
    assert same_game(GAME, again)


def test_same_decks_different_game_is_not_a_duplicate():
    other = [(t + 3.0, c, s) for t, c, s in GAME]  # same cards, other times (a rematch)
    assert not same_game(GAME, other)


def test_the_later_recording_is_the_duplicate():
    plays = {"bl_2026-11-15_r11_m01": GAME, "bl_2026-11-14_r38_m01": GAME, "bl_2026-11-14_r09_m01": GAME[::2] * 2}
    assert duplicates(plays) == {"bl_2026-11-15_r11_m01": "bl_2026-11-14_r38_m01"}
    assert recorded_order("tv_2026-10-13_r153_m01") > recorded_order("tv_2026-10-13_r35_m01")
