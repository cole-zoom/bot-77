"""The cycle tracker (bot77.live.cycle): the reader is the eyes, the tracker the memory (2026-11-02)."""
from bot77.live.cycle import CycleTracker

G, R, P, M, B, H, F, A = 1, 2, 3, 4, 5, 6, 7, 8  # Giant, Ram, Mini P.E.K.K.A, Mega Minion, Bomber, Hut, Fireball, Arrows
DECK = [G, R, P, M, B, H, F, A]
COST = {G: 5, R: 4, P: 4, M: 3, B: 2, H: 5, F: 4, A: 3}


def feed(tr, t0, t1, slots, nxt, elixir, dt=0.1):
    t = t0
    while t < t1 - 1e-9:
        tr.update(t, slots, nxt, elixir(t) if callable(elixir) else elixir)
        t = round(t + dt, 3)
    return t


def started(hand=(G, R, P, M), nxt=B):
    tr = CycleTracker(DECK, COST)
    feed(tr, 0.0, 1.0, list(hand), nxt, 7.0)
    assert tr.started and tr.hand == list(hand) and tr.queue == [nxt, None, None, None]
    return tr


def play(tr, t, slot, new, nxt_after, el_before, cost, slide=1.2):
    """Our card in `slot` is played at t: elixir drops by its cost, the slot reads blank while `new` slides in."""
    hand = list(tr.hand)
    blank = hand.copy(); blank[slot] = None
    after = hand.copy(); after[slot] = new
    feed(tr, t, t + slide, blank, None, el_before - cost)
    return feed(tr, t + slide, t + slide + 1.0, after, nxt_after, el_before - cost + 0.5)


def test_starts_from_a_steady_opening_hand():
    tr = CycleTracker(DECK, COST)
    feed(tr, 0.0, 0.3, [G, R, P, M], B, 7.0)
    assert not tr.started  # not held long enough yet
    feed(tr, 0.3, 1.0, [G, R, None, M], B, 7.0)  # a slot unsure: no start
    assert not tr.started
    feed(tr, 1.0, 2.0, [G, R, P, M], B, 7.0)
    assert tr.started


def test_a_play_moves_next_into_the_slot_and_the_card_to_the_back():
    tr = started()
    play(tr, 2.0, 1, B, H, el_before=7.0, cost=COST[R])  # Battle Ram played, Bomber (Next) comes in
    assert tr.hand == [G, B, P, M]
    assert tr.queue == [H, None, None, R]
    assert [e.kind for e in tr.events] == ["start", "play", "next_learned"]


def test_a_blank_slot_still_knows_its_card():
    tr = started()
    feed(tr, 2.0, 30.0, [G, R, None, M], B, 7.0)  # the reader can't make out slot 3 for 28 s (the lock-out bug)
    assert tr.hand == [G, R, P, M] and tr.mismatches() == 0


def test_a_one_frame_misread_is_ignored():
    tr = started()
    tr.update(2.0, [G, R, F, M], B, 7.0)  # a Fireball "appears" for one read
    feed(tr, 2.1, 4.0, [G, R, P, M], B, 7.0)
    assert tr.hand == [G, R, P, M] and tr.mismatches() == 0 and [e.kind for e in tr.events] == ["start"]


def test_a_misread_of_next_without_an_elixir_drop_is_not_a_play():
    tr = started()
    feed(tr, 2.0, 2.3, [G, R, B, M], B, 7.0)  # slot 3 reads Bomber (the Next) for 0.3 s, no elixir spent
    feed(tr, 2.3, 4.0, [G, R, P, M], B, 7.2)
    assert tr.hand == [G, R, P, M] and tr.queue[0] == B


def test_a_failed_tap_changes_nothing():
    tr = started()
    feed(tr, 2.0, 5.0, [G, R, P, M], B, 7.0)  # the bot tapped, nothing happened on screen
    assert tr.hand == [G, R, P, M] and tr.queue == [B, None, None, None]


def test_the_screen_wins_when_it_keeps_showing_something_else():
    tr = started()
    feed(tr, 2.0, 3.0, [G, R, F, M], B, 7.0)  # slot 3 shows Fireball for a whole second: the tracker was wrong
    assert tr.hand == [G, R, F, M] and tr.mismatches() == 1


def test_the_whole_cycle_is_known_after_the_first_plays():
    tr = started()  # hand G R P M, next B; queue order of H F A unknown
    t = play(tr, 2.0, 0, B, H, 9.0, COST[G])   # Giant out, Bomber in, Next Hut
    t = play(tr, t, 1, H, F, 9.0, COST[R])     # Ram out, Hut in, Next Fireball
    assert tr.queue == [F, A, G, R]  # Arrows deduced: the one card left for the one unknown place
    t = play(tr, t, 2, F, A, 9.0, COST[P])
    assert tr.hand == [B, H, F, M] and tr.queue == [A, G, R, P]
    # and it predicts what comes in next without being told
    t = play(tr, t, 3, A, G, 9.0, COST[M])
    assert tr.hand == [B, H, F, A] and tr.queue == [G, R, P, M]


def test_two_plays_in_quick_succession():
    tr = started()
    feed(tr, 2.0, 3.2, [None, None, P, M], None, 1.0)  # Giant and Ram both played, both slots blank
    feed(tr, 3.2, 4.2, [B, None, P, M], H, 1.3)
    feed(tr, 4.2, 5.2, [B, H, P, M], None, 1.6)
    assert tr.hand == [B, H, P, M] and tr.queue[2:] == [G, R]


def test_a_card_played_the_moment_it_arrives_is_a_chain_not_a_mismatch():
    tr = started()
    t = play(tr, 2.0, 0, B, H, 9.0, COST[G])  # Giant out, Bomber in, Next Hut
    t = play(tr, t, 1, H, F, 9.0, COST[R])    # queue now F A G R
    # slot 1 (Hut) is played; Fireball slides in and is played inside 0.2 s; Arrows settles there
    feed(tr, t, t + 1.2, [B, None, P, M], None, 9.0 - COST[H])
    feed(tr, t + 1.2, t + 1.3, [B, F, P, M], A, 9.0 - COST[H])  # one read of the Fireball
    feed(tr, t + 1.3, t + 2.5, [B, None, P, M], None, 9.0 - COST[H] - COST[F])
    feed(tr, t + 2.5, t + 3.5, [B, A, P, M], G, 9.0 - COST[H] - COST[F] + 0.4)
    assert tr.hand == [B, A, P, M] and tr.queue == [G, R, H, F] and tr.mismatches() == 0
