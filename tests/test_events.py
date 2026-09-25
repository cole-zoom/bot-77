"""Synthetic hand/elixir timelines for the play detector, modelled on real sequences."""

from bot77.events import FrameState, detect_plays
from bot77.readers.elixir import ElixirRead
from bot77.readers.hand import SLOTS, SlotRead

HOG, LOG, SKEL, ESPIRIT, MM, TESLA = 1, 2, 3, 4, 5, 6
COSTS = {HOG: 4, LOG: 2, SKEL: 1, ESPIRIT: 1, MM: 4, TESLA: 4}
NAMES = {HOG: "Hog Rider", LOG: "The Log", SKEL: "Skeletons", ESPIRIT: "Electro Spirit", MM: "Mighty Miner", TESLA: "Tesla"}
FPS = 10
REGEN = 1 / 2.8 / FPS  # elixir per frame at x1


def card(cid):
    return SlotRead("card", cid, NAMES[cid], "normal", 0.8, 0.4)


EMPTY = SlotRead("empty")


class Timeline:
    """Build frames step by step: hand edits and elixir spends at chosen frames."""

    def __init__(self, hand, nxt, elixir, queue):
        self.hand = dict(zip(SLOTS, hand))
        self.next = nxt
        self.queue = list(queue)  # cards that come up as "next" after each play
        self.elixir = elixir
        self.button = "absent"  # right ability button
        self.frames = []

    def run(self, n):
        for _ in range(n):
            self.elixir = min(self.elixir + REGEN, 10)
            reads = {s: (card(c) if c else EMPTY) for s, c in self.hand.items()}
            reads["next"] = card(self.next)
            self.frames.append(FrameState(len(self.frames) / FPS, reads, ElixirRead(round(self.elixir, 2), int(self.elixir), True, 1.0),
                                          abilities=("absent", self.button)))
        return self

    def drag(self, slot):
        self.hand[slot] = None
        return self

    def cancel(self, slot, cid):
        self.hand[slot] = cid
        return self

    def spend(self, amount):
        self.elixir -= amount
        return self

    def refill(self, slot):
        self.hand[slot], self.next = self.next, self.queue.pop(0)
        return self


def detect(tl, **kw):
    return detect_plays(tl.frames, COSTS, NAMES, **kw)


def test_drag_release_refill_is_one_high_confidence_play():
    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, 6.0, [ESPIRIT]).run(10).drag("slot3").run(8).spend(4).run(4).refill("slot3").run(10)
    (p,) = detect(tl)
    assert (p.name, p.slot, p.confidence) == ("Hog Rider", "slot3", "high")
    assert abs(p.measured_cost - 4) < 0.3


def test_cancelled_drag_is_not_a_play():
    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, 6.0, [ESPIRIT]).run(10).drag("slot1").run(10).cancel("slot1", TESLA).run(10)
    assert detect(tl) == []


def test_ability_only_after_an_unused_deploy():
    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, 9.0, [ESPIRIT, HOG]).run(5)
    tl.drag("slot4").run(5).spend(4).run(4).refill("slot4").run(20)  # Mighty Miner deployed
    tl.spend(1).run(20)  # ability
    tl.spend(1).run(20)  # a second 1-elixir drop with no card: can't be the ability again
    plays = detect(tl, ability_costs={MM: 1}, ability_uses={MM: 1})
    assert [(p.kind, p.name, p.confidence) for p in plays] == [
        ("play", "Mighty Miner", "high"), ("ability", "Mighty Miner", "medium")]


def test_sliver_snap_is_not_an_ability():
    """Right after a play the fractional sliver vanishes (3.64 -> 3.0): no whole elixir lost."""
    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, 9.0, [ESPIRIT]).run(5)
    tl.drag("slot4").run(5).spend(4).run(4).refill("slot4").run(10)  # Mighty Miner deployed
    tl.elixir = int(tl.elixir) + 0.02  # sliver snaps away
    tl.run(20)
    assert [p.kind for p in detect(tl, ability_costs={MM: 1}, ability_uses={MM: 1})] == ["play"]


def test_hand_trumps_a_wrong_measured_cost():
    """The bar animation can throw the measured cost off; a lone vacated card that's replaced by
    the previous next card is still the play."""
    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, 8.0, [ESPIRIT]).run(10).drag("slot3").run(5).spend(3.4).run(4).refill("slot3").run(10)
    (p,) = detect(tl)
    assert p.name == "Hog Rider" and p.confidence == "medium" and "cost mismatch" in p.notes[0]


def test_covered_elixir_bar_is_ignored():
    """The emote panel covers the hand and elixir bar; the bar reads 0 under it."""
    from bot77.readers.elixir import ElixirRead as ER

    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, 8.0, [ESPIRIT]).run(10)
    covered = {s: SlotRead("unknown") for s in SLOTS} | {"next": SlotRead("unknown")}
    for k in range(10):
        tl.frames.append(FrameState(len(tl.frames) / FPS, covered, ER(0.0, 0, True, 1.0)))
    tl.run(10)
    assert detect(tl) == []


def test_two_cards_released_together_are_split_by_cost():
    tl = Timeline([TESLA, LOG, SKEL, MM], ESPIRIT, 9.0, [HOG, LOG]).run(5)
    tl.drag("slot3").drag("slot4").run(5).spend(5).run(4).refill("slot3").run(3).refill("slot4").run(10)
    plays = detect(tl)
    assert sorted(p.name for p in plays) == ["Mighty Miner", "Skeletons"]


def test_card_that_slides_in_and_is_dragged_straight_out_still_counts():
    tl = Timeline([TESLA, LOG, SKEL, HOG], ESPIRIT, 8.0, [MM, LOG]).run(5)
    tl.drag("slot3").run(3).spend(1).run(4).refill("slot3").run(1).drag("slot3").run(1).spend(1).run(4).refill("slot3").run(10)
    assert [p.name for p in detect(tl)] == ["Skeletons", "Electro Spirit"]


def test_card_dragged_while_still_sliding_in_counts():
    """The next card slides into an empty slot and is dragged out mid-slide: it never reads as a
    card in the slot, but "next" advanced and elixir dropped by its cost."""
    tl = Timeline([TESLA, LOG, SKEL, HOG], ESPIRIT, 8.0, [MM]).run(5)
    tl.drag("slot3").run(3).spend(1).run(4)  # Skeletons played
    tl.next = MM  # E-Spirit leaves "next" to slide into slot 3 ...
    tl.run(2).spend(1).run(10)  # ... and is released before it ever reads as a card
    assert [p.name for p in detect(tl)] == ["Skeletons", "Electro Spirit"]


def _mm_deployed(elixir=9.0):
    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, elixir, [ESPIRIT, LOG]).run(5)
    tl.drag("slot4").run(5).spend(4).run(4).refill("slot4")
    tl.button = "ready"
    return tl.run(20)


def test_button_ability_then_hog_in_one_drop():
    """Ability + Hog released together: one 5-elixir drop, the button goes ready -> dark."""
    tl = _mm_deployed(9.5).drag("slot3").run(5)
    tl.button = "dark"
    tl.spend(5).run(4).refill("slot3").run(10)
    plays = detect(tl, ability_costs={MM: 1})
    assert [(p.kind, p.name) for p in plays] == [("play", "Mighty Miner"), ("ability", "Mighty Miner"), ("play", "Hog Rider")] \
        or [(p.kind, p.name) for p in plays] == [("play", "Mighty Miner"), ("play", "Hog Rider"), ("ability", "Mighty Miner")]
    hog = [p for p in plays if p.name == "Hog Rider"][0]
    assert "cost mismatch" not in " ".join(hog.notes)


def test_button_ability_even_when_whole_elixir_unchanged():
    """At triple elixir an ability can leave the whole count unchanged (5.83 -> 5.14)."""
    tl = _mm_deployed()
    tl.elixir = 5.83
    tl.run(2)
    tl.button = "dark"
    tl.elixir = 5.14
    tl.run(10)
    assert [p.kind for p in detect(tl, ability_costs={MM: 1})] == ["play", "ability"]


def test_button_flicker_is_not_an_ability():
    tl = _mm_deployed()
    tl.button = "dark"
    tl.run(1)
    tl.button = "ready"
    tl.run(10)
    assert [p.kind for p in detect(tl, ability_costs={MM: 1})] == ["play"]


def test_hovered_card_is_not_played_by_a_sliver_snap():
    """Card dragged over the board (slot empty) while the sliver snaps 4.66 -> 4.0."""
    tl = Timeline([TESLA, LOG, HOG, MM], SKEL, 4.66, [ESPIRIT]).run(5).drag("slot1").run(3)
    tl.elixir = 4.0
    tl.run(10).cancel("slot1", TESLA).run(10)
    assert detect(tl) == []


def test_button_dark_while_unaffordable_is_not_an_ability():
    """Playing E-Spirit at 1.1 elixir leaves under 1: the ready button goes dark, no ability."""
    tl = Timeline([TESLA, ESPIRIT, HOG, MM], SKEL, 5.0, [LOG, HOG]).run(5)
    tl.drag("slot4").run(3).spend(4).run(3).refill("slot4")  # Mighty Miner, leaving ~1.1 elixir
    tl.button = "ready"
    tl.run(3).drag("slot2").run(1)
    tl.spend(1).run(2)
    tl.button = "dark"
    tl.run(3).refill("slot2").run(10)
    assert [(p.kind, p.name) for p in detect(tl, ability_costs={MM: 1})] == [("play", "Mighty Miner"), ("play", "Electro Spirit")]


def test_button_lags_the_elixir_drop():
    """The button can go dark ~1 s after the ability's elixir drop."""
    tl = _mm_deployed(9.5).drag("slot3").run(5).spend(5).run(4).refill("slot3").run(6)
    tl.button = "dark"
    tl.run(10)
    assert sorted((p.kind, p.name) for p in detect(tl, ability_costs={MM: 1})) == [
        ("ability", "Mighty Miner"), ("play", "Hog Rider"), ("play", "Mighty Miner")]
