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
        self.frames = []

    def run(self, n):
        for _ in range(n):
            self.elixir = min(self.elixir + REGEN, 10)
            reads = {s: (card(c) if c else EMPTY) for s, c in self.hand.items()}
            reads["next"] = card(self.next)
            self.frames.append(FrameState(len(self.frames) / FPS, reads, ElixirRead(round(self.elixir, 2), int(self.elixir), True, 1.0)))
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
        ("play", "Mighty Miner", "high"), ("ability", "Mighty Miner", "medium"), ("ability", "Mighty Miner", "low")]


def test_two_cards_released_together_are_split_by_cost():
    tl = Timeline([TESLA, LOG, SKEL, MM], ESPIRIT, 9.0, [HOG, LOG]).run(5)
    tl.drag("slot3").drag("slot4").run(5).spend(5).run(4).refill("slot3").run(3).refill("slot4").run(10)
    plays = detect(tl)
    assert sorted(p.name for p in plays) == ["Mighty Miner", "Skeletons"]


def test_card_that_slides_in_and_is_dragged_straight_out_still_counts():
    tl = Timeline([TESLA, LOG, SKEL, HOG], ESPIRIT, 8.0, [MM, LOG]).run(5)
    tl.drag("slot3").run(3).spend(1).run(4).refill("slot3").run(1).drag("slot3").run(1).spend(1).run(4).refill("slot3").run(10)
    assert [p.name for p in detect(tl)] == ["Skeletons", "Electro Spirit"]
