"""Split a video into matches using the match clock."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from bot77.readers.clock import ClockRead

NEW_MATCH_DROP_S = 20  # elapsed time falling by more than this means a new match started
MAX_GAP_S = 8  # no readable clock for longer than this ends the current match
MIN_MATCH_READS = 10  # fewer clock reads than this is noise, not a match
JUMP_CONFIRM_READS = 2  # a forward jump (an edit cutting footage) must be continued by this many reads


@dataclass
class Match:
    t_start: float  # video time the match clock started (extrapolated from the first read)
    t_end: float  # video time of the last readable clock
    reads: list[tuple[float, ClockRead]]

    @property
    def last(self) -> ClockRead:
        return self.reads[-1][1]

    def elapsed_at(self, t: float) -> float:
        """Match time at video time t, from the nearest earlier clock read. Uses the clock rather
        than video time so jump cuts in edited videos don't shift it."""
        times = [rt for rt, _ in self.reads]
        i = max(bisect.bisect_right(times, t) - 1, 0)
        rt, r = self.reads[i]
        if i + 1 < len(self.reads):
            nt, nr = self.reads[i + 1]
            return min(r.elapsed_s + (t - rt), nr.elapsed_s)
        return r.elapsed_s + (t - rt)


def _continues(reads: list[tuple[float, ClockRead]], k: int, n: int) -> bool:
    """Do the next n readable clocks after index k keep pace with reads[k]?"""
    t0, r0 = reads[k]
    seen = 0
    for t, r in reads[k + 1:]:
        if r.elapsed_s is None:
            continue
        if not (0 <= r.elapsed_s - r0.elapsed_s <= (t - t0) + 3):
            return False
        seen += 1
        if seen >= n:
            return True
    return False


def segment_matches(reads: list[tuple[float, ClockRead]]) -> list[Match]:
    """`reads` are (video_time, clock) pairs in time order; only reads with an elapsed time
    count. A match continues while elapsed time keeps pace with video time; a forward jump is
    accepted when later reads continue from it (edited videos cut footage out)."""
    matches: list[Match] = []
    current: list[tuple[float, ClockRead]] = []

    def close():
        if len(current) >= MIN_MATCH_READS:
            t0, r0 = current[0]
            matches.append(Match(t0 - r0.elapsed_s, current[-1][0], list(current)))
        current.clear()

    for k, (t, r) in enumerate(reads):
        if r.elapsed_s is None:
            continue
        if current:
            pt, pr = current[-1]
            gap = t - pt
            drop = pr.elapsed_s - r.elapsed_s
            if gap > MAX_GAP_S or (drop > NEW_MATCH_DROP_S and _continues(reads, k, JUMP_CONFIRM_READS)):
                close()
            elif drop > 3:
                continue  # a lone backward misread
            elif r.elapsed_s - pr.elapsed_s > gap + 3 and not _continues(reads, k, JUMP_CONFIRM_READS):
                continue  # a lone forward misread; a continued one is a jump cut
        current.append((t, r))
    close()
    return matches
