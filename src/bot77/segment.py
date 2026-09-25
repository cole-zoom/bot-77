"""Split a video into matches using the match clock."""

from __future__ import annotations

from dataclasses import dataclass

from bot77.readers.clock import ClockRead

NEW_MATCH_DROP_S = 20  # elapsed time falling by more than this means a new match started
MAX_GAP_S = 8  # no readable clock for longer than this ends the current match
MIN_MATCH_READS = 10  # fewer clock reads than this is noise, not a match


@dataclass
class Match:
    t_start: float  # video time the match clock started (extrapolated to elapsed = 0)
    t_end: float  # video time of the last readable clock
    reads: list[tuple[float, ClockRead]]

    @property
    def last(self) -> ClockRead:
        return self.reads[-1][1]


def segment_matches(reads: list[tuple[float, ClockRead]]) -> list[Match]:
    """`reads` are (video_time, clock) pairs in time order; only reads with an elapsed time
    count. A match continues while elapsed time keeps pace with video time."""
    matches: list[Match] = []
    current: list[tuple[float, ClockRead]] = []

    def close():
        if len(current) >= MIN_MATCH_READS:
            starts = sorted(t - r.elapsed_s for t, r in current)
            matches.append(Match(starts[len(starts) // 2], current[-1][0], list(current)))
        current.clear()

    for t, r in reads:
        if r.elapsed_s is None:
            continue
        if current:
            pt, pr = current[-1]
            gap = t - pt
            drop = pr.elapsed_s - r.elapsed_s
            # Big drop or long gap -> new match. A single wild read (OCR glitch) would also look
            # like a drop, so require the next read to agree before splitting (handled by
            # MIN_MATCH_READS discarding one-read fragments).
            if gap > MAX_GAP_S or drop > NEW_MATCH_DROP_S:
                close()
            elif r.elapsed_s - pr.elapsed_s > gap + 3:
                continue  # impossible jump forward: skip the outlier read
        current.append((t, r))
    close()
    return matches
