# Ground truth

Play-by-play labels from Cole's reviews, independent of event ids, so any detector run can be
scored against them with `bot77.review.score_against_truth`.

| File | Matches | True events | Notes |
|---|---|---|---|
| `truth_uz4VVzGlOjE_m01-02.json` | Alina, matches 1–2 | 118 | Round 1; detector was tuned on these. Two plays added from review notes (`from_note`). |
| `truth_uz4VVzGlOjE_m03-04.json` | Alina, matches 3–4 | 112 | Round 2; held out when first scored. |

`reviews/` holds the raw exports from the review page. The round 2 export also carries stale
verdicts for matches 1–2 (from before a re-detect); only matches 3–4 were used from it.
