# Ground truth

Play-by-play labels from Cole's reviews, independent of event ids, so any detector run can be
scored against them with `bot77.review.score_against_truth`, or all at once with
`uv run bot77 benchmark`.

| File | Matches | True events | Notes |
|---|---|---|---|
| `truth_uz4VVzGlOjE_m01-02.json` | Alina, matches 1–2 | 118 | Round 1; detector was tuned on these. Two plays added from review notes (`from_note`). |
| `truth_uz4VVzGlOjE_m03-04.json` | Alina, matches 3–4 | 112 | Round 2; held out when first scored. |
| `truth_UFQFnrtWUE4_m01-03.json` | Ian77, matches 1–3 | 188 | Second creator, split layout, ~3k opponents, jump cuts; fully held out: 188/188 on first scoring. |
| `truth_uz4VVzGlOjE_m05.json` | Alina, match 5 | 72 | Round 3; fully held out: 72/72 on first scoring. Tesla + Mighty Miner at 0:49 right but in swapped order with a wrong elixir split (noted). |

`reviews/` holds the raw exports from the review page. The round 2 export also carries stale
verdicts for matches 1–2 (from before a re-detect); only matches 3–4 were used from it.
