# Step A — pro move predictor (v1, 2026-10-21)

Goal (decisions.md, 2026-09-24): from strong players' games, predict what a player plays next — which card, when and
where. Use: the starting policy for B (behaviour cloning before any self-play), and a model of the opponent.

## v1: which card
- **Data:** every keyed replay (Battle Log + TV Royale, 471 games). Replays show both hands, so both players are exact:
  the bottom player from `<id>` (layout replay_bottom), the top player from `<id>_top`. One decision per play with a
  clean hand read just before it (`bot77.moves.decision_rows`), stored as 4 candidate rows in the LanceDB table
  `move_samples` (`scripts/build_moves.py`): 24,142 decisions (train 19,506, held-back 2,413, test 1,662, held-out 561).
- **Splits:** the same games as C. Test = config/replay_test_games.json; held-out (bot77.holdout) never used; held-back
  tunes. `league` (TV Royale tab, 0 = Ultimate Champion, -1 = Battle Log) lets training keep to top ladder.
- **Features:** the candidate card (id, cost, type, hand slot, affordable, time since last played, cheapest / dearest in
  hand, times played so far), the player's elixir, clock, multiplier, next card, own and opponent recent plays (cards
  and gaps), opponent cards seen; response rates (how often a card is chosen right after the opponent's / own last
  card; cross-fitted by game on train).
- **Model:** gradient boosting on candidate rows (`scripts/train_moves.py` -> data/models/moves_v1); highest score wins.

## Results
| | held-back top-1 / top-2 | test top-1 / top-2 |
|---|---|---|
| random | 24.0% / 52.0% | 22.7% / 50.3% |
| cheapest affordable | 29.1% / 54.0% | 31.2% / 59.1% |
| card played most so far | 35.8% / 63.7% | 32.7% / 61.6% |
| **A v1** | **46.6% / 76.0%** | **44.0% / 72.9%** |
| A v1, top ladder only | 41.2% / 65.8% (199) | 42.0% / 73.1% (610) |

**C's effect:** the bottom player scored with the opponent's history from C's detections (what B would see) instead of
the replay's top hand: held-back 43.6% -> 43.9%, test 43.4% -> 41.1% top-1 (top-2 73.8% -> 73.7%). A v1 leans mostly on
the player's own state, so C's misses and fakes cost little — and the opponent's moves are underused.

## Next
1. **When:** a timing model (play in the next second or not) sampled over the whole match, not only at plays.
2. **Where:** placement from C's deploy spots (x, y tiles) — lane / zone first.
3. **Opponent context:** what is on the board (units alive by lane, tower HP) from C, and the opponent's elixir estimate
   (bot77.elixir), so A answers the opponent rather than just cycling its own deck.
