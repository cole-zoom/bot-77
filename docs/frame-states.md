# frame_states: every moment of every game, queryable

Status: built 2026-11-02 for the 70 run40v4 games with a replay (122,369 rows, scripts/build_frame_states.py). Cole: "Should we build a lance table with columns for whats on the board
and whats in the hand to make mining easier?" ... "why not every frame? ... record the final query in the design docs".

## Why

The miner (scripts/mine_moments.py) rebuilds each moment from four tables (hud_states, detections, plays, tower_hp),
plus the live log, and every threshold change means rerunning it. Its detectors find 9 of Cole's 31 noted mistakes
(decisions.md 2026-11-02). With one row per frame holding the hands, elixir, board, towers and what happened next,
each failure mode becomes a query. It can be checked against Cole's 31 in seconds and tuned until it finds them.
Each mode's final query is recorded below.

## Rows: every frame

One row per hand read: the replay HUD is read every 0.1 s (hud_states, median step 0.100 s), so ~1,700 rows a
3-minute game, ~136k for the 80 run40v4 games. Units come from the detector every 0.25 s (detections, median 0.250 s):
a row carries the latest detection at or before it, and `units_age` says how old that is. Nothing is dropped. Rows that
repeat the one before (same hands, elixir within 0.1, same units frame) have `changed = false`, so a query can skip
them; dedupe is a filter, not a loss.

Sources: the bot's games (replay_bot), Battle Log and TV pro replays (replay_bl, replay_tv), Cole's games (cole). The
same query can show how pros play the state the bot got wrong.

## Schema (LanceDB table `frame_states`, data/lancedb; `board_states` is already A's per-decision table)

Engine board coordinates (x 0-18, y 0-34, our side y < 16.5, river 16.5; bot77.sim.geometry). Worth = a card's elixir
cost shared over its bodies (Skeletons 0.33 each).

| column | type | notes |
|---|---|---|
| state_id | string | `{game}@{t:.1f}` |
| game | string | live game id or replay id |
| replay_id | string | the replay it was read from |
| source | string | replay_bot / replay_bl / replay_tv / cole |
| model | string | the checkpoint playing (bot games) |
| result | string | win / loss |
| t | float32 | game seconds |
| clock | string | `0:44`, `1:12 OT` |
| phase | string | regulation / overtime |
| multiplier | int8 | elixir speed 1 / 2 / 3 |
| changed | bool | differs from the row before |
| our_hand, their_hand | list<string> | 4 card names, `?` unread |
| our_next, their_next | string | |
| our_elixir, their_elixir | float32 | |
| elixir_lead | float32 | ours minus theirs |
| their_cycle | list<struct{card, seconds_ago}> | their last 8 plays |
| our_counters_in_hand | list<string> | our answers to their win conditions, held now |
| their_wincons_in_hand | list<string> | |
| units | list<struct{card, team, x, y, lane, half, worth, air}> | team own / opp; lane left / right; half ours / theirs |
| units_age | float32 | seconds since that detection |
| our_left_value, our_right_value | float32 | worth of our troops per lane |
| their_left_value, their_right_value | float32 | worth of theirs per lane |
| threat_left, threat_right | float32 | worth of theirs on our half, per lane |
| their_wall_left, their_wall_right | float32 | worth of theirs on their half, per lane (what a push runs into) |
| best_arrows_value, best_fireball_value | float32 | elixir the best-placed spell would win now: each unit's worth x the share of its HP the spell takes (Arrows 369, Fireball 689); centres up to y 21 |
| best_arrows_xy, best_fireball_xy | list<float32> | its centre |
| tower_hp | struct{own_l, own_r, own_king, opp_l, opp_r, opp_king} | fractions, 0 = down |
| towers_ours, towers_theirs | int8 | princess towers standing |
| king_awake | bool | |
| tempo | string | chill / press / even / desperate (bot77.rl.situations rules) |
| our_play_now, their_play_now | struct{card, x, y} | a play seen at this row (null if none) |
| our_play_next, their_play_next | struct{card, x, y, in_s} | the next play and how soon |
| our_spell_hit | float32 | for a spell of ours cast at this row: elixir it won (worth x the share of HP it takes) |
| play_nearest_foe | float32 | our play's distance to their nearest ground unit worth 1.5+ |
| our_tower_loss_6s, our_tower_loss_12s | float32 | HP fraction our princess towers lose over the next 6 / 12 s |
| their_tower_loss_6s, their_tower_loss_12s | float32 | the same for theirs |
| bot_hand_read | list<string> | what the live reader saw (bot games) |
| blind | bool | a card locked out by the reader bug (decisions.md 2026-11-02) |
| p_wait | float32 | the model's p(wait) at the nearest decision |
| p_cards | list<float32> | its p per slot |
| value | float32 | its value estimate |
| vector | fixed_size_list<float32>[176] | 6x8 grid of worth per team (96, half weight), our hand and next (16), their hand hashed (24), elixir, towers, time, lane values; the play at this row at double weight (card, spot, its lane, elixir lead). Later the network's own features |
| modes | list<string> | detectors that fire here |
| cole_label | struct{mode, verdict, should, note} | from his reviews |

## Querying: all in LanceDB (2026-11-02)

- Filters (`where`, DataFusion SQL): list membership `array_has(our_hand, 'Arrows')`, `array_length(...)`, struct
  fields `tower_hp.opp_r`, `our_play_now.card`, `coalesce`, `IN`, `NOT`. No `CASE` (written as ORs, see
  `lane()` in scripts/query_modes.py) and no lambdas over the units list: what a query needs from inside the units
  is a column computed at build time instead (`play_nearest_foe`, the per-lane values).
- Hybrid: vector search with a prefilter, `tbl.search(vec).where(filter, prefilter=True)`. The filter says what kind
  of row a mode can be about (our play is a Giant / Battle Ram / defender; their tower is low; ...), the vector finds
  the ones nearest Cole's labelled rows. DuckDB was tried for the unit lambdas and dropped: one engine, and vector
  search only lives in LanceDB.

## The failure-mode queries

The exact filters (rule) and prefilters (hybrid) live in scripts/query_modes.py, QUERIES and SCOPE; it prints each
mode's score against Cole's notes. Change them there and update the table below with the new score.

**Scores, 2026-11-02** (27 of Cole's 31 notes are in games with a replay; 32 note x mode pairs; hybrid is
leave-one-out from his other notes of the mode, k = 60 nearest rows each; "random" picks the same number of moments
from the prefilter at random, 200 draws):

| mode | rule | hybrid | either | moments a game (rule / hybrid) | random at the hybrid's volume |
|---|---|---|---|---|---|
| missed spell | 4/8 | 1/8 | 5/8 | 2.4 / 0.3 | 0.0 |
| pointless push | 0/9 | 6/9 | 6/9 | 0.2 / 5.0 | 4.3 (at 3.6 a game) |
| spent the counter | 0/2 | 0/2 | 0/2 | 2.6 / 0.8 | |
| wasted spell | 0/3 | 2/3 | 2/3 | 0.0 / 1.2 | 1.5 |
| didn't wait | 1/2 | 0/2 | 1/2 | 0.8 / 0.8 | |
| placement | 0/1 | 0/1 | 0/1 | 0.5 / 0.0 | |
| didn't defend / kite | 1/5 | 0/5 | 1/5 | 0.9 / 0.1 | |
| didn't finish | 0/2 | 0/2 | 0/2 | 0.3 / 0.1 | |
| **all** | **6** | **9** | **15 of 32** | | |

Read: the missed-spell rule is real (4 of 8 at 2.4 a game, where random finds none). The hand-made vector is not yet:
on pointless pushes it barely beats random, on wasted spells it is at chance, with or without the play features (v2,
176 floats). With 27 positives and no negatives nothing can learn what separates a bad push from a fine one. The way
forward is Cole marking candidates yes / no: every no is the negative the search is missing, and a small classifier
on the vector and the columns reranks the next batch.

**more like this** (no rule): the nearest rows to one of Cole's noted moments, same source, not blind.

```python
tbl.search(row["vector"]).where("NOT blind AND source = 'replay_bot' AND (" + SCOPE[mode][0] + ")", prefilter=True).limit(60)
```

## Building it

scripts/build_frame_states.py: per game, the replay's hud_states (both sides), detections, plays and
tower_hp (the ingest chain), plus the live log (decisions.json, bot_actions) for bot games; derived columns computed in
one pass; one LanceDB write per game; a scalar index on `game` and `source`. data/ stays gitignored.
