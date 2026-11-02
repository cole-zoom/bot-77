# board_states: every moment of every game, queryable

Status: design, 2026-11-02. Not built yet. Cole: "Should we build a lance table with columns for whats on the board
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

## Schema (LanceDB table `board_states`, data/lancedb)

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
| best_arrows_value, best_fireball_value | float32 | worth the best-placed spell would hit now (zone up to y 21) |
| best_arrows_xy, best_fireball_xy | list<float32> | its centre |
| tower_hp | struct{own_l, own_r, own_king, opp_l, opp_r, opp_king} | fractions, 0 = down |
| towers_ours, towers_theirs | int8 | princess towers standing |
| king_awake | bool | |
| tempo | string | chill / press / even / desperate (bot77.rl.situations rules) |
| our_play_now, their_play_now | struct{card, x, y} | a play seen at this row (null if none) |
| our_play_next, their_play_next | struct{card, x, y, in_s} | the next play and how soon |
| our_spell_hit | float32 | for a spell of ours cast at this row: worth it hit (bot_actions spot) |
| our_tower_loss_6s, our_tower_loss_12s | float32 | HP fraction our princess towers lose over the next 6 / 12 s |
| their_tower_loss_6s, their_tower_loss_12s | float32 | the same for theirs |
| bot_hand_read | list<string> | what the live reader saw (bot games) |
| blind | bool | a card locked out by the reader bug (decisions.md 2026-11-02) |
| p_wait | float32 | the model's p(wait) at the nearest decision |
| p_cards | list<float32> | its p per slot |
| value | float32 | its value estimate |
| vector | fixed_size_list<float32>[160] | 6x8 grid of worth per team (96), both hands (8 card slots of our deck + their seen cards), elixir, towers; later the network's own features |
| modes | list<string> | detectors that fire here |
| cole_label | struct{mode, verdict, should, note} | from his reviews |

## Querying (checked 2026-11-02, lancedb 0.39, duckdb 2.0)

- LanceDB `where` handles list membership (`array_has(our_hand, 'Arrows')`), struct fields (`tower_hp.opp_r`,
  `our_play_next.in_s`), list length, and vector search with a prefilter.
- LanceDB `where` does not take lambdas over the units list (`array_filter(units, u -> ...)` fails). Unit-level
  questions use the per-lane columns, or DuckDB over the table's Arrow data: `unnest(units)` and
  `list_filter(units, lambda u: u.team = 'opp')` both work. DuckDB is a project dependency (added 2026-11-02).

## The failure-mode queries

The modes are Cole's (scripts/failure_modes.py). The thresholds are the miner's current ones (draft). Each query gets
tuned against Cole's 31 notes once the table exists, and the final version replaces the draft here, with its hit
rate on his notes. A query finds rows; consecutive hits within a few seconds are one moment (the first row).

**missed_spell** (draft): a spell's worth on the table, in hand, affordable, and not cast within 3 s.

```sql
best_arrows_value >= 3.5 AND array_has(our_hand, 'Arrows') AND our_elixir >= 3
  AND NOT (our_play_next.card = 'Arrows' AND our_play_next.in_s <= 3)
-- Fireball: best_fireball_value >= 5 AND array_has(our_hand, 'Fireball') AND our_elixir >= 4 ...
```

**pointless_push** (draft): Giant / Battle Ram at the bridge (or a defender sent over) with no elixir lead or into a
wall, not when behind on towers, defending, or counterpushing.

```sql
our_play_now.card IN ('Giant', 'Battle Ram') AND our_play_now.y >= 12.5
  AND towers_ours >= towers_theirs
  AND (CASE WHEN our_play_now.x < 9 THEN threat_left ELSE threat_right END) < 2
  AND (CASE WHEN our_play_now.x < 9 THEN our_left_value ELSE our_right_value END) < 3
  AND (elixir_lead < -0.5 OR (CASE WHEN our_play_now.x < 9 THEN their_wall_left ELSE their_wall_right END) >= 6)
```

**spent_counter** (draft): our last counter used on a non-win-condition while their win condition is in hand.

```sql
our_play_now.card IN ('Mini P.E.K.K.A', 'Goblin Hut', 'Mega Minion')
  AND array_length(their_wincons_in_hand) > 0 AND array_length(our_counters_in_hand) = 0  -- after the play
```

**wasted_spell** (draft): a spell of ours that hit less than its cost.

```sql
our_play_now.card IN ('Arrows', 'Fireball')
  AND our_spell_hit < (CASE our_play_now.card WHEN 'Arrows' THEN 2.5 ELSE 3.5 END)
```

**too_early** (draft): a defender on our half while their push hasn't crossed.

```sql
our_play_now.card IN ('Mini P.E.K.K.A', 'Mega Minion', 'Bomber') AND our_play_now.y < 15.5
  AND threat_left + threat_right < 1
  AND (CASE WHEN our_play_now.x < 9 THEN their_left_value ELSE their_right_value END) >= 3  -- coming (DuckDB: y 16.5-23.5)
```

**bad_placement** (draft, DuckDB): a defender in the other lane from their push, or Mini P.E.K.K.A / Bomber dropped
within 1.5 tiles of a ground attacker.

```sql
SELECT * FROM board_states WHERE our_play_now.card IN ('Mini P.E.K.K.A', 'Bomber')
  AND len(list_filter(units, lambda u: u.team = 'opp' AND NOT u.air AND u.worth >= 1.5
      AND (u.x - our_play_now.x)^2 + (u.y - our_play_now.y)^2 <= 2.25)) > 0
-- wrong lane: our_play_now.x < 9 AND threat_left = 0 AND threat_right >= 3 (and the mirror)
```

**no_defence** (draft): their troops on a standing tower of ours, elixir and a troop in hand, no play for 4 s, the
tower losing HP.

```sql
(threat_left >= 3 AND tower_hp.own_l > 0 OR threat_right >= 3 AND tower_hp.own_r > 0)
  AND our_elixir >= 3 AND (our_play_next.in_s > 4 OR our_play_next IS NULL) AND our_tower_loss_6s >= 0.05
```

**missed_finish** (draft): their princess tower within a spell's tower damage, the spell in hand, not cast.

```sql
((tower_hp.opp_l > 0 AND tower_hp.opp_l <= 0.07) OR (tower_hp.opp_r > 0 AND tower_hp.opp_r <= 0.07))
  AND array_has(our_hand, 'Fireball') AND our_elixir >= 4
  AND NOT (our_play_next.card IN ('Fireball', 'Arrows') AND our_play_next.in_s <= 5)
```

**more like this** (no rule): the nearest rows to each of Cole's noted moments, same source, not blind.

```python
tbl.search(row["vector"]).where("source = 'replay_bot' AND NOT blind AND changed", prefilter=True).limit(50)
```

Every review query adds `NOT blind` (moments caused by the reader bug go to no one) and, for the bot's games,
`source = 'replay_bot'`.

## Building it

scripts/build_board_states.py (to write): per game, the replay's hud_states (both sides), detections, plays and
tower_hp (the ingest chain), plus the live log (decisions.json, bot_actions) for bot games; derived columns computed in
one pass; one LanceDB write per game; a scalar index on `game` and `source`. data/ stays gitignored.
