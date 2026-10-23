# Features

Every feature the models see, generated from `src/bot77/features/registry.py` by `scripts/features_doc.py` (tests/test_features.py keeps the registry in step with the code). Level: `decision` = one value per moment, `card` = per card in hand, `spot` = per candidate spot. All positions are in the deciding player's frame (`bot77.features.frames`).

88 features.

## hand

| Feature | Level | Meaning | Since |
|---|---|---|---|
| `cand_card` | card | the card (id; categorical) | A v1 |
| `cand_cost` | card | its elixir cost | A v1 |
| `cand_type` | card | troop 0 / building 1 / spell 2 | A v1 |
| `cand_slot` | card | hand slot 0-3 | A v1 |
| `cand_affordable` | card | elixir >= cost - 0.3 | A v1 |
| `cand_waited_s` | card | seconds since this card was last played (60+ if never) | A v1 |
| `cand_is_cheapest` | card | cheapest card in hand | A v1 |
| `cand_is_dearest` | card | most expensive card in hand | A v1 |
| `cand_last_own_gap` | card | times this player has played the card so far this game | A v1 |

## elixir / clock

| Feature | Level | Meaning | Since |
|---|---|---|---|
| `elixir` | decision | elixir now | A v1 |
| `t_match` | decision | seconds since the match started | A v1 |
| `multiplier` | decision | elixir multiplier (1 / 2 / 3) | A v1 |
| `next_card` | decision | the next card (id; categorical) | A v1 |

## history

| Feature | Level | Meaning | Since |
|---|---|---|---|
| `own_since_s` | decision | seconds since the player's last play | A v1 |
| `opp_since_s` | decision | seconds since the opponent's last play | A v1 |
| `opp_seen_n` | decision | different opponent cards seen so far | A v1 |
| `own_prev1` | decision | the player's 1st-last card (id) | A v1 |
| `own_prev2` | decision | the player's 2nd-last card (id) | A v1 |
| `own_prev3` | decision | the player's 3rd-last card (id) | A v1 |
| `own_prev4` | decision | the player's 4th-last card (id) | A v1 |
| `opp_prev1` | decision | the opponent's 1st-last card (id) | A v1 |
| `opp_prev2` | decision | the opponent's 2nd-last card (id) | A v1 |
| `opp_prev3` | decision | the opponent's 3rd-last card (id) | A v1 |
| `opp_prev4` | decision | the opponent's 4th-last card (id) | A v1 |
| `opp_prev1_dt` | decision | seconds since the opponent's 1st-last play | A v1 |
| `opp_prev2_dt` | decision | seconds since the opponent's 2nd-last play | A v1 |
| `opp_prev3_dt` | decision | seconds since the opponent's 3rd-last play | A v1 |
| `opp_prev4_dt` | decision | seconds since the opponent's 4th-last play | A v1 |

## rates

| Feature | Level | Meaning | Since |
|---|---|---|---|
| `resp_rate` | card | how often players pick this card right after the opponent's last card (smoothed) | A v1 |
| `own_rate_after` | card | how often players pick this card right after their own last card (smoothed) | A v1 |
| `elixir_after` | card | elixir left after playing it | A v1 |
| `resp_threat` | card | how often players answer the main threat on the board with this card (smoothed) | A v2 |

## board

| Feature | Level | Meaning | Since |
|---|---|---|---|
| `b_opp_val_our_half` | decision | elixir of the enemy cards on our half (each card once per lane) | A v2 |
| `b_opp_val_our_L` | decision | ... in our left lane | A v2 |
| `b_opp_val_our_R` | decision | ... in our right lane | A v2 |
| `b_opp_val_field` | decision | elixir of all enemy cards on the field | A v2 |
| `b_own_val_field` | decision | elixir of our cards on the field | A v2 |
| `b_own_val_attacking` | decision | elixir of our cards on their half | A v2 |
| `b_opp_nearest_y` | decision | how far down the nearest enemy unit is (tiles; 0 = none) | A v2 |
| `b_opp_air_our` | decision | enemy air units on our half | A v2 |
| `b_opp_tank_our` | decision | an enemy tank on our half | A v2 |
| `b_opp_swarm_our` | decision | enemy swarm units on our half | A v2 |
| `b_opp_wincon_field` | decision | an enemy win condition on the field | A v2 |
| `b_own_tank_field` | decision | our tank on the field | A v2 |
| `b_threat_card` | decision | the main threat: the dearest enemy card on our half (id; categorical; -1 none) | A v2 |
| `b_threat_lane` | decision | its lane: left 0 / right 1 (-1 none) | A v2 |
| `b_tower_own_L` | decision | our left princess tower standing | A v2 |
| `b_tower_own_R` | decision | our right princess tower standing | A v2 |
| `b_tower_opp_L` | decision | their left princess tower standing (our left) | A v2 |
| `b_tower_opp_R` | decision | their right princess tower standing | A v2 |

## card vs board

| Feature | Level | Meaning | Since |
|---|---|---|---|
| `c_can_hit_threat` | card | this card can hit the main threat (air vs ground-only; -1 no threat) | A v2 |
| `c_air_answer` | card | air is coming and this troop / building hits air | A v2 |
| `c_spell_value` | card | for spells: elixir of the cards in its best target group / its cost | A v2 |
| `c_cost_vs_threat` | card | this card's cost minus the threat's | A v2 |
| `c_wincon` | card | a win condition (the card you rely on for tower damage) | A v2 |
| `c_tank` | card | a tank | A v2 |
| `c_ground_only` | card | can't hit air | A v2 |
| `c_air_unit` | card | a flying unit | A v2 |
| `c_swarm` | card | many small units | A v2 |
| `c_ranged` | card | shoots from range | A v2 |
| `c_wincon_under_attack` | card | a win condition while there's a threat on our half | A v2 |

## spot

| Feature | Level | Meaning | Since |
|---|---|---|---|
| `s_kind` | spot | what the spot is anchored on: fixed / threat / threat_ahead / opp_unit / behind_tank / spell_group / enemy_tower (categorical) | place v1 |
| `s_x` | spot | spot x (tiles) | place v1 |
| `s_y` | spot | spot y (tiles) | place v1 |
| `s_lane` | spot | left 0 / right 1 | place v1 |
| `s_mid` | spot | distance from the middle of the arena (x) | place v1 |
| `s_own_half` | spot | on our half | place v1 |
| `s_d_threat` | spot | distance to the main threat (-1 none) | place v1 |
| `s_dx_threat` | spot | x offset from the threat | place v1 |
| `s_dy_threat` | spot | y offset from the threat | place v1 |
| `s_same_lane_threat` | spot | same lane as the threat (-1 none) | place v1 |
| `s_d_opp` | spot | distance to the nearest enemy unit | place v1 |
| `s_n_opp_near` | spot | enemy units within 2.5 tiles | place v1 |
| `s_val_opp_near` | spot | elixir of the enemy cards within 2.5 tiles | place v1 |
| `s_lane_own_val` | spot | elixir of our cards in this lane | place v1 |
| `s_lane_opp_val_theirs` | spot | enemy cards in this lane on their half | place v1 |
| `s_lane_opp_val_ours` | spot | enemy cards in this lane on our half | place v1 |
| `s_lane_opp_tower` | spot | their tower in this lane standing | place v1 |
| `s_lane_own_tower` | spot | our tower in this lane standing | place v1 |
| `s_d_own` | spot | distance to our nearest unit | place v1 |
| `p_type` | spot | the card's type | place v1 |
| `p_ranged` | spot | the card shoots from range | place v1 |
| `p_tank` | spot | the card is a tank | place v1 |
| `p_wincon` | spot | the card is a win condition | place v1 |
| `p_air` | spot | the card flies | place v1 |
| `p_ground_only` | spot | the card can't hit air | place v1 |
| `p_swarm` | spot | the card is a swarm | place v1 |
| `p_card` | spot | the card (id; categorical) | place v1 |
