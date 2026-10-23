"""bot77.features (2026-10-22): every feature the models see, in one place, shared by training and the live bot.

  frames    the frame conventions (player's frame, mirroring, taps vs box centres) and card roles
  hand      hand / elixir / clock / history features of a decision and each card in hand (A v1)
  rates     response-rate features: fitted on train games, cross-fitted, saved with the model
  board     the board: summary and card-vs-board features (A v2, placement)
  spots     placement candidates and their features
  datasets  training rows built from the `moments` / `plays` tables
  registry  every feature's group, level and meaning (docs/features.md)
"""
from bot77.features.board import (CAND_BOARD_FEATURES, SUMMARY_FEATURES, add_board_features, candidate_features,
                                  live_board, main_threat, spell_value, summary_features)
from bot77.features.frames import (ATTACKING_Y, CROSSED_Y, FEET_DY, FLIP_Y, RIVER_Y, TOWER_CLASSES, roles,
                                   to_player_frame)
from bot77.features.hand import (CAND_FEATURES, CTX_FEATURES, HISTORY, TIMING_FEATURES, TYPES, WAIT_ID, candidates,
                                 hud_before, timing_context, wait_row)
from bot77.features.rates import RateTables, add_response_rates
from bot77.features.registry import BY_NAME, FEATURES, Feature
from bot77.features.spots import ENEMY_TOWERS, KINDS, spot_features, spots
