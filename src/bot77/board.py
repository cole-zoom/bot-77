"""Moved to bot77.features (2026-10-22): frames / roles in bot77.features.frames, board features in
bot77.features.board. Kept so older imports work."""
from bot77.features.board import (CAND_BOARD_FEATURES, SUMMARY_FEATURES, add_board_features, candidate_features,  # noqa: F401
                                  live_board, main_threat, spell_value, summary_features)
from bot77.features.frames import (ATTACKING_Y, CROSSED_Y, FLIP_Y, TOWER_CLASSES, roles, to_player_frame)  # noqa: F401
