"""Render generator sprites from the game's own 2D art (decisions.md, 2026-10-10, "Game asset extraction").

Source: the Play Asset Delivery pack of Cole's BlueStacks install, unpacked to ~/tools/bluestacks/pack (local files only,
never committed: the repo is public). Decoder: sc-editor's CLI (danila-schelkov/sc-editor, MIT), wrapped by
~/tools/sc-editor/sce, which reads Supercell's SC2 (.sc v5) + .sctx files and renders a display object to a PNG.

Every unit animation export is named <prefix><anim><n>_<dir>, e.g. giant1_enemy_run1_9: one movie clip per animation and
direction (1 = facing away, 9 = facing the camera; Sparky / evo Cannon use 18). SPEC maps our classes (config/classes_v2.json)
to a file and the red (opponent) / blue export prefixes, one entry per skin (Star Level skins are "prestige" files).
Several prefixes in one entry are layers, composited in order (e.g. Ram Rider = ram + rider). Buildings and a few
others have no animation naming: their exports are listed one by one ("exp").

Renders use a fixed 512x512 canvas centred on the unit's origin (layers line up), then crop to the alpha box. Native
scale is kept; setup_synth_v3.py rescales to the arena crop.

  uv run python scripts/extract_game_sprites.py [--only giant,knight] [--jobs 8]
Writes ~/tools/bluestacks/sprites/<class>/<class>_<team>_gs-....png (team 0 blue, 1 red, as KataCR's generator expects)
and sprites/manifest.json.
"""
import argparse, collections, hashlib, json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / "tools/bluestacks"
SC, INSPECT, RENDER, OUT = ROOT / "pack/assets/sc", ROOT / "inspect", ROOT / "render", ROOT / "sprites"
SCE = Path.home() / "tools/sc-editor/sce"
CANVAS, ORIGIN_DY = 512, -90  # camera offset (any non-zero offset switches sc-editor to an origin-centred camera)
ANIM = re.compile(r"^(run|idle|attack|attacking|walk|charge|dash|jump|deploy|hover|ability|reborn)(\d+)_(\d+)$", re.I)
FRAMES = {"run": 4, "attack": 4, "attacking": 4, "idle": 2}  # frames sampled per animation and direction (others: 2)
# Removed after the contact-sheet review (2026-10-10): evo Minion Horde (renders as a dark silhouette), Elixir
# Collector (frames carry a UI badge), Mother Witch skin 2 (solid gold blob), Cannon Cart skin 2, Archer Queen's ability (a cloak shape).
SKIP_ANIM = {"archer-queen": {"ability"}}


def A(file, red, blue=None, skin="base"):
    """Animated unit: red / blue layer prefixes (blue None = same art for both teams)."""
    red = [red] if isinstance(red, str) else red
    blue = red if blue is None else [blue] if isinstance(blue, str) else blue
    return ("anim", file, red, blue, skin)


def E(file, red, blue=None, skin="base", frames=3):
    """Exports rendered as they are (buildings, projectiles)."""
    red = [red] if isinstance(red, str) else red
    blue = red if blue is None else [blue] if isinstance(blue, str) else blue
    return ("exp", file, red, blue, skin, frames)


P2, P3, P4, P5, P6, P7 = (f"chr_prestige{n}_dl" for n in (2, 3, 4, 5, 6, 7))
P = "chr_prestige_dl"
SPEC = {
    "skeleton": [A("chr_skeleton", "skeleton1_"), A(P2, "skeleton_prestige_", skin="s1"), A(P2, "skeleton_prestige2_", skin="s2")],
    "skeleton-evolution": [A("chr_skeleton_evolution", "evoskeleton1_")],
    "electro-spirit": [A("chr_electro_spirit_dl", "electrospirit_")],
    "fire-spirit": [A("chr_fire_firespirit", "firespirit_")],
    "ice-spirit": [A("chr_ice_spirits", "icespirit_")],
    "ice-spirit-evolution": [A("chr_ice_spirit_evolution", "icespirit_evolution_")],
    "heal-spirit": [A("chr_heal_spirit_dl", "healspirit_")],
    "goblin": [A("chr_goblin", "goblin1_enemy_", "goblin1_"), A(P4, "goblin_prestige_", skin="s1"), A(P4, "goblin_prestige2_", skin="s2")],
    "spear-goblin": [A("chr_goblin_archer", "goblinArcher_", "goblinArcher_blue_"),
                     A(P, "spear_goblin_prestige_", skin="s1"), A(P, "spear_goblin_prestige2_", skin="s2")],
    "bomber": [A("chr_bomber", "chr_bomber_RED_", "chr_bomber_"), A("chr_bomber", "chr_bomber_prestige2_", skin="s2")],
    "bomber-evolution": [A("chr_bomber_evolution", "bomber_evolution_enemy_", "bomber_evolution_")],
    "bat": [A("chr_bats", "bat_"), A(P3, "bat_prestige_", skin="s1")],
    "bat-evolution": [A("chr_bats_evolution", "bats_evolution_")],
    "ice-golem": [A("chr_snowman", "snowman_red_", "snowman_blue_")],
    "barbarian": [A("chr_barbarian", "barbarian1_enemy_", "barbarian1_"), A(P2, "barbarian_prestige_", skin="s1")],
    "barbarian-evolution": [A("chr_barbarian_evolution", "enemy_barbarian1_", "barbarian1_")],
    "elite-barbarian": [A("chr_barbarian", "angry_barbarian_red_", "angry_barbarian_"), A(P6, "angry_barbarian_prestige_", skin="s1")],
    "wall-breaker": [A("chr_wall_breaker_dl", "chr_wall_breaker_red_", "chr_wall_breaker_"),
                     A("chr_wall_breaker_dl", "wall_breaker_prestige2_", skin="s2"), A("chr_wall_breaker_dl", "wall_breaker_prestige3_", skin="s3")],
    "wall-breaker-evolution": [A("chr_wall_breakers_evolution", "wall_breakers_evolution_enemy_", "wall_breakers_evolution_"),
                               A("chr_wall_breakers_evolution", "wall_breakers_solo_evolution_enemy_", "wall_breakers_solo_evolution_", skin="solo")],
    "archer": [A("chr_archer", "archer1_enemy_", "archer1_"), A(P6, "archer_prestige_", skin="s1")],
    "archer-evolution": [A("chr_archer_evolution", "archer_evolution_enemy_", "archer_evolution_")],
    "knight": [A("chr_knight", "Knight_enemy_", "Knight_"), A(P4, "Knight_prestige_", skin="s1"), A(P4, "Knight_prestige2_", skin="s2")],
    "knight-evolution": [A("chr_knight_evolution", "knight_evolution_enemy_", "knight_evolution_")],
    "minion": [A("chr_minion", "minion1_enemy_", "minion1_"), A(P5, "minion_prestige_", skin="s1"),
               A(P5, "minion_prestige2_enemy_", "minion_prestige2_", skin="s2")],
    "firecracker": [A("chr_firecracker_dl", "chr_firecracker_red_", "chr_firecracker_")],
    "firecracker-evolution": [A("chr_firecracker_evolution", "evofirecracker_enemy_", "evofirecracker_")],
    "royal-recruit": [A("chr_guard02_dl", "guard02_red_shield_", "guard02_blue_shiled_"),
                      A("chr_guard02_dl", "guard02_red_", "guard02_blue_", skin="noshield"),
                      A("chr_guard02_dl", "guard02_prestige2_shield_", skin="s2"), A("chr_guard02_dl", "guard02_prestige2_", skin="s2noshield")],
    "royal-recruit-evolution": [A("chr_royal_recruit_evolution", "royal_recruit_enemy_shield_evolution_", "royal_recruit_shield_evolution_"),
                                A("chr_royal_recruit_evolution", "royal_recruit_enemy_evolution_", "royal_recruit_evolution_", skin="noshield")],
    "mega-minion": [A("chr_mega_minion", "mega_minion_red_", "mega_minion_"), A(P5, "mega_minion_prestige_", skin="s1")],
    "dart-goblin": [A("chr_goblin_blowdart", "goblin_blowdart_red_", "goblin_blowdart_"),
                    A(P4, "blowdart_prestige_red_", "blowdart_prestige_", skin="s1")],
    "dart-goblin-evolution": [A("chr_goblin_blowdart_evolution", "Blowdart_Goblin_Evo_red_", "Blowdart_Goblin_Evo_")],
    "elixir-golem-big": [A("chr_elixir_golem_dl", "elixir_golem_red_", "elixir_golem_")],
    "elixir-golem-mid": [A("chr_elixir_golem_dl", "elixir_golemite_red_", "elixir_golemite_")],
    "elixir-golem-small": [A("chr_elixir_golem_dl", "elixir_blob_")],
    "guard": [A("chr_skeleton_warrior", "skeletonWarrior_shield_", "skeletonWarrior_blue_shield_"),
              A("chr_skeleton_warrior", "skeletonWarrior_", "skeletonWarrior_blue_", skin="noshield"),
              A(P, "skeletonwarrior_prestige_shield_", skin="s1"), A(P, "skeletonwarrior_prestige_", skin="s1noshield")],
    "miner": [A("chr_miner", "miner_red_", "miner_blue_"), A(P7, "miner_prestige_", skin="s1")],
    "princess": [A("chr_princess", "princess_red_", "princess_")],
    "princess-evolution": [A("chr_princess_evolution", "princess_evo_red_", "princess_evo_")],
    "ice-wizard": [A("chr_ice_wizard", "ice_wizard_red_", "ice_wizard_")],
    "royal-ghost": [A("chr_ghost_dl", "ghost_red_", "ghost_blue_")],
    "royal-ghost-evolution": [A("chr_royal_ghost_evolution", "royal_ghost_evo_red_", "royal_ghost_evo_")],
    "bandit": [A("chr_bandit", "bandit_red_", "bandit_")],
    "fisherman": [A("chr_fisherman_dl", "fisherman_red_", "fisherman_")],
    "skeleton-dragon": [A("chr_skeleton_dragon_dl", "skeletondragon_enemy_", "skeletondragon_")],
    "mini-pekka": [A("chr_mini_pekka", "minipekka_red_", "minipekka_"), A(P5, "minipekka_prestige_", skin="s1")],
    "musketeer": [A("chr_musketeer", "chr_musketeer_", "chr_musketeer_blue_"), A(P2, "musketeer_prestige_", skin="s1"),
                  A(P2, "musketeer_prestige2_", skin="s2")],
    "musketeer-evolution": [A("chr_musketeer_evolution", "chr_musketeer_evo_red_", "chr_musketeer_evo_")],
    "goblin-brawler": [A("chr_goblin_thug_dl", "goblin_thug_red_", "goblin_thug_"),
                       A("chr_goblin_thug_dl", "goblin_thug_prestige2_", skin="s2"), A("chr_goblin_thug_dl", "goblin_thug_prestige3_", skin="s3")],
    "valkyrie": [A("chr_valkyrie", "valkyrie1_enemy_", "valkyrie1_"), A(P2, "valkyrie1_prestige_", skin="s1")],
    "valkyrie-evolution": [A("chr_valkyrie_evolution", "valkyrie_evolution_enemy_", "valkyrie_evolution_")],
    "battle-ram": [A("chr_battle_ram", "battle_ram_red_", "battle_ram_"), A(P3, "battle_ram_prestige_", skin="s1")],
    "battle-ram-evolution": [A("chr_battle_ram_evolution", "battle_ram_evolution_enemy_", "battle_ram_evolution_")],
    "flying-machine": [A("chr_flying_machine", "flying_machine_red_", "flying_machine_")],
    "hog-rider": [A("chr_hog_rider", "hog_rider_red_", "hog_rider_"), A(P4, "hog_rider_prestige_", skin="s1"),
                  A(P4, "hog_rider_prestige2_", skin="s2")],
    "battle-healer": [A("chr_battle_healer_dl", "battle_healer_red_", "battle_healer_blue_")],
    "zappy": [A("chr_zappies_dl", "zappie_red_", "zappie_blue_"), A("chr_zappies_dl", "zappie_prestige_red_", "zappie_prestige_", skin="s1")],
    "baby-dragon": [A("chr_baby_dragon", "baby_dragon_enemy_", "baby_dragon_"), A(P6, "baby_dragon_prestige_", skin="s1")],
    "baby-dragon-evolution": [A("chr_baby_dragon_evolution", "baby_dragon_evo_enemy_", "baby_dragon_evo_")],
    "inferno-dragon": [A("chr_baby_dragon", "inferno_dragon_red_", "inferno_dragon_"), A(P, "inferno_dragon_prestige_", skin="s1"),
                       A(P, "inferno_dragon_prestige2_", skin="s2")],
    "inferno-dragon-evolution": [A("chr_inferno_dragon_evolution", ["inferno_dragon_evo_barrel_enemy_", "inferno_dragon_evo_enemy_"],
                                   ["inferno_dragon_evo_barrel_", "inferno_dragon_evo_"])],
    "dark-prince": [A("chr_black_knight", "BlackKnight_shield_red_", "BlackKnight_shield_blue_"),
                    A("chr_black_knight", "BlackKnight_red_", "BlackKnight_blue_", skin="noshield")],
    "hunter": [A("chr_hunter_dl", "hunter_red_", "hunter_"), A("chr_hunter_dl", "hunter_prestige_", skin="s1"),
               A("chr_hunter_dl", "hunter_prestige2_", skin="s2")],
    "hunter-evolution": [A("chr_hunter_evolution", "evo_hunter_red_", "evo_hunter_")],
    "electro-wizard": [A("chr_electro_wizard", "electro_wizard_red_", "electro_wizard_")],
    "phoenix-big": [A("chr_phoenix_dl", "Phoenix_")],
    "magic-archer": [A("chr_magic_archer_dl", "magic_archer_red_", "magic_archer_"),
                     A("chr_magic_archer_dl", "magic_archer_red_prestige_", "magic_archer_prestige_", skin="s1"),
                     A("chr_magic_archer_dl", "magic_archer_red_prestige2_", "magic_archer_prestige2_", skin="s2")],
    "lumberjack": [A("chr_rage_barbarian", "rage_barbarian_red_", "rage_barbarian_")],
    "lumberjack-evolution": [A("chr_lumberjack_evolution", "lumberjack_evolution_enemy_", "lumberjack_evolution_"),
                             A("chr_lumberjack_evolution", "lumberjack_ghost_evolution_enemy_", "lumberjack_ghost_evolution_", skin="ghost")],
    "night-witch": [A("chr_dark_witch", "dark_witch_red_", "dark_witch_"), A(P3, "dark_witch_prestige_", skin="s1")],
    "mother-witch": [A("chr_mother_witch_dl", "motherwitch_red_", "motherwitch_"),
                     A("chr_mother_witch_prestige_dl", "motherwitch_prestige_", skin="s1")],
    "golden-knight": [A("chr_golden_knight_dl", "GoldenKnight_enemy_", "GoldenKnight_")],
    "skeleton-king": [A("chr_skeleton_king_dl", "SkeletonKing_enemy_", "SkeletonKing_")],
    "mighty-miner": [A("chr_mighty_miner_dl", "MightyMiner_enemy_", "MightyMiner_")],
    "rascal-boy": [A("chr_rascals_dl", "rascal_boy_red_", "rascal_boy_"), A(P, "rascal_boy_prestige_", skin="s1")],
    "rascal-girl": [A("chr_rascals_dl", "rascal_girl_red_", "rascal_girl_"), A(P, "rascal_girl_prestige_", skin="s1")],
    "giant": [A("chr_giant", "giant1_enemy_", "giant1_"), A(P3, "giant_prestige_", skin="s1")],
    "wizard": [A("chr_wizard", "wizard1_red_", "wizard1_"), A(P2, "wizard_prestige_", skin="s1")],
    "wizard-evolution": [A("chr_wizard_evolution", "wizard1_red_shield_", "wizard1_shield_")],
    "royal-hog": [A("chr_royal_hog_dl", "royal_hog_red_", "royal_hog_"), A("chr_royal_hog_dl", "royal_hog_prestige_", skin="s1")],
    "royal-hogs-evolution": [A("chr_royal_hog_evolution", "royal_hogs_evolution_red_", "royal_hogs_evolution_"),
                             A("chr_royal_hog_evolution", "royal_hogs_evolution_ground_red_", "royal_hogs_evolution_ground_", skin="ground")],
    "witch": [A("chr_witch", "witch1_enemy_", "witch1_"), A(P2, "witch1_prestige_", skin="s1")],
    "witch-evolution": [A("chr_witch_evolution", ["evo_witch_red_", "evo_witch_weapon_gem_"], ["evo_witch_", "evo_witch_weapon_gem_"])],
    "balloon": [A("chr_balloon", "balloon1_enemy_", "balloon1_"), A(P6, "balloon_prestige_", skin="s1"), A(P6, "balloon_prestige2_", skin="s2")],
    "prince": [A("chr_prince", "Prince_red_", "Prince_")],
    "electro-dragon": [A("chr_electro_dragon_dl", "electro_dragon_red_", "electro_dragon_")],
    "electro-dragon-evolution": [A("chr_electro_dragon_evolution", "electro_dragon_evo_red_", "electro_dragon_evo_")],
    "bowler": [A("chr_bowler", "bowler_red_", "bowler_blue_"), A(P3, "bowler_blue_prestige_", skin="s1")],
    "executioner": [A("chr_axe_man", "axe_man_red_", "axe_man_"), A(P6, "axe_man_prestige_", skin="s1")],
    "executioner-evolution": [A("chr_executioner_evolution", "executioner_evo_red_", "executioner_evo_")],
    "cannon-cart": [A("chr_movingcannon", "moving_cannon_red_", "moving_cannon_"),
                    A("chr_movingcannon", "broken_cannon_red_", "broken_cannon_", skin="broken"),
                    A(P, "moving_cannon_prestige_", skin="s1")],  # skin 2's directions don't match the base (no underlay)
    "ram-rider": [A("chr_ram_rider_dl", ["ram_red_", "rider_"], ["ram_", "rider_"])],
    "archer-queen": [A("chr_archer_queen_dl", "ArcherQueen_enemy_", "ArcherQueen_")],
    "monk": [A("chr_mighty_monk_dl", "Monk_enemy_", "Monk_")],
    "royal-giant": [A("chr_royal_giant", "royalGiant_red_", "royalGiant_"), A(P, "royal_giant_prestige_", skin="s1")],
    "royal-giant-evolution": [A("chr_royalgiant_evolution", "enemy_royalgiant_evo_", "royalgiant_evo_")],
    "giant-skeleton": [A("chr_giant_skeleton", "giant_skeleton_enemy_", "giant_skeleton_"),
                       A(P4, "giant_skeleton_prestige_red_", "giant_skeleton_prestige_", skin="s1"),
                       A(P4, "giant_skeleton_prestige_red2_", "giant_skeleton_prestige2_", skin="s2")],
    "goblin-giant": [A("chr_goblin_giant_dl", ["goblin_bag_red_", "goblin_giant_red_"], ["goblin_bag_blue_", "goblin_giant_"]),
                     A("chr_goblin_giant_dl", ["goblin_bag_prestige2_", "goblin_giant_prestige2_"], skin="s2")],
    "goblin-giant-evolution": [A("chr_goblin_giant_evolution", ["Goblin_Giant_Evolution_Bag_", "Goblin_Giant_Evolution_red_"],
                                 ["Goblin_Giant_Evolution_Bag_", "Goblin_Giant_Evolution_"])],
    "sparky": [A("chr_zap_machine", "ZapMachine_", "ZapMachine_Blue_")],
    "pekka": [A("chr_pekka", "pekka_enemy_", "pekka_"), A(P2, "pekka_prestige_", skin="s1")],
    "pekka-evolution": [A("chr_pekka_evolution", "pekka_evolution_red_", "pekka_evolution_")],
    "electro-giant": [A("chr_electro_giant_dl", "electrogiant_enemy_", "electrogiant_")],
    "mega-knight": [A("chr_mega_knight", "mega_knight_red_", "mega_knight_"),
                    A(P5, "mega_knight_red_prestige_", "mega_knight_prestige_", skin="s1")],
    "mega-knight-evolution": [A("chr_mega_knight_evolution", "megaknight_evolution_red_", "megaknight_evolution_")],
    "lava-hound": [A("chr_lava_hound", "lava_hound_red_", "lava_hound_"), A(P5, "lava_hound_prestige_", skin="s1")],
    "lava-pup": [A("chr_lava_pups", "lava_pups_")],
    "golem": [A("chr_golem_dl", "golem1_red_", "golem1_"), A("chr_golem_dl", "golem1_prestige_", skin="s1")],
    "golemite": [A("chr_golemite", "golemite_red_", "golemite_"), A(P, "golemite_prestige_", skin="s1")],
    "little-prince": [A("chr_little_prince_dl", "LittlePrince_enemy_", "LittlePrince_")],
    "royal-guardian": [A("chr_little_prince_guard_dl", "PrinceGuard_enemy_", "PrinceGuard_")],
    "skeleton-barrel": [A("chr_skeleton_balloon", "skeleton_balloon_red_", "skeleton_balloon_"),
                        A(P, "skeleton_balloon_prestige_", skin="s1"), A(P, "skeleton_balloon_prestige2_", skin="s2")],
    "skeleton-barrel-evolution": [A("chr_skeleton_balloon_evolution", "evo_skeleton_balloon_red_", "evo_skeleton_balloon_")],
    "skeleton-army-evolution": [A("chr_skeleton_army_evolution", "skeleton_general_shield_red_", "skeleton_general_shield_"),
                                A("chr_skeleton_army_evolution", "skeleton_general_red_", "skeleton_general_", skin="noshield")],
    "goblin-cage-evolution": [A("chr_goblin_cage_evolution", "evo_goblincage_enemy_", "evo_goblincage_")],
    "furnace": [A("chr_furnace_rework", "furnace_rework_red_", "furnace_rework_")],
    "furnace-evolution": [A("chr_furnace_evolution", ["furnace_evo_red_", "furnace_evo_hat_"], ["furnace_evo_", "furnace_evo_hat_"])],
    # cards from 2025-26 (not in KataCR)
    "goblinstein-doctor": [A("chr_goblinstein_doctor", "Goblinstein_Doctor_red_", "Goblinstein_Doctor_")],
    "goblinstein-monster": [A("chr_goblinstein_monster", "Goblinstein_Monster_red_", "Goblinstein_Monster_")],
    "berserker": [A("chr_berserker", "berserker_enemy_", "berserker_")],
    "boss-bandit": [A("chr_boss_bandit", "BossBandit_Enemy_", "BossBandit_")],
    "goblin-demolisher": [A("chr_goblin_demolisher", "goblin_demolisher_red_", "goblin_demolisher_"),
                          A("chr_goblin_demolisher", "goblin_demolisher_2_red_", "goblin_demolisher_2_", skin="phase2")],
    "goblin-machine": [A("chr_goblin_machine", "goblin_machine_")],
    "suspicious-bush": [A("chr_goblin_bush", "bush_goblin_bush_red_", "bush_goblin_bush_")],
    "rune-giant": [A("chr_giant_buffing", "buffing_giant_", "buffing_giant_blue_")],  # codename "buffing giant" (to verify)
    "spirit-empress-ground": [A("chr_merge_maiden", "merge_maiden_enemy_", "merge_maiden_")],  # codename (to verify)
    "spirit-empress-air": [A("chr_merge_maiden_dragon", "merge_maiden_dragon_enemy_", "merge_maiden_dragon_")],
    # buildings
    "cannon": [E("building_basic_cannon", "building_enemy_cannon", "building_cannon"),
               E("building_basic_cannon", "cannon_prestige_1_red", "cannon_prestige_1", skin="s1"),
               E("building_basic_cannon", "cannon_prestige_2_red", "cannon_prestige_2", skin="s2")],
    "cannon-evolution": [A("building_cannon_evolution", "building_cannon_evolution_red_", "building_cannon_evolution_")],
    "tesla": [E("building_tesla", "tesla1_red", "tesla1_blue", frames=4), E("building_tesla", "tesla2_red_idle1_1", "tesla2_blue_idle1_1", skin="down", frames=1)],
    "tesla-evolution": [E("building_tesla_evolution", "tesla_enemy_evolution", "tesla_evolution", frames=4)],
    "mortar": [A("building_mortar", "building_mortar1_enemy_", "building_mortar1_"), A("building_prestige_dl", "mortar_prestige_red_", skin="s1")],
    "mortar-evolution": [A("building_mortar_evolution", "enemy_mortar_evolution_", "mortar_evolution_")],
    "x-bow": [E("building_xbow", "building_xbow_red_attack1", "building_xbow_attack1_1", frames=4),
              E("building_prestige_dl", "xbow_red_prestige", skin="s1"), E("building_prestige_dl", "xbow_red_prestige2", skin="s2")],
    "inferno-tower": [E("building_inferno_tower", "inferno_red_idle1_1", "inferno_blue_idle1_1", frames=1)],
    "bomb-tower": [E("building_bomb_tower", ["building_bomb_tower_red", "building_bomb_tower_top_red"],
                     ["building_bomb_tower_blue", "building_bomb_tower_top_blue"], frames=1)],
    "tombstone": [E("building_tombstone", "building_tombstone_red", "building_tombstone_blue", frames=4),
                  E("building_prestige_dl", "tombstone_prestige_red", "tombstone_prestige_blue", skin="s1", frames=4)],
    "goblin-hut": [E("building_goblin_hut", "Building_Goblin_Hut_2025Rework_Red", "Building_Goblin_Hut_2025Rework_Blue", frames=1)],
    "barbarian-hut": [E("building_barbarian_hut", "building_barbarian_hut_red", "building_barbarian_hut_blue", frames=1)],
    "goblin-drill": [E("building_goblin_drill", "enemy_goblin_drill", "goblin_drill", frames=6)],
    "goblin-drill-evolution": [E("building_goblin_drill_evolution", "goblin_drill_evolution_enemy", "goblin_drill_evolution", frames=6)],
    # Vines: a coil drawn around the trapped unit, back layer + front layer, five sizes; loop and growing frames
    "vines": [E("spell_vine", [f"{v}_back_{ph}", f"{v}_front_{ph}"], skin=f"{v.split('_')[-1]}{ph}", frames=n)
              for v in ("vine_S", "vine_M", "vine_L", "vine2_XL", "vine2_XXL") for ph, n in (("loop", 5), ("start", 4))],
    # projectiles that are labelled while flying
    "goblin-barrel": [E("spell_goblin_barrel", "spell_goblin_barrel_projectile", frames=4)],
    "goblin-barrel-evolution": [E("spell_goblin_barrel_evolution", "spell_goblin_barrel_evolution_projectile", frames=4)],
    "royal-delivery": [E("spell_royale_delivery_dl", "royale_delivery_red", "royale_delivery_blue", frames=4)],
    "giant-snowball": [E("spell_snowball_dl", "snowball_projectile_enemy", "snowball_projectile", frames=2)],
    "giant-snowball-evolution": [E("spell_snowball_evolution", "snowball_evolution_projectile_enemy", "snowball_evolution_projectile", frames=2)],
}


def inspect(file):
    j = INSPECT / f"{file}.sc.json"
    if not j.exists():
        j.write_text(subprocess.run([str(SCE), "inspect", "-i", str(SC / f"{file}.sc"), "--json"], capture_output=True, text=True).stdout)
    d = json.loads(j.read_text())
    fc = {c["id"]: c["frameCount"] for c in d["clips"]}
    return {k: fc.get(v, 1) for k, v in d["exports"].items()}


def frame_idx(n, k):
    return sorted({int(i * n / k) for i in range(k)}) if n > 1 else [0]


def plan(spec):
    """-> sprites: list of (class, team, skin, tag, layers [(file, export, frame, under)]), renders: {(file, frame): {export}}

    Star Level ("prestige") skins mostly hold only the pieces drawn over the base unit (gold trim, hats), so a skin entry
    gets the class's base entry as an underlay; main() drops the underlay when the skin turns out to be full art."""
    sprites, renders, missing = [], collections.defaultdict(set), []
    for cls, entries in spec.items():
        base_e = entries[0]
        for e in entries:
            kind, file = e[0], e[1]
            ex = inspect(file)
            skinned = e is not base_e and "prestige" in (file + "".join(e[2])).lower()
            bex = inspect(base_e[1]) if skinned else {}
            for team, layers in ((1, e[2]), (0, e[3])):
                under = (base_e[2] if team == 1 else base_e[3]) if skinned else []
                if kind == "exp":
                    n = ex.get(layers[0])
                    if n is None:
                        missing.append((cls, layers[0])); continue
                    for f in frame_idx(n, e[5]):
                        L = [(base_e[1], x, min(f, bex[x] - 1), True) for x in under if x in bex]
                        L += [(file, x, min(f, ex.get(x, 1) - 1), False) for x in layers if x in ex]
                        sprites.append((cls, team, e[4], f"f{f:03d}", L))
                        for lf, x, ff, _ in L:
                            renders[(lf, ff)].add(x)
                    continue
                found = 0
                for name, n in ex.items():
                    if not name.startswith(layers[0]):
                        continue
                    m = ANIM.match(name[len(layers[0]):])
                    if not m:
                        continue
                    found += 1
                    anim, suffix = m[1].lower(), name[len(layers[0]):]
                    if anim in SKIP_ANIM.get(cls, ()):
                        continue
                    for f in frame_idx(n, FRAMES.get(anim, 2)):
                        L = [(base_e[1], p + suffix, min(f, bex[p + suffix] - 1), True) for p in under if p + suffix in bex]
                        L += [(file, p + suffix, min(f, ex[p + suffix] - 1), False) for p in layers if p + suffix in ex]
                        sprites.append((cls, team, e[4], f"{anim}{m[2]}-d{int(m[3]):02d}-f{f:03d}", L))
                        for lf, x, ff, _ in L:
                            renders[(lf, ff)].add(x)
                if not found:
                    missing.append((cls, layers[0]))
    return sprites, renders, missing


def render(job):
    (file, f), exports = job
    out = RENDER / file / f"f{f:03d}"
    todo = sorted(x for x in exports if not (out / f"{x}.png").exists())
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), 80):
        cmd = [str(SCE), "render", "-i", str(SC / f"{file}.sc"), "-o", str(out), "-w", str(CANVAS), "-h", str(CANVAS),
               "-tx", "0.001", "-ty", str(ORIGIN_DY), "--start-frame", str(f), "--end-frame", str(f)]
        for x in todo[i:i + 80]:
            cmd += ["--exports", x]
        subprocess.run(cmd, capture_output=True, text=True)
    return sum((out / f"{x}.png").exists() for x in exports), len(exports)


def over(dst, src):
    a = src[:, :, 3:4] / 255.0
    out = dst.astype(np.float32)
    out[:, :, :3] = src[:, :, :3] * a + out[:, :, :3] * (1 - a)
    out[:, :, 3:4] = 255 * (a + out[:, :, 3:4] / 255.0 * (1 - a))
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    spec = {k: v for k, v in SPEC.items() if not a.only or k in a.only.split(",")}
    classes = json.loads(Path("config/classes_v2.json").read_text())
    assert not set(spec) - set(classes), set(spec) - set(classes)
    sprites, renders, missing = plan(spec)
    print(f"{len(spec)} classes, {len(sprites)} sprites, {len(renders)} render calls; prefixes with no animation: {missing}", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        done = list(pool.map(render, renders.items()))
    print("rendered", sum(d[0] for d in done), "of", sum(d[1] for d in done), flush=True)

    manifest, stats, seen = [], collections.Counter(), set()
    for cls, team, skin, tag, layers in sprites:
        img = {True: None, False: None}
        for file, x, f, under in layers:
            p = RENDER / file / f"f{f:03d}" / f"{x}.png"
            if not p.exists():
                continue
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            img[under] = im if img[under] is None else over(img[under], im)
        main_, under_ = img[False], img[True]
        if main_ is not None and under_ is not None:
            # skin = overlay pieces unless it covers most of the base unit by itself
            m_, u_ = main_[:, :, 3] > 8, under_[:, :, 3] > 8
            full = (m_ & u_).sum() >= 0.7 * u_.sum()  # covers most of the base unit itself (a gold sword alone does not)
            stats["skin_full_art" if full else "skin_overlay"] += 1
            img = main_ if full else over(under_, main_)
        else:
            img = main_
        if img is None:
            stats["not_rendered"] += 1; continue
        ys, xs = np.nonzero(img[:, :, 3] > 8)
        if len(xs) == 0 or xs.max() - xs.min() < 6 or ys.max() - ys.min() < 6:
            stats["empty"] += 1; continue
        if xs.min() == 0 or ys.min() == 0 or xs.max() == CANVAS - 1 or ys.max() == CANVAS - 1:
            stats["clipped_by_canvas"] += 1; continue
        crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        px = crop[crop[:, :, 3] > 128][:, :3].astype(np.float32)
        # some skin frames include mask shapes sc-editor draws as flat black / dark blocks
        if len(px) == 0 or (px.max(1) < 40).mean() > 0.25 or px.std(0).mean() < 12:
            stats["dark_mask_artifact"] += 1; continue
        h = hashlib.md5(cv2.resize(crop, (24, 24)).tobytes()).hexdigest()
        if (cls, team, h) in seen:
            stats["duplicate"] += 1; continue
        seen.add((cls, team, h))
        f = OUT / cls / f"{cls}_{team}_gs-{skin}-{tag}.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(f), crop)
        manifest.append({"file": str(f), "class": cls, "team": team, "skin": skin, "tag": tag, "size": [int(crop.shape[1]), int(crop.shape[0])]})
        stats["kept"] += 1
    old = [m for m in json.loads((OUT / "manifest.json").read_text())] if (OUT / "manifest.json").exists() and a.only else []
    manifest = [m for m in old if m["class"] not in spec] + manifest
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    per = collections.Counter(m["class"] for m in manifest)
    print(dict(stats)); print("per class:", dict(sorted(per.items())))


if __name__ == "__main__":
    main()
