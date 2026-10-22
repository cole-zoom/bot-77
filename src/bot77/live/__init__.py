"""Step B v0 (2026-10-21): play Clash Royale in BlueStacks with the step-A move predictor.

state.py   one screen frame -> what the bot knows (hand, next, elixir, clock, enemy units) + the opponent's plays
policy.py  when to play, which card (A v1) and where
device.py  ADB: screenshots, taps, screen recognition (battle / end / home)
loop.py    battle -> play -> end -> repeat, with safety stops and a LanceDB log of every action
Offline checks: scripts/live_shadow.py (the bot's choices on Cole's recorded games) and scripts/live_dryrun.py (the
whole loop against a recorded video instead of BlueStacks).
"""
