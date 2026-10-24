"""The simulator (docs/simulator.md): the ClashRoyaleEnv engine (MIT, forked as third_party/clash-royale-env) set up
from our data and read back in our frame. Nothing outside this package imports the engine.

  geometry   our tile frame <-> the engine's board, teams
  cards      our card ids / names <-> engine ids
  engine     Sim: one simulated match (play, step, units, towers, snapshot)
  scenario   scenario_from_moment: a Sim set up from a `moments` row
"""
