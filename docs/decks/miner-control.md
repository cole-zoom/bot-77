# Miner Control
Source: *ULTIMATE* Miner Control Guide · Nate · 2023-09-25 · https://www.youtube.com/watch?v=C_BUFDd4XTE (main
source, "Nate"). Also: How to Play Miner LIKE A PRO · Mohamed Light · 2026-06-08 ·
https://www.youtube.com/watch?v=WOg2GUP0ezA ("ML"); HOW TO PLAY MINER CONTROL · Mohamed Light Extra · 2026-01-12 ·
https://www.youtube.com/watch?v=wkW1CtVY7Xw ("MLX"). Both ML videos are mostly stream chat; they're cited only
where they teach something.

Coordinates are engine cells (x, y): the bot is team 0 and defends low y, the opponent is team 1 and defends high y.
Bot princess towers (3, 6) / (14, 6), bot king (8.5, 2.5). Opponent princess towers (3, 27) / (14, 27). River
y 15.5-17.5.

## Deck
Nate's Miner Rocket list (the captions confirm it's a 3.1 deck, Nate 1:35):
Miner 3 · Rocket 6 · The Log 2 · Royal Delivery 3 · Inferno Tower 5 · Knight 3 · Electro Spirit 1 · Goblins 2.
Average 3.1 (25/8). Cheapest 4-card cycle 8 (Electro Spirit + Log + Goblins + a 3-cost card).
sim: all in the engine, and the costs above are `engine_cards()` costs.

Why this list: it's the deck the long guide plays and teaches with, for an hour and a half across 19 matchups.
Nate runs Evo Knight. The scripted opponent uses the normal Knight. Variants: Mohamed Light says he'd rather play
Miner Poison but plays Rocket because his Poison isn't maxed (MLX 15:54). He also swaps Electro Spirit for Ice
Spirit by matchup (MLX 1:16, 6:05), and his newer list pairs Miner with Magic Archer and Wall Breakers (ML 5:31,
18:11). The opponent plays the Rocket list as written.

## Game plan
It defends everything cheaply and positively, chips one tower with Miner plus one cheap card, and finishes that
tower with Rocket cycle in double elixir or overtime (Nate 12:57, 64:32). It's built to drag the game out:
"a 5-minute game every time" (MLX 4:56), and it loses when it gets too aggressive (Nate 79:29). It sticks to one lane
and never splits against beatdown (Nate 11:45, 29:22, 55:36, 70:17). It's scared of a tank it can't Inferno (when
the Inferno is out of cycle) and of air units, since only Inferno, Electro Spirit and Rocket hit air.

## Card roles
- **Miner**: win condition. Chips a tower, tanks for Goblins, and hits support troops on defense (Nate 35:55, 83:52).
- **Rocket**: finishes towers, and hits troops when it also catches the tower or a pump (Nate 12:18, 81:06; MLX 19:28).
- **Inferno Tower**: tank killer. Placed in the middle, never too far forward, and saved for the tank (Nate 6:53, 26:36, 82:45).
- **Knight**: mini-tank. Kills support behind a tank, stops bridge troops, leads counter-pushes (Nate 2:06, 27:10).
- **Royal Delivery**: small-spell defense. With the Log it kills Musketeer or Magic Archer (Nate 0:32, 63:59); also used against Minions (Nate 25:32).
- **The Log**: small spell and pre-log for Miner (Nate 35:55).
- **Electro Spirit**: cycle and chain stun. Chosen over Ice Spirit for its chain (Nate 14:00).
- **Goblins**: cheap ground DPS on defense, and Miner's chip partner (Nate 12:18, 28:49).

## Opening
- 0:00: waits to ~9 elixir, then plays Miner alone at the safe spot of a bot princess tower. Against an unknown deck,
  Nate always opens Miner in the safe spot (Nate 54:33). Safe spot = two tiles off the tower's outer back corner, where
  the defender can't wake its king with it (Nate 81:37): (1, 5) on the left tower, (16, 5) on the right **[inferred
  tiles: check legal and outside the bot king's 7.0 range]**.
- If the bot plays first, it answers with the cheapest positive defense and banks the rest.
- It never opens with Rocket or Inferno Tower, and never pairs Miner with Goblins before it has seen the bot's small
  spells **[inferred]**.

## Against our deck
All **[inferred]** unless cited.
- **Giant**: Inferno at (8.5, 24) once the Giant is past the bridge; Knight or Goblins on the support, Log or Delivery
  on small troops. That's how Nate defends Goblin Giant (Nate 26:36, 27:10).
- **Battle Ram**: Knight or Goblins at the bridge, in front of the Ram, then Log or Delivery on the Barbarians. Inferno only
  if the Ram leads a Giant push.
- **Mini P.E.K.K.A**: Goblins in front of it (2 for 4), or Knight plus Electro Spirit. Never Inferno for a lone Mini PEKKA.
- **Mega Minion**: Electro Spirit plus tower, or Inferno if it's already up. This is the deck's weak spot: Knight,
  Goblins, Miner and the Recruit can't hit it.
- **Bomber**: Knight, or Royal Delivery (Delivery alone should kill it, check in the sim), or Miner on it if it's
  near the tower.
- **Goblin Hut**: Rocket it if Hut and tower both fit in the blast; otherwise ignore it and Log the Spear Goblins
  when they clump with another troop.
- **Fireball / Arrows**: both kill Goblins and Electro Spirit, and either one plus the tower kills the Miner + Goblins
  chip. So Miner + Goblins only when those spells are out of the bot's cycle.
- **Cards of ours that hurt it**: Mega Minion (little air defense), Fireball + Arrows on its cheap defenders, and
  Giant + Mini PEKKA while the Inferno is out of cycle.

## Punish rules (checked top to bottom, every tick)
Cycle tracking: a card the bot plays can't come back to its hand until it has played 4 other cards. "Out of cycle"
means it was among the bot's last 3 plays.
1. IF a bot princess tower's HP ≤ Rocket's crown-tower damage (read from the engine) AND opp elixir ≥ 6 AND no bot
   troop is past y 18 → Rocket the tower now (Nate 12:57). Same check for Miner's damage: Miner at the safe spot.
2. IF a bot Giant is at y ≥ 14 heading up a lane AND Inferno is in hand AND opp elixir ≥ 5 → Inferno at (8.5, 24) when
   the Giant reaches y ≈ 17. If a Mini PEKKA or Ram walks ahead of the Giant, Goblins (or Knight) go in front of it first
   so the Inferno locks onto the Giant. Log or Delivery go on the support once it's within 3 tiles of the Inferno
   (Nate 26:36, 82:45).
3. IF bot troops worth ≥ 6 elixir fit in one Rocket blast (or ≥ 3 elixir plus a princess tower, or a Goblin Hut plus
   the tower) AND opp elixir ≥ 6 AND no other push needs the Inferno → Rocket centred to hit the most (Nate 12:18,
   36:26; MLX 19:28).
4. IF the bot spent > 5 elixir in one lane within 8 s → defend positively and cheaply (rule 2 for the Giant, Goblins
   for Mini PEKKA or Ram, Knight or Delivery for Bomber, Electro Spirit for Mega Minion). Then, while bot elixir ≤ 3,
   counter in the same lane: the surviving Knight or Goblins plus Miner at that lane's safe spot. It doesn't switch
   lanes (Nate 29:22, 70:17) **[inferred trigger]**.
5. IF the bot paid ≥ 4 elixir to kill a lone Miner (or Miner + one cheap card) → Miner again on the same tower
   within 10 s, with Electro Spirit, while the bot is low **[inferred]**.
6. IF Arrows AND Fireball are both out of the bot's cycle AND opp elixir ≥ 5 → Miner + Goblins at the damaged tower's
   safe spot. If only one is out, Miner + Knight. If neither, Miner alone **[inferred]**.
7. IF bot elixir ≤ 2 (counted) AND no bot troop is on the opponent's half AND opp elixir ≥ 4 → Miner + Electro Spirit at
   the damaged tower **[inferred]**.
8. IF the bot plays a Goblin Hut out of rule 3's reach → leave it. Keep chipping the same tower, and Log the
   Spear Goblins only when they stack with another troop **[inferred]**.

DEFAULT:
- Single elixir: play passively (Nate 64:32). Defend everything on its own side. Keep 5 elixir for the Inferno
  while the bot's Giant is in cycle. At ≥ 9 elixir, Miner at the safe spot of the most damaged tower (left tower if
  none is damaged).
- Double elixir: Miner + one cheap card every cycle on the same tower. Rocket the tower when it's within two
  Rockets and no push is coming.
- Overtime: Rocket cycle on the lowest tower whenever elixir ≥ 6 and nothing needs defending (Nate 64:32, 66:11).

## Scenarios
- **Giant into Inferno** · 1:20 · bot 10, opp 8 · empty board · opp hand Inferno, Goblins, Miner, Log; next Knight ·
  Learn: count the Inferno. While it's in the opponent's cycle, the Giant starts at the back and the push waits until
  the Inferno is spent, or a Mini PEKKA draws the Inferno first. · Bad play: Giant + Mini PEKKA at the left bridge
  together. Rule 2 fires (Inferno, Goblins on the Mini PEKKA), then rule 4 (Knight + Miner counter, left lane).
- **Cheap answer to chip** · 0:50 · bot 7, opp 4 · opp Miner + Goblins just landed at (1, 5) on the bot's left tower ·
  opp hand Rocket, Inferno, Knight, Electro Spirit; next Log · Learn: Arrows (3) clears it with the tower's help. ·
  Bad play: Mini PEKKA + Bomber (7) on the chip. Rule 5 fires: Miner + Electro Spirit right back, and the bot is a
  card behind for the next push.
- **Hut in Rocket range** · 1:40 · bot 6, opp 7 · empty board · opp hand Rocket, Miner, Goblins, Knight; next
  Delivery · Learn: put the Goblin Hut in the middle, around (8.5, 9), well clear of both princess towers. · Bad
  play: Hut right in front of a princess tower. Rule 3 fires: one Rocket hits Hut and tower.
- **Rocket race** · 3:30 overtime · bot 8, opp 6 · bot left tower within one Rocket of falling, opp towers full ·
  opp hand Rocket, Miner, Log, Goblins; next Inferno · Learn: Rocket is 6 elixir. Every Rocket opens a window for a
  Giant or Ram push on the other lane, so the bot has to push rather than sit back. · Bad play: holding elixir to
  defend. Rule 1 fires and the tower falls for free.

## Open questions for Cole
1. Is Nate's 2023 Miner Rocket list still the version to model, or a current one (Miner Poison, or ML's Miner +
   Magic Archer + Wall Breakers)?
2. The captions only ever say "goblins": Goblins (2) or Goblin Gang (3)? I went with Goblins.
3. Rule 4 counters in the same lane (Nate's one-lane rule), unlike the Hog file's other-lane punish. Is that how a
   good Miner Control player punishes an all-in?
4. Does Royal Delivery alone kill our Bomber at sim levels, and does Electro Spirit + tower kill a Mega Minion in
   time?
