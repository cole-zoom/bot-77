# Classic Log Bait
Source: Classic Log Bait RETURNS to Clash Royale!! · Ryley - Clash Royale · 2026-08-25 ·
https://www.youtube.com/watch?v=7R5FP2IYWsw

## Deck
Goblin Barrel 3 · Princess 3 · Goblin Gang 3 · Knight 3 · Inferno Tower 5 · Rocket 6 · The Log 2 · Ice Spirit 1.
Average 3.25; cheapest 4-card cycle 9 (Ice Spirit, Log, two 3s). Ryley's forms: Evo Goblin Barrel, Evo Princess,
Hero Knight (0:35); the scripted opponent plays all eight in normal form. All eight are in the engine (Evo Barrel and
Evo Princess exist there, a Hero Knight doesn't).
Our Mini P.E.K.K.A is named `Mini PEKKA` there.

## Game plan
Chip with Goblin Barrel and Princess, and keep pressure in both lanes so the bot's small spells can't cover
everything. Ryley calls it "a split-lane deck" and says the Barrel is only worth having if it threatens both sides
(9:48). Defense is cheap troops plus Inferno Tower, which is there for beatdown (0:35). Once a lead is set, or the
attack can't break through, it rocket-cycles the tower, ideally in the other lane (3:17, 3:50). It struggles against
heavy air with the Inferno out of cycle (18:57), and against decks with Log (18:23). For us that means Arrows
and Fireball.

## Card roles
- Goblin Barrel: win condition. Most effective when the bot's spells are out of cycle (2:11) or the bot is low on elixir (6:33).
- Princess: chip and defense from long range. It hides off to the side or in the back (2:11, 17:17). Placed at the bridge when the bot can't punish it (10:20). Two at once is the goal in double elixir (7:38, 8:11).
- Goblin Gang: defense against a single target (repeated prediction plays at 2:11, 10:20). Split it in the back against splash (6:33).
- Knight: mini-tank. Cycled in the corner as a pre-placed defender (2:44). Played at the bridge to back up Barrel or Princess (6:33, 12:59). Placed to wake its king tower (14:05).
- Inferno Tower: building and tank killer. Placed low or in the middle (7:05, 23:16). Goes down after a Log so it retargets onto the big unit (7:38).
- Rocket: finisher, plus a 2-for-1 on a key support unit (7:05). Rocket cycle in the last stretch (3:50, 15:41).
- The Log: the opponent's only small spell. It clears swarms and Spear Goblins, or gets cast ahead of time (2:11).
- Ice Spirit: 1-elixir freeze, which buys a Princess shot or stalls a unit (3:17, 10:20). Also cycles.

## Opening
- At 0:00 with 7+ elixir: Goblin Barrel on a princess tower (8:43, 13:32, 21:05). An early Barrel got about 600 damage (8:43).
- If the bot plays first: answer it as cheaply as possible (Gang or Knight + Ice Spirit, Log on swarms). Then Barrel on the other lane if the bot spent 4 or more (9:48).
- Never opens with Inferno Tower or Rocket. Never opens with a Princess at the bridge **[inferred]**.
- Ryley cycles both evo cards early (5:59). Our opponent doesn't use evos, so: get Barrel and Princess into play within the first 20 s **[inferred]**.

## Against our deck
- Giant: Inferno Tower in the middle of its side, (8.5, 24.5), so the Giant gets pulled from either lane (7:05, 23:16). Knight or Gang goes behind the Inferno on the support troops. We have no Zap, Electro or Freeze to reset the Inferno. Fireball and Arrows don't kill it. So a Giant played into an Inferno that's still in hand loses **[inferred]**.
- Battle Ram: Knight in the Ram's path, or Gang on the Ram after it jumps. Ice Spirit freezes it at the bridge (23:16, 23:47).
- Mini P.E.K.K.A: Goblin Gang in front of it at the river, or Knight + Ice Spirit **[inferred]**. Inferno if it comes with Giant.
- Mega Minion: Princess, the spear half of Goblin Gang, or the Inferno. Log can't hit it **[inferred]**.
- Bomber: Princess outranges it. Knight tanks the bombs. Barrel never goes into a Bomber's lane **[inferred]**.
- Goblin Hut: Log on the spawns, Princess for chip. It ignores the Hut unless a Rocket also catches the tower **[inferred]**.
- Fireball: our clean answer to Princess, Gang and Barrel goblins. After it's used, Gang and Barrel are safe for 4 bot plays **[inferred]**.
- Arrows: our only cheap answer to Barrel, Gang, Princess and Ice Spirit. The whole deck is built to make Arrows land on the wrong target. It makes us spend Arrows (bait with Princess or Gang), then sends the Barrel through (7:38: Barrel to pull out Arrows and keep the Princess stack) **[inferred for our deck]**.
- What hurts it: Bomber at the tower vs Barrel, Arrows saved for Barrel, Mega Minion vs Princess, Giant timed for when the Inferno is out of cycle **[inferred]**.

## Punish rules (checked top to bottom, every tick)
Cycle count: a card the bot plays comes back to its hand 4 bot plays later. `arrows_out` = Arrows played ≤ 3 bot
plays ago, same for `fireball_out`. `inferno_in_hand` is the opponent's own hand.
Coordinates are (x, y) in the engine frame: the bot (team 0) defends low y, the opponent (team 1) high y. Lanes x 3.0 /
14.0, middle x 8.5. Opponent bridge spot y 18-19, its back y 31-33. Bot princess towers near (3, 6) / (14, 6);
the opponent's mirror at y ~28.5. "Other lane" means the lane x the bot's last push wasn't in.
1. IF a bot Giant is on the opponent's side of the river and `inferno_in_hand` and elixir ≥ 5. THEN Inferno Tower at (8.5, 24.5). Play it when the Giant reaches y ≥ 18.5 (7:05, 23:16). If there's a support troop, Knight on it 1 s later. If the bot casts Log-able support, Log first, then Inferno (7:38).
2. IF the bot spent > 5 elixir in one lane in the last 8 s and its counted elixir ≤ 3. THEN defend that lane with the cheapest stop (rule 1, Gang, Knight + Ice Spirit). Then Goblin Barrel on the other bot princess tower (lane x, 6) plus Princess at (lane x, 19.0), within 3 s of the defense going down (7:05, 9:48, 8:11).
3. IF `arrows_out` and `fireball_out`. THEN Goblin Barrel now on a bot princess tower with no Bomber within 4 tiles: target (3, 6) or (14, 6), the lower-HP one on ties. If elixir ≥ 6, add Goblin Gang at (lane x, 18.5) or Princess at (lane x, 19.0) (2:11: Barrel after outcycling the Log) **[inferred thresholds]**.
4. IF `arrows_out` and the bot doesn't have a Bomber near that tower. THEN Barrel that tower at (lane x, 6). If only Fireball is in hand, Fireball for 3 elixir is a trade it accepts **[inferred]**.
5. IF the bot has Arrows (not `arrows_out`) and the opponent has a Barrel to play. THEN bait first: Princess at (lane x, 19.0) or Gang at (lane x, 18.5), then hold the Barrel until Arrows is cast, or until the bot drops to ≤ 2 elixir (7:38) **[inferred]**.
6. IF the bot's counted elixir ≤ 2 and the opponent's ≥ 6. THEN Knight at (lane x, 18.5), then Goblin Barrel on (lane x, 6) 1 s later (6:33, 12:59).
7. IF a bot Battle Ram or Mini P.E.K.K.A crosses alone. THEN Goblin Gang at (unit x, unit y + 2.5), capped at y 26 (or Knight there + Ice Spirit on the unit). Then Princess at (lane x, 19.0) on the counterpush (10:20).
8. IF a bot Bomber is within 4 tiles of a bot tower. THEN Princess at (lane x, 19.5) in that lane (outranges the Bomber). The Barrel goes in the other lane only **[inferred]**.
9. IF the bot has a Mega Minion, Bomber or Goblin Hut within 2 tiles of a bot princess tower and Rocket is in hand. THEN Rocket the clump plus the tower (7:05) **[inferred targets]**.
10. IF any bot princess tower HP ≤ Rocket's crown-tower damage (sim value). THEN Rocket it now. IF it's ≤ 2× that and the clock ≤ 1:00 (or overtime). THEN Rocket the tower every time Rocket comes up, other lane preferred (3:50, 17:50).
11. IF the clock ≤ 1:30, the opponent is ahead on towers, and no rule above fired. THEN only defend, plus Rocket cycle the same tower (3:50, 15:41).

DEFAULT:
- Single elixir: Knight or Princess in the back corner, (2.5 or 14.5, 31.5), as a pre-placed defender (2:44). Barrel when a spell is out (rule 3/4). Keep the Inferno for Giant or Ram. Never play the Inferno on nothing.
- Double elixir: pressure both lanes. Princess at the bridge on both sides when the bot can't Arrows both. Barrel every cycle. Rocket cycle once a tower is under 2× Rocket damage (8:11, 3:50).
- Overtime: Rocket the weakest tower. Barrel only when `arrows_out`. Otherwise defend **[inferred]**.

## Scenarios
- ARROWS ON THE BAIT · 0:45 · bot 5 / opp 7 · Goblin Gang walking down the bot's left lane; bot Fireball played 2 plays
  ago. Opp hand: Goblin Barrel, Princess, Knight, Ice Spirit; next Inferno Tower. Bot hand: Arrows, Bomber, Giant, Mega
  Minion. Learn: take the Gang with Bomber behind the tower (or let the tower plus Mega Minion do it) and keep Arrows
  for the Barrel. Bad play: Arrows on the Gang → rule 3 Barrel, nothing left to answer it.
- GIANT INTO A LIVE INFERNO · 1:20 · bot 9 / opp 7 · empty board. The opponent last played its Inferno 6 plays ago
  (so it's in hand). Opp hand: Inferno Tower, Knight, Log, Princess; next Goblin Barrel. Learn: count the Inferno. Make it
  appear first (Battle Ram at the bridge), and play the Giant right after it's used up or once it's 3+ plays out of cycle.
  Bad play: Giant at the bridge + Mini P.E.K.K.A behind → rule 1 Inferno, Knight on the Mini, then rule 2 Barrel + Princess the other lane.
- ONE-LANE ALL-IN AT THE START · 0:05 · bot 10 / opp 10 · empty board. Opp hand: Goblin Barrel, Princess, Goblin Gang,
  Inferno Tower; next Ice Spirit. Learn: against bait, don't commit > 5 in one lane before the opponent has shown
  anything. Open with Goblin Hut or Bomber in the back, or one Battle Ram. Bad play: Giant + Battle Ram in one lane →
  rule 1/7 defends for 5-8, rule 2 Barrel + Princess the other lane while the bot has ≤ 2.
- PRINCESS AT THE BRIDGE, ARROWS IN HAND · 2:10 (double) · bot 6 / opp 8 · opponent Princess just placed at the bridge in the bot's right lane.
  Bot right tower 1400. Opp hand: Goblin Barrel, Gang, Knight, Rocket; next Log. Bot hand: Arrows, Mega Minion, Fireball,
  Battle Ram. Learn: kill the Princess with Mega Minion (or ignore it and Ram the other lane), and keep both spells for the
  Barrel. Bad play: Arrows on the Princess → rule 3/4 Barrel. Fireball alone saves only part of the tower, which then drops into
  Rocket range → rule 10.
- ROCKET RANGE IN THE ENDGAME · 2:45 (double) · bot 7 / opp 4 · bot left tower 600 (≤ 2× Rocket), opponent towers
  full. Opp hand: Rocket, Knight, Ice Spirit, Log; next Goblin Barrel. Learn: a tower in Rocket range is already lost.
  Use the elixir to take a tower back (Giant + Mega Minion in the right lane while the opponent sits at 4), not to stack
  defense at the dead tower. Bad play: Bomber + Goblin Hut next to the low tower → rule 9 Rocket takes tower and both cards.

## Open questions for Cole
1. Rule 1 placement: (8.5, 24.5) in the middle. Or does a good player put it 1-2 tiles toward the Giant's lane so it
   locks sooner?
2. Rule 3/4: is "Arrows played ≤ 3 bot plays ago" the right Barrel window, or do players at your level also
   wait for the bot's elixir to drop below 3?
3. Should the opponent ever Log our Bomber or Goblin Hut spawns ahead of the Barrel (Log-Barrel), or keep the Log only for
   Gang-type defense?
4. Mini P.E.K.K.A answer: Goblin Gang at the river, or let the Inferno pull it once the Giant is out of cycle?
