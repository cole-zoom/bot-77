# 2.6 Hog Cycle
Source: How To Push Through CHAMPION League With 2.6 Hog in Clash Royale! · Leviathan CR · 2026-10-18 ·
https://www.youtube.com/watch?v=66ooo4LP0Is

Placements are (x, y) in the engine frame: 18 x 34, the bot defends low y, this opponent high y, river y 15.5-17.5.
Lanes x 3.0 / 14.0; the opponent's bridge spots (2.5, 18.5) / (14.5, 18.5); its Cannon spot (8.5, 24.5). "Push lane" is
the lane the bot's troops are in; "the other bridge" is the opponent's bridge spot in the other lane.

## Deck
Hog Rider 4 · Musketeer 4 · Cannon 3 · Ice Golem 2 · Skeletons 1 · Ice Spirit 1 · Fireball 4 · The Log 2 ·
average 2.6 (21/8) · 4-card cycle 6 (Ice Spirit, Skeletons, Ice Golem, Log) · sim: all 8 in the engine, costs match.

The video runs Evo Skeletons, Evo Cannon and Hero Musketeer (its ability is a second building / burst, 13:38, 18:30).
The scripted opponent uses the normal forms of all eight.

## Game plan
A pure cycle deck: defend each push with the cheapest answer, then spend the saved elixir on Hog pressure (23:19). It
almost always attacks the lane opposite the bot's push (31:28) and keeps pressing once it leads (5:58). It wins on
chip damage plus Fireball/Log finishes. It's scared of being out-cycled into a heavy push while its Cannon is out of
hand, and of a single-lane stack it can't stop cheaply. It's weaker in single elixir against bait and breaks through
in double (1:38, 24:24).

## Card roles
- Hog Rider: win condition; also kites troops away from its tower (8:12, 32:37).
- Musketeer: support and air defence; sniped behind the Hog at the bridge for counterpushes (2:11, 14:10).
- Cannon: defensive building, played in the middle to pull ground pushes; placed out of reach of a Fireball that
  could also hit Musketeer (6:33). Can retarget a building-targeter for one hit's loss (16:21).
- Ice Golem: cheap tank and kiter; pulls a push away from the tower toward the other lane (5:26, 8:44).
- Skeletons: cheapest distraction and DPS on single-target melee troops; also cycle (3:16).
- Ice Spirit: freezes defenders for the Hog, tanks a hit for the Musketeer, cycles (2:11, 14:42).
- Fireball: medium spell for support clusters, and to finish towers (2:11, 21:08).
- The Log: swarm and ground-support cleaner; cycled early so another one comes back in time (25:30).

## Opening
- At 0:00 it waits; its usual first play is the Hog at a bridge (0:00). **[inferred]** Hog at (2.5, 18.5) or
  (14.5, 18.5), seeded-random, Ice Spirit at (x, 19.5) behind it, once elixir hits 7 and the bot hasn't played.
- If the bot plays first, it answers with the cheapest card that holds (rules below), then counterpushes with the Hog
  in the opposite lane (31:28).
- It never opens with Musketeer at the bridge, Cannon, Fireball or the Log. **[inferred]**

## Against our deck
All **[inferred]** except where cited.
- **Giant:** Cannon at (8.5, 24.5) as the Giant crosses the river; Musketeer at (8.5, 28.0) once the
  support is known. If the Giant is alone at the back, it ignores it and Hogs the other lane (7:39).
- **Battle Ram:** Cannon at (8.5, 24.5) plus Log for the Barbarians (31:28), or Ice Spirit + Musketeer when the Cannon
  is out of cycle (20:36). Then Ice Golem + Hog counterpush with the surviving Musketeer (20:36).
- **Mini P.E.K.K.A:** Skeletons on top of it as it reaches the princess tower, or Ice Golem kite plus the
  Cannon. Never the Musketeer alone in its path.
- **Mega Minion:** Musketeer (outranges it) or Ice Spirit + tower; never the Log.
- **Bomber:** Ice Spirit + tower, or Musketeer from outside its range; never Skeletons near it.
- **Goblin Hut:** it Hogs the lane the Hut doesn't cover, Logs the Spear Goblins, and Fireballs the Hut only when the
  Fireball also hits the tower.
- **Fireball:** kept away from: Musketeer is never placed within 2.5 tiles of the Cannon (6:33).
- **Arrows:** Skeletons and Ice Spirit die to it, so after the bot's Arrows are seen it cycles them freely.
- **Our cards that hurt it:** Mini P.E.K.K.A and Goblin Hut (hard Hog counters), Bomber (kills its Skeletons), Fireball
  on Musketeer + Cannon, and a Giant + Mini P.E.K.K.A stack in one lane that its Cannon can't hold alone.

## Cole's placements (his 5 Hog games vs the model, 2026-11-06)

Every landed play, board frame (Cole's side is high y; his bridge is y 18, his king y 30.5, the river y 15.5-17.5;
x is the same on his screen). Seeds 93405, 93651, 8051, 8237, 8427 in `data/live/deck_play.jsonl`; what the model
had just played comes from replaying the last three tick for tick. Once one of the model's princess towers is
down, Cole also plays into the pocket on its side (y 12-15), marked "pocket".

- **Hog Rider (27):** at the outer edge of the bridge, x 0-1 or 17, y 18-19 (20 of 27). The rules used x 2.5 / 14.5.
  Behind his own princess tower twice ((0, 26), (1, 32)) to make a slower push. Pocket 5 times (x 2-13, y 12-14).
  Usually within 0-4 s after the model spent: Arrows, Fireball, Mini P.E.K.K.A or Mega Minion.
- **Ice Spirit (29):** right behind the Hog at the bridge ((1-3, 18-19), about 1 s later), or in the middle at the
  river ((7-8, 18-20)) on a Bomber / Mini P.E.K.K.A. A few in a back corner to cycle ((0, 32), (16, 32), (17, 27)).
  Pocket late.
- **Ice Golem (28):** the middle near the river, (6-9, 19-22), to kite (15). At the bridge with a Hog
  ((2, 18), (14, 18)). Pocket late ((8-11, 12-13)).
- **Musketeer (29):** the middle, (7-10, 23-26) (14), the main defender against Mini P.E.K.K.A, Mega Minion and Bomber.
  Behind the king, (8-9, 33) (6), to start a push or cycle. In a lane corner, (16-17, 23-24) (2). Pocket late.
- **Skeletons (27):** the middle, (7-10, 19-26) (15), on Mini P.E.K.K.A / Bomber / Battle Ram. Lane side,
  (2-5, 18-23), with a Hog or on a lane push. Pocket late.
- **Cannon (9):** the middle, (6-9, 23-24) (7), on Battle Ram 3-4 s after it's played. In the pocket at the river
  ((10, 15), (14, 15)) after a tower fell.
- **Fireball (10):** on the Goblin Hut 1.5-4.5 s after it lands (5): (5-6, 8-9) or (11, 8-9) with the Hut at
  (8-9, 9-12). In the sim only one of those also hit the tower. The rest finish towers or the pocket.
- **The Log (7):** from the river or his bridge (y 16-21), in the lane of a Bomber walking up (4). Once early, at
  (3, 18), on the opening Bomber.

## Punish rules (checked top to bottom, every tick)
"Bot hand" = the 4 cards the bot hasn't played among its last 4 plays (cycle tracking, 7:06, 11:26). "Committed" =
elixir of the bot cards played in one lane in the last 6 s, troops still alive.

1. **(Cole)** IF the bot has committed >5 elixir in one lane, nothing of its in the other lane → Hog + Ice Spirit at
   the other bridge (Hog (x, 18.5), Ice Spirit (x, 19.5)); Cannon at (7.5, 24.5) if the push is in the left lane or
   (9.5, 24.5) if right, as the push crosses the river; if the lead troop isn't locked on the Cannon 1 s after it
   crosses, Ice Golem 1 tile in front of it (lower y), 1.5 tiles toward x 8.5, to pull it; Skeletons on the
   Mini P.E.K.K.A or the bot's highest-DPS troop as it reaches the Cannon. If opponent elixir <8, the Cannon goes first
   and the Hog when elixir reaches 5.
2. IF the bot plays a card of 4+ elixir at y ≤ 6 (behind its princess towers) in single elixir → Hog at the
   opposite bridge within 1 s; add Ice Spirit if elixir ≥5 (7:39: a "four plus elixir card in the back, go for a hog").
3. IF the bot's counted elixir ≤2, no bot troop on the opponent's half, opponent elixir ≥4 → Hog at the bridge of the
   lane with the lower-HP princess tower; add Musketeer at (x, 20.0) behind it if opponent elixir ≥8 (13:38-14:10).
4. IF Mini P.E.K.K.A is out of the bot hand and no Goblin Hut is on the board → Hog (+ Ice Golem in front if elixir
   ≥6) at once in the opposite lane of the bot's last play (3:49, 7:06). IF Mini P.E.K.K.A is in the bot hand → no lone
   Hog; Hog only with Ice Golem in front and Skeletons ready. **[inferred thresholds]**
5. IF Fireball is out of the bot hand → Musketeer goes to the bridge behind any Hog push (14:10); IF Fireball is in
   the bot hand → Musketeer only at (8.5, 28.0) or deeper, ≥3 tiles from the Cannon and its other troops.
   **[inferred]**
6. IF the bot plays Battle Ram at the bridge → Cannon at (8.5, 24.5) as it crosses + Log on the Barbarians at the
   Cannon; with no Cannon in hand, Ice Spirit on the Ram at (ram x, 18.5) + Musketeer at (8.5, 28.0) (20:36, 31:28).
7. IF the bot attacks both lanes at once → Cannon at (6.0, 24.5) or (11.0, 24.5) on the side with more elixir,
   Musketeer at (lane x, 27.0) on the other side (26:34); Fireball the side with two or more support troops (21:08).
8. IF a bot Bomber is alive within 5 tiles of the target, or Arrows is in the bot hand and the bot has ≥3 elixir →
   no Skeletons; Ice Spirit + tower or an Ice Golem kite instead. **[inferred]**
9. IF the opponent leads in crowns or total tower HP → Hog every cycle, never let elixir sit at 10 (5:58; the
   "every cycle" threshold is **[inferred]**).
10. Guard on rules 2-4 and 9 and the default Hog: IF the Hog would leave the opponent <3 elixir while the bot's Giant,
    Battle Ram or Mini P.E.K.K.A is in the bot hand and the bot has ≥5 elixir → hold the Hog and defend first (28:44,
    29:50: the fancy Hog lost the trade).
11. IF a bot princess tower is at or under the Fireball's crown-tower damage (Fireball + Log in overtime) → spell it.
    **[inferred]**

DEFAULT:
- Single elixir: defend with the cheapest card that holds, let the tower take chip damage rather than overspend
  (8:44, 23:19); Hog the lane opposite the bot's last push when elixir ≥4 and rule 10 doesn't hold it.
- Double elixir: Ice Golem + Hog or Hog + Musketeer each cycle; Fireball the tower plus a support troop when both are
  in the radius (24:24, 25:30).
- Overtime: Hog cycle in the weaker lane; Fireball + Log on the tower when that's lethal (rule 11). **[inferred]**

## Scenarios
All **[inferred]**; each still has to pass the README's check (rule 6).
- **ONE-LANE ALL-IN** (Cole's example) · 2:40 · bot 2 / opp 9 · bot Giant at (14.0, 4.0), Mini P.E.K.K.A just
  placed at (14.5, 14.5) · opp hand Hog, Ice Spirit, Cannon, Ice Golem; next Skeletons · learn:
  don't put 9 in one lane in single elixir with no Hog answer left · punished: rule 1, Hog + Ice Spirit hits the left
  tower while Cannon + Skeletons hold the right.
- **HUT PULLS THE HOG** · 2:20 · bot 5 / opp 7 · empty board · opp hand Hog, Ice Golem, Log, Musketeer; next Cannon ·
  learn: Goblin Hut at about (8.5, 9.0) on our side, pulls the Hog from both lanes and outtrades it · punished: Hut
  in a back corner (or no answer) lets Hog + Ice Golem connect for a full hit.
- **TRACK THE MINI P.E.K.K.A** · 1:50 · bot 6 / opp 6 · bot just spent Mini P.E.K.K.A on a Hog, Hog dead ·
  opp hand Ice Golem, Skeletons, Cannon, Fireball; next Hog · learn: the Hog is 1 card away; keep Goblin Hut or Mega
  Minion and 4 elixir for it, don't spend into a Giant now · punished: rule 4, Hog comes back while the bot is low.
- **FIREBALL THE PAIR** · 1:10 (double) · bot 7 / opp 4 · opp Cannon (8.5, 24.5), Musketeer (8.5, 26.5), both
  defending our Battle Ram · opp hand Hog, Ice Spirit, Log, Skeletons; next Ice Golem · learn: Fireball hitting
  Musketeer + Cannon wins 7 for 4 · punished: Fireball on the tower or Skeletons tells rule 5 Fireball is out, and a
  Hog + Musketeer bridge push follows.
- **COUNTER THE EMPTY HAND** · 0:50 (double) · bot 8 / opp 1 · opp Hog + Ice Golem + Musketeer just played in the left
  lane, Cannon 2 cards from coming back · learn: Mini P.E.K.K.A on the Hog, then Battle Ram + Mini P.E.K.K.A counterpush
  the same lane while it's low · punished: overdefending with Mini P.E.K.K.A + Bomber + Mega Minion leaves no counter,
  and the next Hog (rule 3 or the default) arrives while the bot is empty.

## Open questions for Cole
- Should the opponent's Hog timing use the bot's cycle (rule 4) from the first 4 plays, or only once all 8 bot cards
  have been seen?
- Rule 1's ">5 in one lane, last 6 s" window: right size, or should a back-of-king Giant count only once it crosses?
- Does 2.6 really never Skeletons into our Bomber, or is "Skeletons + tower on Mini P.E.K.K.A" worth it anyway?
- Fireball on Goblin Hut: worth it only with tower damage, or does a good 2.6 player just Hog the other lane?
