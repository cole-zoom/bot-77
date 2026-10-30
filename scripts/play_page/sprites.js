"use strict";
// The play page's unit art: original pixel sprites drawn in code on a 16 x 16 grid (no art from the game). Every name
// the engine can put on the board has an entry (tests/test_play_page.py checks it against the engine's registry):
//
//   "Name": [kind, label, main colour, second colour, options]
//
// kinds: H humanoid (skel: a skeleton), Q rider on an animal, O rock golem, W winged flyer, L balloon, Y barrel,
// P spirit, B building, T crown tower, C cart / machine, R battle ram, D bush.
// The label is unique, so two units never look the same; the team shows as the outline colour (blue = you, red =
// them) and the disc under the unit (drawn by app.js). Sprites are painted once per (name, team) and cached.
// Spells are areas, not sprites: SPELLS gives each its colour, pattern and label.
(function () {
const SKIN = "#f2c49b", GOB = "#74c044", BONE = "#ebe6d6", IRON = "#c7ced6", GOLD = "#f2c84b", WOOD = "#9a6a3c";
const DARK = "#2a2630", PINK = "#e79aa0";

const SPRITES = {
  // ---- crown towers
  "King Tower": ["T", "King", "#8e8a83", GOLD, {style: "king", sz: 4.2}],
  "Princess Tower": ["T", "Prn", "#8e8a83", "#e58ab6", {style: "princess", sz: 3.2}],
  // ---- troops (the engine's registry, by card id)
  "Knight": ["H", "Kn", "#a9b3bd", "#59636e", {hat: "helmet", w: "sword"}],
  "Archers": ["H", "Arc", "#4f9a5a", "#2f6a3a", {hat: "hair", hatC: "#ef6fa8", w: "bow"}],
  "Giant": ["H", "Gnt", "#a0703f", "#6b4a2a", {hat: "hair", hatC: "#e0782c", w: "none", bulk: 1, sz: 1.9}],
  "Goblins": ["H", "Gob", "#5a7a2e", "#3a4f1d", {skin: GOB, hat: "ears", w: "dagger", sz: 0.95}],
  "Mini PEKKA": ["H", "MP", "#4a5878", "#2a3248", {hat: "visor", hatC: "#3b4766", w: "bigsword"}],
  "Musketeer": ["H", "Msk", "#7a4fa0", "#4a2f66", {hat: "helmet", hatC: "#3a3a44", w: "gun"}],
  "Barbarians": ["H", "Brb", "#8a5a32", "#5a3a1e", {hat: "horns", hatC: "#9aa1a8", w: "sword"}],
  "Bomber": ["H", "Bmb", "#6e4fa0", "#463266", {skel: 1, hat: "none", w: "bomb", sz: 1.0}],
  "Valkyrie": ["H", "Val", "#c9a46a", "#7a5a32", {hat: "hair", hatC: "#ff8a2a", w: "axe"}],
  "Wizard": ["H", "Wiz", "#3b6fd0", "#24448a", {hat: "wizard", hatC: "#2f5ab0", w: "staff", wc: "#ff7a1a"}],
  "Skeleton Army": ["H", "SkA", "#55505e", "#38343f", {skel: 1, hat: "none", w: "sword", sz: 0.8}],
  "P.E.K.K.A.": ["H", "PK", "#3d3456", "#241f36", {hat: "visor", hatC: "#2f2846", w: "bigsword", bulk: 1, sz: 1.8}],
  "Prince": ["Q", "Prc", "#d8b24a", "#7a4a26", {an: "horse", hat: "crown", w: "lance"}],
  "Hog Rider": ["Q", "Hog", "#7a4a2a", PINK, {an: "hog", hat: "mohawk", hatC: "#1e1a18", w: "hammer"}],
  "Elite Barbarians": ["H", "eBrb", "#4a6aa0", "#2a3a5a", {hat: "horns", hatC: "#5a6470", w: "sword", sz: 1.3}],
  "Royal Giant": ["H", "RG", "#2e4a8a", GOLD, {hat: "crown", w: "gun", bulk: 1, sz: 1.9}],
  "Golem": ["O", "Glm", "#7d7468", "#b07ef0", {sz: 2.3}],
  "Dart Goblin": ["H", "Dart", "#3a6a5a", "#22443a", {skin: GOB, hat: "mask", hatC: "#22443a", w: "gun", sz: 0.95}],
  "Lumberjack": ["H", "LJ", "#c0392b", "#6b1f18", {hat: "beard", hatC: "#7a4a1e", w: "axe"}],
  "Bowler": ["H", "Bwl", "#6a7fb0", "#3a4a70", {skin: "#8fa6d8", hat: "bald", w: "boulder", bulk: 1, sz: 1.6}],
  "Spear Goblins": ["H", "SpG", "#8a6a2e", "#5a441d", {skin: GOB, hat: "ears", w: "spear", sz: 0.95}],
  "Skeletons": ["H", "Sk", "#55505e", "#38343f", {skel: 1, hat: "none", w: "dagger", sz: 0.8}],
  "Ice Wizard": ["H", "IWz", "#9fd8f0", "#5aa6c8", {hat: "hood", hatC: "#e8f6ff", w: "staff", wc: "#7fe8ff"}],
  "Electro Wizard": ["H", "EWz", "#2f4fa0", "#1c2f66", {hat: "hair", hatC: "#e8f0ff", w: "staff", wc: "#fff36b"}],
  "Executioner": ["H", "Exe", "#3a3340", "#1f1b24", {hat: "hood", hatC: "#141216", w: "axe", sz: 1.3}],
  "Giant Skeleton": ["H", "GSk", "#55505e", "#38343f", {skel: 1, hat: "none", w: "bomb", bulk: 1, sz: 1.8}],
  "Ice Golem": ["O", "IGo", "#9fd4ec", "#ffffff", {sz: 1.4}],
  "Minions": ["W", "Min", "#3fb3a6", "#2a7a72", {style: "minion", sz: 0.95}],
  "Minion Horde": ["W", "Hrd", "#2f8f86", "#1f5f59", {style: "minion", sz: 0.95}],
  "Mega Minion": ["W", "MM", "#2a6f78", "#1a454b", {style: "minion", helm: 1, sz: 1.3}],
  "Baby Dragon": ["W", "BD", "#5fbf4a", "#3a8a2e", {style: "dragon", sz: 1.4}],
  "Balloon": ["L", "Bln", "#c8a46a", "#8a2f2f", {sz: 1.8}],
  "Dark Prince": ["Q", "DPr", "#2a2a33", "#3a2a2a", {an: "horse", hat: "visor", hatC: "#1a1a22", w: "club"}],
  "Royal Ghost": ["H", "RGh", "#d9e2f0", "#9aa6c0", {hat: "hood", hatC: "#c0cbe0", w: "sword", ghost: 1}],
  "Mega Knight": ["H", "MK", "#4a4a55", "#2a2a33", {hat: "visor", hatC: "#3a3a44", w: "none", bulk: 1, sz: 1.8}],
  "Battle Healer": ["H", "Hlr", "#f0e6d0", "#c9a43a", {hat: "hair", hatC: "#f0d27a", w: "hammer"}],
  "Bandit": ["H", "Bdt", "#7a3fa0", "#4a2566", {hat: "mask", hatC: "#2a1a33", w: "dagger"}],
  "Berserker": ["H", "Brs", "#c0502b", "#6b2a18", {hat: "horns", hatC: "#c0502b", w: "axe"}],
  "Miner": ["H", "Mnr", "#8a6a3a", "#5a4424", {hat: "lamp", hatC: "#c9a43a", w: "pick"}],
  "Fisherman": ["H", "Fsh", "#3a6aa0", "#22446b", {hat: "hood", hatC: "#f0c84a", w: "rod"}],
  "Ronin": ["H", "Ron", "#8a2a2a", "#4a1414", {hat: "straw", hatC: "#d8b46a", w: "bigsword"}],
  "Goblin Machine": ["C", "GMc", "#8a8f99", GOB, {style: "machine", sz: 1.7}],
  "Inferno Dragon": ["W", "ID", "#a03a2a", "#ff8a2a", {style: "dragon", sz: 1.4}],
  "Electro Dragon": ["W", "ED", "#3a5ab0", "#fff36b", {style: "dragon", sz: 1.5}],
  "Night Witch": ["H", "NW", "#3a2a4a", "#1e1428", {hat: "wizard", hatC: "#1e1428", w: "staff", wc: "#b06bff"}],
  "Phoenix": ["W", "Phx", "#ff5a2a", "#ffd23a", {style: "phoenix", sz: 1.4}],
  "Sparky": ["C", "Spk", "#a07a3a", "#7fe8ff", {style: "sparky", sz: 1.7}],
  "Princess": ["H", "Pss", "#e58ab6", "#a04a7a", {hat: "tiara", hatC: "#f7d36b", w: "bow"}],
  "Hunter": ["H", "Hnt", "#7a5a3a", "#4a3624", {hat: "beard", hatC: "#c0c0c0", w: "gun"}],
  "Magic Archer": ["H", "MA", "#3a8a5a", "#1e5a36", {hat: "hood", hatC: "#2e6e48", w: "bow"}],
  "Firecracker": ["H", "FC", "#d8443a", "#8a2a22", {hat: "hair", hatC: "#ffb03a", w: "gun"}],
  "Skeleton Dragons": ["W", "SkD", BONE, "#8a8478", {style: "dragon", sz: 1.2}],
  "Goblin Demolisher": ["H", "GDm", "#7a5a2e", "#4a361c", {skin: GOB, hat: "cap", hatC: "#5a3a1e", w: "bomb", sz: 1.15}],
  "Flying Machine": ["W", "FM", WOOD, "#6b4a2a", {style: "machine", sz: 1.4}],
  "Mother Witch": ["H", "MW", "#6a3a8a", "#3a1f4a", {hat: "hood", hatC: "#4a2a66", w: "staff", wc: "#7fff6b"}],
  "Cannon Cart": ["C", "CC", "#7a5a3a", "#3a3a3a", {style: "cart", sz: 1.5}],
  "Furnace": ["C", "Frn", "#6a5a52", "#ff7a1a", {style: "furnace", sz: 1.4}],
  "Witch": ["H", "Wch", "#a05ab0", "#5a2a66", {hat: "hood", hatC: "#7a3a8a", w: "staff", wc: "#e06bff"}],
  "Ice Spirit": ["P", "ISp", "#9fe6ff", "#ffffff", {sz: 0.8}],
  "Fire Spirit": ["P", "FSp", "#ff7a2a", "#ffd23a", {sz: 0.8}],
  "Heal Spirit": ["P", "HSp", "#c8f06b", "#ffffff", {sz: 0.8}],
  "Electro Spirit": ["P", "ESp", "#6bb6ff", "#fff36b", {sz: 0.8}],
  "Guards": ["H", "Grd", "#3a3a55", "#22223a", {skel: 1, hat: "helmet", hatC: "#6a6a7a", w: "spearshield", sz: 0.9}],
  "Royal Recruits": ["H", "Rec", "#2e4a8a", "#c9a43a", {hat: "helmet", hatC: "#c9a43a", w: "spearshield"}],
  "Bats": ["W", "Bat", "#6a3a8a", "#3a1f4a", {style: "bat", sz: 0.8}],
  "Zappies": ["C", "Zpy", "#4a5a7a", "#fff36b", {style: "zappy", sz: 1.0}],
  "Three Musketeers": ["H", "3M", "#a04f7a", "#662f4a", {hat: "helmet", hatC: "#2a2a33", w: "gun"}],
  "Battle Ram": ["R", "Ram", WOOD, "#8a5a32", {sz: 1.7}],
  "Royal Hogs": ["Q", "RH", "#2e4a8a", "#f0c070", {an: "hog", hat: "crown", w: "none", sz: 1.15}],
  "Wall Breakers": ["H", "WB", "#7a5a3a", "#4a3624", {skel: 1, hat: "cap", hatC: "#7a5a3a", w: "bomb", sz: 0.85}],
  "Electro Giant": ["H", "EGi", "#3a5aa0", "#fff36b", {hat: "bald", w: "zapfist", bulk: 1, sz: 1.9}],
  "Suspicious Bush": ["D", "Bsh", "#3f8a3a", "#2a5f28", {sz: 1.3}],
  "Rune Giant": ["H", "RnG", "#6a7f9a", "#9fe0ff", {skin: "#9ab0c8", hat: "bald", w: "none", bulk: 1, sz: 1.8}],
  "Ram Rider": ["Q", "RmR", "#e58ab6", "#e8e0cc", {an: "ram", hat: "hair", hatC: "#ef6fa8", w: "none"}],
  "Goblin Giant": ["H", "GG", "#5a7a2e", "#3a4f1d", {skin: GOB, hat: "ears", w: "spear", bulk: 1, sz: 1.9}],
  "Skeleton Barrel": ["L", "SkB", "#8a8f99", WOOD, {barrel: 1, sz: 1.4}],
  "Elixir Golem": ["O", "ExG", "#c06bd8", "#ffd0ff", {sz: 1.7}],
  "Lava Hound": ["W", "LH", "#4a2f2a", "#ff7a1a", {style: "hound", sz: 2.1}],
  "Goblin Gang": ["H", "GGg", "#6a5a2e", "#433a1d", {skin: GOB, hat: "ears", w: "dagger", sz: 0.95}],
  "Rascals": ["H", "Rsc", "#4a7aa0", "#2a4a66", {hat: "cap", hatC: "#c0392b", w: "club", sz: 1.2}],
  "Mighty Miner": ["H", "MMn", "#a06a3a", "#5a3a1e", {hat: "lamp", hatC: "#ff9a2a", w: "pick", sz: 1.3}],
  "Golden Knight": ["H", "GK", GOLD, "#a07a1e", {hat: "helmet", hatC: GOLD, w: "sword"}],
  "Skeleton King": ["H", "SK", "#3a2a4a", "#1e1428", {skel: 1, hat: "crown", w: "club", bulk: 1, sz: 1.6}],
  "Archer Queen": ["H", "AQ", "#4a2a5a", "#2a1433", {hat: "crown", w: "bow"}],
  "Monk": ["H", "Mnk", "#e08a2a", "#8a4a14", {hat: "bald", w: "staff", wc: "#f0d27a"}],
  "Little Prince": ["H", "LPr", "#3a6ad0", GOLD, {hat: "crown", w: "gun", sz: 1.0}],
  "Goblinstein": ["H", "Gst", "#4a6a3a", "#2a3a1e", {skin: "#8ac05a", hat: "bald", w: "zapfist", bulk: 1, sz: 1.8}],
  "Boss Bandit": ["H", "BBd", "#5a1f7a", "#2e0f40", {hat: "mask", hatC: "#1a0f22", w: "dagger", sz: 1.3}],
  "Spirit Empress": ["H", "SpE", "#6bd8e0", "#2a8a92", {hat: "tiara", hatC: "#fff3a0", w: "staff", wc: "#bff6ff"}],
  "Hero Knight": ["H", "hKn", "#a9b3bd", GOLD, {hat: "helmet", w: "sword", hero: 1}],
  "Hero Wizard": ["H", "hWz", "#3b6fd0", GOLD, {hat: "wizard", hatC: "#2f5ab0", w: "staff", wc: "#ff7a1a", hero: 1}],
  "Hero Musketeer": ["H", "hMsk", "#7a4fa0", GOLD, {hat: "helmet", hatC: "#3a3a44", w: "gun", hero: 1}],
  "Hero Giant": ["H", "hGnt", "#a0703f", GOLD, {hat: "hair", hatC: "#e0782c", w: "none", bulk: 1, sz: 1.9, hero: 1}],
  "Hero Mini P.E.K.K.A.": ["H", "hMP", "#4a5878", GOLD, {hat: "visor", hatC: "#3b4766", w: "bigsword", hero: 1}],
  "Hero Magic Archer": ["H", "hMA", "#3a8a5a", GOLD, {hat: "hood", hatC: "#2e6e48", w: "bow", hero: 1}],
  "Hero Goblins": ["H", "hGob", "#5a7a2e", GOLD, {skin: GOB, hat: "ears", w: "dagger", sz: 0.95, hero: 1}],
  "Hero Mega Minion": ["W", "hMM", "#2a6f78", GOLD, {style: "minion", helm: 1, sz: 1.3, hero: 1}],
  "Hero Ice Golem": ["O", "hIGo", "#9fd4ec", GOLD, {sz: 1.4, hero: 1}],
  // ---- buildings
  "Cannon": ["B", "Can", "#3a3a3a", "#6f5a44", {style: "cannon"}],
  "Tesla": ["B", "Tsl", "#f2d23a", "#8a96a0", {style: "tesla"}],
  "Bomb Tower": ["B", "BT", "#8a7a6a", "#222222", {style: "bombtower"}],
  "Inferno Tower": ["B", "InT", "#6b4b6b", "#ff6a1a", {style: "inferno"}],
  "X-Bow": ["B", "XB", WOOD, "#c7ced6", {style: "xbow"}],
  "Mortar": ["B", "Mor", "#55555f", "#6f5a44", {style: "mortar"}],
  "Barbarian Hut": ["B", "BHt", "#8a5a32", "#a03a2a", {style: "hut"}],
  "Goblin Hut": ["B", "Hut", "#8a6a3a", "#4f8a2e", {style: "hut"}],
  "Tombstone": ["B", "Tmb", "#8c8c94", "#55555f", {style: "tomb"}],
  "Goblin Cage": ["B", "Cge", "#6b4a2a", GOB, {style: "cage"}],
  "Goblin Drill": ["B", "Drl", "#bbbbc4", "#7a5a3a", {style: "drill"}],
  "Elixir Collector": ["B", "Col", "#777780", "#d14bd8", {style: "collector"}],
  // ---- bodies a card spawns (negative card ids in the engine; found by name)
  "Golemite": ["O", "Gmt", "#7d7468", "#b07ef0", {sz: 1.2}],
  "Lava Pups": ["W", "Pup", "#5a3a2a", "#ff9a3a", {style: "pup", sz: 0.8}],
  "Bush Goblins": ["H", "BGb", "#3f8a3a", "#2a5f28", {skin: GOB, hat: "ears", w: "dagger", sz: 0.95}],
  "Goblin Brawler": ["H", "GBr", "#7a3a2a", "#4a2418", {skin: GOB, hat: "ears", w: "none", sz: 1.2}],
  "Elixir Golemite": ["O", "ExGm", "#c06bd8", "#ffd0ff", {sz: 1.1}],
  "Elixir Blob": ["O", "Blob", "#d88ae8", "#ffffff", {sz: 0.75}],
  "Cursed Hog": ["Q", "CHg", "#6a3a8a", "#b07ab0", {an: "hog", rider: 0, sz: 1.0}],
  "Kamikaze Goblin Demolisher": ["H", "KGD", "#a04a2a", "#5a2a18", {skin: GOB, hat: "cap", hatC: "#a04a2a", w: "bomb", sz: 1.05}],
  "Guardienne": ["H", "Gdn", "#5a7ad0", "#2a3a6a", {hat: "helmet", hatC: "#c9d4ea", w: "spearshield"}],
  "Runner": ["H", "Run", "#d0a03a", "#7a5a1e", {hat: "bandana", hatC: "#c0392b", w: "none", sz: 0.95}],
  "Souldier": ["H", "Sld", "#4a4a6a", "#2a2a3a", {skel: 1, hat: "helmet", hatC: "#7a7a8a", w: "sword", sz: 0.9}],
  "Skeleton Shadow": ["H", "SkS", "#2a2233", "#141018", {skel: 1, hat: "hood", hatC: "#2a2233", w: "dagger", sz: 0.9, ghost: 1}],
  "Lumberjack Ghost": ["H", "LJG", "#d0dcef", "#8a9ab8", {hat: "beard", hatC: "#b0bcd0", w: "axe", ghost: 1}],
  "Trusty Turret": ["B", "TT", "#6a7a8a", "#3a3a3a", {style: "turret", sz: 1.5}],
  "Decoy": ["D", "Dcy", "#8a7a5a", "#5a4a32", {sz: 1.0}],
  "Hero Barbarian Barrel": ["Y", "hBB", WOOD, GOLD, {sz: 1.1, hero: 1}],
};

// Spells as areas on the board: colour, pattern, label.
const SPELLS = {
  "Arrows": ["#c9b48a", "arrows", "Arr"], "Fireball": ["#ff7a2a", "fire", "FB"], "Zap": ["#8fd8ff", "zap", "Zap"],
  "Rocket": ["#ff5a3a", "fire", "Rkt"], "Lightning": ["#fff36b", "bolt", "Ltn"], "Poison": ["#7fd04a", "bubbles", "Psn"],
  "The Log": ["#9a6a3c", "log", "Log"], "Giant Snowball": ["#e8f6ff", "snow", "Snow"],
  "Barbarian Barrel": ["#b07a4a", "log", "BBl"], "Goblin Curse": ["#9a5ad0", "bubbles", "Crs"],
  "Earthquake": ["#a07a4a", "quake", "EQ"], "Void": ["#5a3a9a", "void", "Void"], "Vines": ["#3f9a3a", "vines", "Vns"],
  "Tornado": ["#b0c0d0", "swirl", "Trn"], "Freeze": ["#9fe6ff", "snow", "Frz"], "Rage": ["#c04ad0", "pulse", "Rage"],
  "Goblin Barrel": ["#8a6a3a", "drop", "GBl"], "Graveyard": ["#55505e", "grave", "GY"],
  "Royal Delivery": ["#3a6ad0", "drop", "RD"], "Clone": ["#6bd8ff", "pulse", "Cln"], "Mirror": ["#d0e0ff", "pulse", "Mir"],
  "Hero Barbarian Barrel": ["#b07a4a", "log", "hBB"],
};

const TEAM_OUT = ["#1f5fd1", "#d23a2f"];   // outline: blue = you (team 0), red = them
const N = 16, PAD = 1;

function shade(hex, k) {   // k < 0 darker, k > 0 lighter
  const n = parseInt(hex.slice(1), 16), ch = [n >> 16, (n >> 8) & 255, n & 255];
  return "#" + ch.map(v => Math.max(0, Math.min(255, Math.round(k < 0 ? v * (1 + k) : v + (255 - v) * k))))
    .map(v => v.toString(16).padStart(2, "0")).join("");
}
let g = null;
const P = (x, y, w, h, c) => { g.fillStyle = c; g.fillRect(x, y, w, h); };

function hat(kind, c, skin) {
  switch (kind) {
    case "helmet": P(6, 2, 5, 2, c); P(5, 3, 1, 2, c); P(11, 3, 1, 2, c); P(8, 1, 1, 1, shade(c, -.3)); break;
    case "visor": P(5, 2, 7, 6, c); P(6, 4, 5, 1, shade(c, -.45)); P(7, 4, 1, 1, "#7ff6ff"); P(9, 4, 1, 1, "#7ff6ff"); break;
    case "hood": P(5, 2, 7, 2, c); P(5, 3, 1, 5, c); P(11, 3, 1, 5, c); P(6, 1, 5, 1, c); break;
    case "crown": P(6, 1, 5, 2, GOLD); P(6, 0, 1, 1, GOLD); P(8, 0, 1, 1, GOLD); P(10, 0, 1, 1, GOLD); P(8, 1, 1, 1, "#e33"); break;
    case "tiara": P(6, 2, 5, 1, c); P(8, 1, 1, 1, c); break;
    case "horns": P(6, 2, 5, 2, c); P(4, 1, 1, 2, "#fff"); P(12, 1, 1, 2, "#fff"); P(5, 2, 1, 1, "#fff"); P(11, 2, 1, 1, "#fff"); break;
    case "wizard": P(8, 0, 2, 1, c); P(7, 1, 3, 1, c); P(6, 2, 5, 1, c); P(4, 3, 9, 1, c); break;
    case "hair": P(6, 2, 5, 2, c); P(5, 3, 1, 4, c); P(11, 3, 1, 4, c); break;
    case "mohawk": P(8, 0, 1, 3, c); P(7, 1, 3, 1, c); break;
    case "beard": P(6, 2, 5, 1, c); P(6, 6, 5, 2, c); P(7, 8, 3, 1, c); break;
    case "ears": P(4, 4, 2, 1, skin); P(11, 4, 2, 1, skin); P(4, 3, 1, 1, skin); P(12, 3, 1, 1, skin); break;
    case "straw": P(3, 2, 11, 1, c); P(6, 1, 5, 1, c); P(7, 0, 3, 1, c); break;
    case "cap": P(6, 2, 5, 2, c); P(10, 3, 3, 1, c); break;
    case "mask": P(6, 4, 5, 2, c); P(7, 5, 1, 1, "#fff"); P(9, 5, 1, 1, "#fff"); break;
    case "lamp": P(6, 2, 5, 2, c); P(5, 3, 1, 1, c); P(11, 3, 1, 1, c); P(8, 1, 1, 1, "#ffe96b"); break;
    case "bandana": P(6, 3, 5, 1, c); P(11, 3, 2, 1, c); break;
    default: break;   // bald / none
  }
}

function weapon(kind, wc, b) {
  const hx = 12 + b, lx = 4 - b;
  switch (kind) {
    case "sword": P(hx + 1, 4, 1, 7, IRON); P(hx, 10, 3, 1, WOOD); break;
    case "bigsword": P(hx + 1, 1, 2, 10, IRON); P(hx, 10, 4, 1, "#555"); break;
    case "axe": P(hx + 1, 4, 1, 9, WOOD); P(hx + 2, 3, 2, 4, IRON); break;
    case "spear": P(hx + 1, 1, 1, 12, WOOD); P(hx + 1, 0, 1, 1, IRON); P(hx, 1, 3, 1, IRON); break;
    case "lance": P(hx + 1, 0, 1, 12, GOLD); P(hx, 9, 3, 1, GOLD); break;
    case "spearshield": P(hx + 1, 1, 1, 12, WOOD); P(hx, 1, 3, 1, IRON); P(lx - 2, 8, 3, 5, wc || "#c9a63a"); P(lx - 1, 9, 1, 3, "#fff"); break;
    case "bow": P(lx - 2, 5, 1, 8, WOOD); P(lx - 1, 4, 1, 1, WOOD); P(lx - 1, 13, 1, 1, WOOD); P(lx - 1, 5, 1, 8, "#eee"); break;
    case "gun": P(hx + 1, 2, 1, 9, "#3a3a3a"); P(hx, 9, 2, 2, WOOD); break;
    case "staff": P(hx + 1, 3, 1, 10, WOOD); P(hx, 0, 3, 3, wc || "#9cf"); P(hx + 1, 1, 1, 1, "#fff"); break;
    case "hammer": P(hx + 1, 4, 1, 8, WOOD); P(hx, 2, 3, 3, "#8a8a8a"); break;
    case "pick": P(hx + 1, 3, 1, 9, WOOD); P(hx - 1, 3, 5, 1, "#aaa"); P(hx - 1, 4, 1, 1, "#aaa"); P(hx + 3, 4, 1, 1, "#aaa"); break;
    case "dagger": P(hx + 1, 8, 1, 3, IRON); break;
    case "club": P(hx + 1, 4, 2, 7, WOOD); P(hx + 1, 4, 2, 1, "#6b4a2a"); break;
    case "bomb": P(hx - 1, 9, 3, 3, "#222"); P(hx + 1, 8, 1, 1, "#f93"); break;
    case "boulder": P(hx - 1, 7, 4, 4, "#8b8f99"); P(hx, 8, 1, 1, "#b8bcc6"); break;
    case "rod": P(hx + 1, 1, 1, 11, WOOD); P(hx + 2, 1, 1, 6, "#ddd"); P(hx + 2, 7, 1, 1, "#555"); break;
    case "zapfist": P(hx - 1, 10, 3, 3, "#fff36b"); P(lx - 1, 10, 3, 3, "#fff36b"); break;
    default: break;
  }
}

function humanoid(c1, c2, o) {
  const skel = !!o.skel, skin = o.skin || (skel ? BONE : SKIN), b = o.bulk ? 1 : 0;
  const legs = skel ? BONE : shade(c2, -.15);
  P(6 - b, 13, 2, 3, legs); P(9 + b, 13, 2, 3, legs);
  if (skel) { P(6, 8, 5, 5, c1); P(7, 9, 3, 1, BONE); P(7, 11, 3, 1, BONE); P(8, 8, 1, 5, BONE); }
  else { P(5 - b, 8, 7 + 2 * b, 5, c1); P(5 - b, 11, 7 + 2 * b, 1, c2); P(8, 8, 1, 3, shade(c1, .25)); }
  const arm = skel ? BONE : c1;
  P(4 - b, 8, 1, 3, arm); P(12 + b, 8, 1, 3, arm); P(4 - b, 11, 1, 1, skin); P(12 + b, 11, 1, 1, skin);
  P(6, 3, 5, 5, skin);
  const eye = skel ? "#1a1418" : "#2a1d14";
  P(7, 5, 1, 1, eye); P(9, 5, 1, 1, eye);
  if (skel) P(7, 7, 3, 1, "#bdb6a2");
  hat(o.hat, o.hatC || c2, skin);
  weapon(o.w, o.wc, b);
}

function rider(c1, c2, o) {   // an animal facing right, a rider on its back
  const an = o.an || "hog", A = c2;
  P(2, 8, 11, 5, A); P(3, 13, 2, 3, shade(A, -.25)); P(10, 13, 2, 3, shade(A, -.25)); P(1, 8, 1, 2, shade(A, -.2));
  if (an === "horse") { P(11, 4, 3, 6, A); P(13, 5, 2, 3, A); P(12, 3, 1, 1, A); P(13, 6, 1, 1, "#222"); P(10, 4, 1, 5, shade(A, -.35)); }
  else if (an === "ram") { P(11, 6, 4, 5, A); P(10, 5, 3, 2, "#c9b38a"); P(13, 4, 2, 2, "#c9b38a"); P(13, 8, 1, 1, "#222"); P(2, 7, 9, 1, shade(A, .3)); }
  else { P(11, 7, 4, 5, A); P(14, 9, 2, 2, shade(A, -.2)); P(13, 8, 1, 1, "#222"); P(11, 6, 1, 1, shade(A, -.2)); }
  if (o.rider === 0) return;
  P(5, 5, 5, 4, c1); P(6, 1, 4, 4, o.skin || SKIN); P(7, 2, 1, 1, "#2a1d14"); P(9, 2, 1, 1, "#2a1d14");
  const save = g.getTransform();   // the rider's hat and weapon, drawn 2 px up from a standing figure's
  g.translate(-0.5, -2);
  hat(o.hat, o.hatC || c1, o.skin || SKIN);
  g.setTransform(save);
  if (o.w === "hammer") { P(11, 0, 1, 6, WOOD); P(10, 0, 3, 2, "#8a8a8a"); }
  if (o.w === "lance") { P(10, 3, 6, 1, GOLD); }
  if (o.w === "club") { P(11, 0, 2, 6, WOOD); }
}

function golem(c1, c2) {
  const d = shade(c1, -.35);
  P(4, 3, 8, 1, c1); P(3, 4, 10, 10, c1); P(2, 6, 12, 6, c1);
  P(4, 14, 3, 2, d); P(9, 14, 3, 2, d);
  P(5, 7, 2, 1, c2); P(9, 7, 2, 1, c2);
  P(6, 10, 1, 3, d); P(10, 4, 1, 2, d); P(3, 8, 2, 1, d); P(8, 12, 3, 1, d);
  P(5, 4, 2, 1, shade(c1, .3));
}

function winged(c1, c2, o) {
  const s = o.style;
  if (s === "bat") {
    P(7, 6, 3, 4, c1); P(1, 5, 6, 2, c2); P(10, 5, 6, 2, c2); P(2, 7, 2, 1, c2); P(13, 7, 2, 1, c2); P(5, 7, 1, 1, c2); P(11, 7, 1, 1, c2);
    P(7, 7, 1, 1, "#ff4a4a"); P(9, 7, 1, 1, "#ff4a4a"); P(7, 5, 1, 1, c1); P(9, 5, 1, 1, c1);
  } else if (s === "dragon" || s === "phoenix") {
    P(3, 1, 4, 6, c2); P(2, 2, 1, 3, c2); P(9, 1, 4, 5, c2); P(13, 2, 1, 2, c2);
    P(4, 6, 8, 6, c1); P(10, 3, 5, 5, c1); P(1, 9, 3, 2, c1); P(0, 8, 1, 2, c1);
    P(5, 10, 6, 2, shade(c1, .35)); P(12, 4, 1, 1, "#111"); P(14, 6, 1, 1, shade(c1, -.4));
    P(5, 12, 1, 2, shade(c1, -.3)); P(9, 12, 1, 2, shade(c1, -.3));
    if (s === "phoenix") { P(11, 1, 1, 2, "#ffd23a"); P(13, 2, 1, 1, "#ffd23a"); P(0, 7, 1, 1, "#ffd23a"); }
  } else if (s === "hound") {
    P(1, 1, 4, 4, shade(c1, -.2)); P(11, 1, 4, 4, shade(c1, -.2));
    P(2, 4, 12, 9, c1); P(3, 3, 10, 1, c1); P(4, 13, 8, 1, c1);
    P(4, 6, 2, 2, c2); P(10, 8, 2, 2, c2); P(7, 11, 2, 1, c2); P(6, 5, 1, 1, "#ffd23a"); P(9, 5, 1, 1, "#ffd23a"); P(6, 8, 4, 1, "#1a0f0a");
  } else if (s === "pup") {
    P(2, 5, 3, 3, shade(c1, -.2)); P(11, 5, 3, 3, shade(c1, -.2));
    P(5, 6, 6, 6, c1); P(6, 7, 1, 1, c2); P(9, 7, 1, 1, c2); P(6, 10, 4, 1, c2);
  } else if (s === "machine") {
    P(7, 1, 3, 1, "#ddd"); P(8, 2, 1, 3, "#888"); P(2, 7, 12, 4, c1); P(1, 6, 2, 1, c2); P(13, 6, 2, 1, c2);
    P(12, 8, 4, 2, "#444"); P(6, 4, 3, 3, GOB); P(6, 5, 1, 1, "#111"); P(3, 11, 10, 1, c2); P(0, 8, 2, 2, c2);
  } else {   // minion
    P(1, 4, 5, 4, c2); P(11, 4, 5, 4, c2); P(2, 8, 3, 1, c2); P(12, 8, 3, 1, c2);
    P(6, 5, 5, 7, c1); P(7, 12, 3, 2, shade(c1, -.25)); P(6, 4, 1, 1, c1); P(10, 4, 1, 1, c1);
    P(7, 7, 1, 1, "#fff36b"); P(9, 7, 1, 1, "#fff36b"); P(7, 9, 3, 1, shade(c1, -.4));
    if (o.helm) { P(6, 3, 5, 3, "#5a5f6a"); P(5, 4, 1, 2, "#5a5f6a"); P(11, 4, 1, 2, "#5a5f6a"); }
  }
}

function balloon(c1, c2, o) {
  P(5, 0, 6, 1, c1); P(3, 1, 10, 1, c1); P(2, 2, 12, 5, c1); P(3, 7, 10, 1, c1); P(5, 8, 6, 1, c1);
  P(7, 0, 2, 9, c2); P(4, 2, 1, 5, shade(c1, .3));
  P(5, 9, 1, 2, "#555"); P(10, 9, 1, 2, "#555");
  if (o.barrel) { P(5, 11, 6, 5, WOOD); P(5, 12, 6, 1, "#555"); P(5, 14, 6, 1, "#555"); }
  else { P(5, 11, 6, 4, WOOD); P(5, 11, 6, 1, shade(WOOD, -.3)); P(7, 14, 2, 2, "#222"); }
}

function barrel(c1, c2) {
  P(4, 3, 8, 11, c1); P(3, 5, 10, 7, c1); P(3, 5, 10, 1, "#555"); P(3, 10, 10, 1, "#555");
  P(6, 4, 1, 9, shade(c1, .25)); P(7, 1, 2, 2, c2); P(6, 7, 4, 2, c2);
}

function spirit(c1, c2) {
  P(6, 4, 4, 8, c1); P(5, 5, 6, 6, c1); P(4, 6, 8, 4, c1); P(7, 2, 2, 2, c2); P(8, 1, 1, 1, c2);
  P(6, 6, 1, 1, "#fff"); P(7, 8, 1, 1, "#222"); P(9, 8, 1, 1, "#222"); P(6, 12, 1, 1, c1); P(9, 12, 1, 1, c1);
}

function bush(c1, c2) {
  P(2, 7, 12, 7, c1); P(4, 5, 8, 2, c1); P(6, 4, 4, 1, c1); P(3, 9, 2, 2, c2); P(10, 6, 2, 2, c2); P(7, 11, 3, 2, c2);
  P(6, 8, 1, 1, "#fff"); P(9, 8, 1, 1, "#fff"); P(6, 8, 1, 1, "#fff");
}

function cart(c1, c2, o) {
  const s = o.style;
  if (s === "machine") {
    P(3, 5, 10, 8, c1); P(4, 4, 8, 1, c1); P(6, 1, 4, 3, c2); P(7, 2, 1, 1, "#111"); P(9, 2, 1, 1, "#111");
    P(13, 7, 3, 3, "#777"); P(0, 7, 3, 3, "#777"); P(4, 13, 3, 3, "#444"); P(9, 13, 3, 3, "#444"); P(5, 8, 6, 2, shade(c1, -.3));
  } else if (s === "sparky") {
    P(2, 12, 4, 4, "#333"); P(10, 12, 4, 4, "#333"); P(2, 8, 12, 4, c1); P(5, 2, 6, 6, "#f5a623");
    P(6, 3, 4, 4, c2); P(7, 4, 2, 2, "#fff"); P(4, 7, 8, 1, "#555");
  } else if (s === "furnace") {
    P(3, 3, 10, 11, c1); P(6, 8, 4, 3, c2); P(7, 9, 2, 1, "#ffe06b"); P(10, 0, 2, 3, "#555");
    P(4, 14, 2, 2, "#444"); P(10, 14, 2, 2, "#444"); P(3, 6, 10, 1, shade(c1, -.3));
  } else if (s === "zappy") {
    P(3, 12, 3, 3, "#333"); P(10, 12, 3, 3, "#333"); P(4, 6, 8, 6, c1); P(7, 1, 2, 5, "#ccc"); P(6, 0, 4, 2, c2);
    P(5, 8, 1, 1, "#7ff6ff"); P(10, 8, 1, 1, "#7ff6ff");
  } else {   // cannon cart
    P(2, 12, 4, 4, "#333"); P(10, 12, 4, 4, "#333"); P(3, 13, 2, 2, "#777"); P(11, 13, 2, 2, "#777");
    P(2, 8, 12, 4, c1); P(6, 5, 9, 3, c2); P(14, 5, 2, 3, "#111");
  }
}

function ram(c1, c2) {
  P(1, 8, 14, 4, c1); P(14, 8, 2, 4, shade(c1, .3)); P(15, 9, 1, 2, shade(c1, -.3)); P(3, 9, 9, 1, shade(c1, -.2));
  P(4, 3, 3, 4, SKIN); P(9, 3, 3, 4, SKIN); P(4, 2, 3, 1, c2); P(9, 2, 3, 1, c2);
  P(3, 1, 1, 2, "#fff"); P(7, 1, 1, 2, "#fff"); P(8, 1, 1, 2, "#fff"); P(12, 1, 1, 2, "#fff");
  P(4, 12, 2, 4, c2); P(10, 12, 2, 4, c2);
}

function building(c1, c2, o) {
  const s = o.style, base = "#6f5a44";
  switch (s) {
    case "cannon": P(2, 9, 12, 6, c2); P(2, 9, 12, 1, shade(c2, .25)); P(5, 4, 6, 6, c1); P(10, 5, 5, 4, "#222"); P(6, 5, 2, 2, "#5a5a5a"); break;
    case "tesla": P(2, 11, 12, 4, base); P(6, 3, 4, 9, c2); P(5, 1, 6, 3, c1); P(7, 0, 2, 1, "#fff"); P(4, 6, 8, 1, c1); break;
    case "bombtower": P(3, 3, 10, 12, c1); P(3, 6, 10, 1, shade(c1, -.3)); P(3, 10, 10, 1, shade(c1, -.3)); P(2, 2, 3, 2, c1); P(11, 2, 3, 2, c1); P(6, 0, 4, 4, c2); P(9, 0, 1, 1, "#f93"); break;
    case "inferno": P(3, 4, 10, 11, c1); P(5, 2, 6, 2, c1); P(6, 0, 4, 3, c2); P(7, 1, 2, 1, "#ffe06b"); P(5, 7, 6, 1, shade(c1, -.3)); P(7, 10, 2, 5, "#2a1a2a"); break;
    case "xbow": P(2, 11, 12, 4, base); P(7, 6, 2, 6, WOOD); P(2, 5, 12, 2, c1); P(1, 6, 2, 2, c1); P(13, 6, 2, 2, c1); P(7, 1, 2, 9, c2); P(6, 1, 4, 1, c2); break;
    case "mortar": P(2, 11, 12, 4, c2); P(4, 7, 8, 5, c1); P(6, 2, 5, 6, c1); P(6, 2, 5, 1, "#111"); P(5, 9, 6, 1, shade(c1, .25)); break;
    case "hut":
      P(2, 8, 12, 7, c1); P(1, 7, 14, 1, c2); P(2, 6, 12, 1, c2); P(3, 5, 10, 1, c2); P(4, 4, 8, 1, c2); P(5, 3, 6, 1, c2); P(6, 2, 4, 1, c2);
      P(7, 10, 3, 5, "#3a2a1a"); P(3, 10, 2, 2, "#ffe0a0"); P(11, 10, 2, 2, "#ffe0a0"); break;
    case "tomb": P(4, 4, 8, 11, c1); P(5, 3, 6, 1, c1); P(7, 5, 2, 7, c2); P(5, 7, 6, 2, c2); P(2, 14, 12, 2, "#5a4a3a"); break;
    case "cage": P(2, 3, 12, 12, "#3a2a1a"); P(3, 4, 10, 10, shade(c1, -.4)); P(6, 6, 4, 5, c2); P(7, 7, 1, 1, "#111"); P(9, 7, 1, 1, "#111");
      for (let x = 3; x < 14; x += 3) P(x, 3, 1, 12, c1); P(2, 2, 12, 1, c1); break;
    case "drill": P(2, 11, 12, 5, c2); P(3, 10, 10, 1, c2); P(6, 7, 4, 4, c1); P(7, 4, 2, 3, c1); P(7, 2, 1, 2, c1); P(6, 8, 4, 1, shade(c1, -.3)); P(7, 5, 2, 1, shade(c1, -.3)); break;
    case "collector": P(3, 4, 10, 10, c1); P(5, 6, 6, 6, c2); P(6, 7, 1, 2, "#fff"); P(6, 1, 4, 3, "#555"); P(2, 14, 12, 2, base); break;
    case "turret": P(3, 11, 10, 4, "#555"); P(5, 6, 6, 6, c1); P(10, 7, 5, 2, c2); P(7, 4, 2, 2, c1); break;
    default: P(2, 2, 12, 12, c1);
  }
}

function tower(c1, c2, o, team) {
  const stone = c1, dk = shade(c1, -.3), flag = TEAM_OUT[team];
  if (o.style === "king") {
    P(0, 5, 16, 11, stone); P(0, 3, 3, 2, stone); P(5, 3, 2, 2, stone); P(9, 3, 2, 2, stone); P(13, 3, 3, 2, stone);
    P(0, 9, 16, 1, dk); P(0, 13, 16, 1, dk); P(6, 11, 4, 5, "#3a2a1a");
    P(4, 0, 8, 3, GOLD); P(5, 6, 6, 3, flag); P(7, 1, 2, 1, "#e33");
  } else {
    P(1, 5, 14, 11, stone); P(1, 3, 3, 2, stone); P(6, 3, 4, 2, stone); P(12, 3, 3, 2, stone);
    P(1, 10, 14, 1, dk); P(6, 12, 4, 4, "#3a2a1a"); P(4, 6, 8, 3, flag);
    P(6, 0, 4, 3, SKIN); P(6, 0, 4, 1, c2); P(5, 1, 1, 3, c2); P(10, 1, 1, 3, c2);
  }
}

function paint(entry, team) {
  const [kind, , c1, c2, o] = entry;
  switch (kind) {
    case "H": humanoid(c1, c2, o); break;
    case "Q": rider(c1, c2, o); break;
    case "O": golem(c1, c2); break;
    case "W": winged(c1, c2, o); break;
    case "L": balloon(c1, c2, o); break;
    case "Y": barrel(c1, c2); break;
    case "P": spirit(c1, c2); break;
    case "B": building(c1, c2, o); break;
    case "T": tower(c1, c2, o, team); break;
    case "C": cart(c1, c2, o); break;
    case "R": ram(c1, c2); break;
    case "D": bush(c1, c2); break;
  }
  if (o.hero) { P(0, 0, 3, 1, GOLD); P(1, 0, 1, 3, GOLD); P(0, 1, 3, 1, "#fff6b0"); }
}

// A name the table doesn't know still gets its own look: a coloured disc with its initials (never happens for the
// engine's current registry; a test checks that).
function fallback(name) {
  let h = 0; for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const c = "#" + ((h & 0x7f7f7f) + 0x404040).toString(16).padStart(6, "0");
  const lab = name.split(/\s+/).map(w => w[0]).join("").slice(0, 3) || "?";
  return ["P", lab, c, "#ffffff", {sz: 1.0}];
}
const entryOf = name => SPRITES[name] || fallback(name || "?");

const cache = new Map();
function canvasOf(name, team, white) {
  const key = name + "|" + team + (white ? "|w" : "");
  if (cache.has(key)) return cache.get(key);
  const e = entryOf(name), S = N + 2 * PAD;
  const body = document.createElement("canvas"); body.width = body.height = S;
  g = body.getContext("2d"); g.translate(PAD, PAD); paint(e, team); g = null;
  const out = document.createElement("canvas"); out.width = out.height = S;
  const o = out.getContext("2d");
  const sil = document.createElement("canvas"); sil.width = sil.height = S;   // the outline: the shape in the team colour, 1 px out
  const s = sil.getContext("2d"); s.drawImage(body, 0, 0); s.globalCompositeOperation = "source-in";
  s.fillStyle = TEAM_OUT[team] || "#111"; s.fillRect(0, 0, S, S);
  for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) o.drawImage(sil, dx, dy);
  o.drawImage(body, 0, 0);
  if (white) { o.globalCompositeOperation = "source-atop"; o.fillStyle = "rgba(255,255,255,.75)"; o.fillRect(0, 0, S, S); }
  cache.set(key, out);
  return out;
}

window.Sprites = {
  SPRITES, SPELLS, TEAM_OUT,
  has: name => name in SPRITES,
  label: name => entryOf(name)[1],
  size: name => entryOf(name)[4].sz || ({H: 1.2, Q: 1.4, O: 2.0, W: 1.2, L: 1.6, Y: 1.1, P: 0.8, B: 2.2, T: 3.2, C: 1.5, R: 1.6, D: 1.3})[entryOf(name)[0]] || 1.2,
  ghostly: name => !!entryOf(name)[4].ghost,
  // draw `name` for `team` centred at (px, py) in a box `size` device px wide; flash = hit flash
  draw(ctx, name, team, px, py, size, flash) {
    const c = canvasOf(name, team, flash);
    const prev = ctx.imageSmoothingEnabled; ctx.imageSmoothingEnabled = false;
    ctx.drawImage(c, Math.round(px - size / 2), Math.round(py - size / 2), Math.round(size), Math.round(size));
    ctx.imageSmoothingEnabled = prev;
  },
  // a spell's pattern inside a disc of radius (rx, ry) device px; t = seconds since it appeared
  spell(ctx, name, px, py, rx, ry, t, team) {
    // the frame's timestamp can be a few ms before the moment the spell was first seen: a negative age made the
    // rings' radius negative and ellipse() threw, which stopped the page drawing (Cole, 2026-10-30)
    t = Math.max(0, t);
    const [col, style] = SPELLS[name] || ["#cccccc", "pulse"];
    ctx.save();
    ctx.beginPath(); ctx.ellipse(px, py, rx, ry, 0, 0, 7);
    ctx.globalAlpha = .26; ctx.fillStyle = col; ctx.fill();
    ctx.globalAlpha = .9; ctx.lineWidth = Math.max(2, rx * .05); ctx.strokeStyle = TEAM_OUT[team] || col;
    ctx.setLineDash([rx * .18, rx * .1]); ctx.lineDashOffset = -t * rx * .6; ctx.stroke(); ctx.setLineDash([]);
    ctx.clip();
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = Math.max(1.5, rx * .05);
    const R = Math.max(rx, ry);
    const seeded = k => { const s = Math.sin(k * 12.9898) * 43758.5453; return s - Math.floor(s); };
    switch (style) {
      case "fire": case "pulse": for (let k = 0; k < 3; k++) { const r = ((t * 1.4 + k / 3) % 1) * R; ctx.globalAlpha = .7 * (1 - r / R); ctx.beginPath(); ctx.ellipse(px, py, r, r * ry / rx, 0, 0, 7); ctx.stroke(); } break;
      case "arrows": for (let k = 0; k < 14; k++) { const ax = px + (seeded(k) * 2 - 1) * rx, ay = py + (seeded(k + 50) * 2 - 1) * ry; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax - rx * .08, ay - ry * .2); ctx.stroke(); } break;
      case "zap": case "bolt": for (let k = 0; k < (style === "bolt" ? 3 : 5); k++) { let x = px + (seeded(k + Math.floor(t * 8)) * 2 - 1) * rx * .7, y = py - ry; ctx.beginPath(); ctx.moveTo(x, y); for (let j = 0; j < 5; j++) { x += (seeded(k * 7 + j + Math.floor(t * 8)) - .5) * rx * .5; y += ry * .4; ctx.lineTo(x, y); } ctx.stroke(); } break;
      case "bubbles": for (let k = 0; k < 10; k++) { const bx = px + (seeded(k) * 2 - 1) * rx * .8, by = py + ((seeded(k + 9) * 2 - 1) * ry - ((t * .5 + seeded(k + 3)) % 1) * ry * .6); ctx.beginPath(); ctx.arc(bx, by, R * .07, 0, 7); ctx.stroke(); } break;
      case "snow": for (let k = 0; k < 12; k++) { const bx = px + (seeded(k) * 2 - 1) * rx, by = py + (seeded(k + 4) * 2 - 1) * ry; ctx.fillRect(bx - 2, by - 2, 4, 4); } break;
      case "log": ctx.globalAlpha = .8; ctx.fillRect(px - rx * .9, py - ry * .2, rx * 1.8, ry * .4); break;
      case "quake": for (let k = 0; k < 6; k++) { const a = k / 6 * 6.28 + .3; ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + Math.cos(a) * rx * .5, py + Math.sin(a) * ry * .4); ctx.lineTo(px + Math.cos(a + .3) * rx, py + Math.sin(a + .3) * ry); ctx.stroke(); } break;
      case "void": ctx.globalAlpha = .45; ctx.beginPath(); ctx.ellipse(px, py, rx * .55, ry * .55, 0, 0, 7); ctx.fill(); break;
      case "vines": for (let k = 0; k < 5; k++) { ctx.beginPath(); ctx.arc(px + (seeded(k) - .5) * rx, py + (seeded(k + 2) - .5) * ry, R * .35, 0, 3); ctx.stroke(); } break;
      case "swirl": for (let k = 0; k < 3; k++) { ctx.beginPath(); for (let a = 0; a < 9; a += .3) { const r = a / 9 * R; ctx.lineTo(px + Math.cos(a + t * 4 + k * 2.1) * r, py + Math.sin(a + t * 4 + k * 2.1) * r * ry / rx); } ctx.stroke(); } break;
      case "drop": ctx.globalAlpha = .8; ctx.fillRect(px - rx * .3, py - ry * .4, rx * .6, ry * .8); break;
      case "grave": for (let k = 0; k < 6; k++) { const bx = px + (seeded(k) * 2 - 1) * rx * .7, by = py + (seeded(k + 5) * 2 - 1) * ry * .7; ctx.fillRect(bx - R * .05, by - R * .08, R * .1, R * .16); } break;
    }
    ctx.restore();
  },
};
})();
