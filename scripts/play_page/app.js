"use strict";
// The play page (scripts/situation_play.py serves it). One game loop per game: a generation number cancels the old
// one, so a late response from a previous game can't overwrite a new one. Each loop turn posts one engine tick
// (/step), with the play you queued if any; the board is redrawn every animation frame, units sliding between the
// last two ticks.
//
// What reaches the engine is unchanged from the first version of this page (tests/test_play_page.py checks it): a play
// is {slot, x, y} with x, y the integer centre of the clicked tile, sent on the next tick; no play is {}. The board
// is the engine's: B.w x B.h cells, cell i covers [i - 0.5, i + 0.5], so engine point (x, y) is at canvas
// ((x + 0.5) * sx, H - (y + 0.5) * sy). River, bridges and the dead back-row corners come from the server
// (bot77.sim.placement.board_geometry); a card's legal tiles are the engine's is_valid_placement at every cell
// centre, sent with every tick. The tick length and board size come from the engine too (B.tick_s, B.w, B.h).
(function () {
const $ = id => document.getElementById(id);
const KEY = "bot77-play:";
const load = (k, d) => { try { const v = localStorage.getItem(KEY + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } };
const save = (k, v) => { try { localStorage.setItem(KEY + k, JSON.stringify(v)); } catch (e) { /* not remembered */ } };
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sleep = ms => new Promise(r => setTimeout(r, ms));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  return r.json();
}

// ---------------------------------------------------------------- state
let B = null, CAT = null;               // board geometry + clock (engine), card catalogue (engine)
const byName = new Map();               // unit name -> catalogue entry (spawned bodies have no card id of their own)
let S = null, Sp = null, tS = 0;        // latest state, the one before it, when the latest arrived
let prevById = new Map();
let sel = null, pending = null, paused = false, lastPlay = null;
let lastSeed = null, lastTpl = null, lastThreat = "", sitName = "";
let gen = 0, hov = null, hovUnit = null, nPlays = 0, initialIds = new Set(), oppSeen = new Set();
let leakAt = 0, scoring = null;
const SPEEDS = [0.25, 0.5, 1, 2];
let speed = SPEEDS.includes(load("speed", 1)) ? load("speed", 1) : 1;
const opt = {labels: load("labels", true), coords: load("coords", true), dmg: load("dmg", true)};
const flashUntil = new Map(), deployMax = new Map(), bornAt = new Map(), trail = new Map();
let floats = [], puffs = [];
const tickMs = () => (B ? B.tick_s : 0.1) * 1000;
const NX = () => (B ? B.w : 18), NY = () => (B ? B.h : 34);

// ---------------------------------------------------------------- canvas
const cv = $("board"), ctx = cv.getContext("2d");
let W = 0, H = 0, sx = 1, sy = 1, dpr = 1, C = {}, terrain = null;
const X = x => (x + 0.5) * sx, Y = y => H - (y + 0.5) * sy;
function readColors() {
  for (const n of ["grass", "grass2", "path", "river", "bridge", "dead", "grid", "gridlab", "ours", "theirs", "accent", "legal", "illegal", "hover", "elx"]) C[n] = css("--" + n);
  C.team = [C.ours, C.theirs];
}
function resize() {
  dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  if (!r.width) return;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  W = cv.width; H = cv.height; sx = W / NX(); sy = H / NY();
  buildTerrain();
}
const deadAt = (x, y) => !!B && B.dead[y * NX() + x] === "1";
const legalAt = (c, x, y) => !!c && !!c.legal && c.legal[y * NX() + x] === "1";
function cellRect(x, y) { return [x * sx, H - (y + 1) * sy, sx, sy]; }

function buildTerrain() {   // tiles, river, bridges, grid and coordinates: redrawn only on resize / theme / toggle
  if (!W || !B) return;
  terrain = document.createElement("canvas"); terrain.width = W; terrain.height = H;
  const g = terrain.getContext("2d");
  for (let y = 0; y < NY(); y++) for (let x = 0; x < NX(); x++) {
    g.fillStyle = deadAt(x, y) ? C.dead : (x + y) % 2 ? C.grass : C.grass2; g.fillRect(...cellRect(x, y));
  }
  // The lane paths (the arena's lighter dirt track, bridge to tower to tower): one tile wide on each bridge's centre.
  g.fillStyle = C.path;
  for (const bx of B.bridge_x || []) g.fillRect((bx - 0.1) * sx, sy, 1.2 * sx, H - 2 * sy);  // engine x -> px is (x + 0.5) * sx
  const [r0, r1] = B.river;
  g.fillStyle = C.river; g.fillRect(0, H - (r1 + 0.5) * sy, W, (r1 - r0) * sy);
  g.fillStyle = C.bridge;
  for (const bx of B.bridge_cols) g.fillRect(bx * sx, H - (r1 + 0.5) * sy, sx, (r1 - r0) * sy);
  const off = Math.round(dpr) % 2 ? 0.5 : 0;   // every tile outlined
  g.strokeStyle = C.grid; g.lineWidth = Math.max(1, Math.round(dpr)); g.beginPath();
  for (let i = 0; i <= NX(); i++) { const px = Math.round(i * sx) + off; g.moveTo(px, 0); g.lineTo(px, H); }
  for (let j = 0; j <= NY(); j++) { const py = Math.round(H - j * sy) + off; g.moveTo(0, py); g.lineTo(W, py); }
  g.stroke();
  g.lineWidth = 2 * Math.max(1, Math.round(dpr)); g.beginPath();   // the lane split and the river edges
  g.moveTo(Math.round(NX() / 2 * sx), 0); g.lineTo(Math.round(NX() / 2 * sx), H);
  for (const yy of B.river) { const py = Math.round(H - (yy + 0.5) * sy); g.moveTo(0, py); g.lineTo(W, py); }
  g.stroke();
  if (opt.coords) {
    g.fillStyle = C.gridlab; g.font = `600 ${clamp(sx / dpr * .42, 8, 11) * dpr}px ui-monospace,monospace`;
    g.textAlign = "center"; g.textBaseline = "bottom";
    for (let x = 0; x < NX(); x++) g.fillText(x, X(x), H - 1 * dpr);
    g.textAlign = "left"; g.textBaseline = "top";
    for (let y = 1; y < NY(); y++) g.fillText(y, 2 * dpr, H - (y + 1) * sy + 1 * dpr);
  }
}

// ---------------------------------------------------------------- what a unit is
const cardOf = u => (u.card_id >= 0 && CAT && CAT.cards[u.card_id]) || byName.get(u.name) || null;
const isProj = u => !u.tower && !u.name;
const isSpell = u => !u.tower && !!u.name && u.card_id >= 0 && !!CAT && !!CAT.cards[u.card_id] && CAT.cards[u.card_id].spell;
function reachOf(u) {
  if (u.tower) { const r = CAT && CAT.tower_range[u.name]; return r ? {range: r, sight: r, air: true} : null; }
  const c = cardOf(u);
  return c && !c.spell && c.range > 0 ? {range: c.range, sight: c.sight, air: c.air} : null;
}
function bodyR(u) {   // engine body radii: towers Tower::getCollisionRadius, buildings Building::COLLISION_RADIUS
  if (u.tower) return B && Math.abs(u.x - B.center_x) < 1 ? B.king_radius : (B ? B.princess_radius : 1.5);
  return u.building ? 1.0 : u.flying ? 0.45 : 0.55;
}
function posOf(u, a) {
  const p = prevById.get(u.id);
  return p ? [p.x + (u.x - p.x) * a, p.y + (u.y - p.y) * a] : [u.x, u.y];
}
function unitAt(h, a) {   // the unit under the pointer (engine coordinates), nearest first
  if (!S || !h) return null;
  let best = null, bd = 1e9;
  for (const u of S.units) {
    if (isProj(u)) continue;
    const [x, y] = posOf(u, a), lift = u.flying ? 0.45 : 0;
    const d = Math.hypot(x - h.ex, y + lift - h.ey), r = isSpell(u) ? 0.6 : Math.max(bodyR(u), Sprites.size(u.name) * 0.4);
    if (d <= r + 0.1 && d < bd) { bd = d; best = u; }
  }
  return best;
}

// ---------------------------------------------------------------- drawing
function ell(x, y, rx, ry) { ctx.beginPath(); ctx.ellipse(X(x), Y(y), rx * sx, (ry ?? rx) * sy, 0, 0, 7); }
function ring(x, y, r, col, dashed, alpha = 0.9) {
  ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = col; ctx.lineWidth = 2 * dpr;
  if (dashed) ctx.setLineDash([6 * dpr, 5 * dpr]);
  ell(x, y, r); ctx.stroke();
  ctx.globalAlpha = alpha * 0.08; ctx.fillStyle = col; if (!dashed) ctx.fill();
  ctx.restore();
}
function boxLabel(text, px, py, fg, bg, size = 12) {
  ctx.font = `600 ${size * dpr}px ui-monospace,monospace`; ctx.textBaseline = "middle"; ctx.textAlign = "left";
  const w = ctx.measureText(text).width + 10 * dpr, h = (size + 6) * dpr;
  px = clamp(px, 2, W - w - 2); py = clamp(py, h / 2 + 2, H - h / 2 - 2);
  ctx.fillStyle = bg; ctx.fillRect(px, py - h / 2, w, h); ctx.fillStyle = fg; ctx.fillText(text, px + 5 * dpr, py);
}
function ghost(c, x, y, ok, alpha) {   // a card's footprint (troop 0.4, building 1.0) per body, or a spell's radius
  const col = ok ? C.accent : C.theirs;
  ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2 * dpr;
  if (c.spell) {
    ell(x, y, c.spell_radius); ctx.globalAlpha = .2 * alpha; ctx.fill(); ctx.globalAlpha = alpha; ctx.stroke();
    ell(x, y, .15); ctx.fill();
  } else {
    for (const [ox, oy] of (c.offsets && c.offsets.length ? c.offsets : [[0, 0]])) {
      ell(x + ox, y + oy, c.placement_radius); ctx.globalAlpha = .45 * alpha; ctx.fill(); ctx.globalAlpha = alpha; ctx.stroke();
    }
    ctx.globalAlpha = .55 * alpha;
    Sprites.draw(ctx, c.name, 0, X(x), Y(y) - Sprites.size(c.name) * sy * .25, Sprites.size(c.name) * sx);
  }
  ctx.restore();
  const k = CAT && CAT.cards[c.id];
  if (k && !k.spell && k.range > 0 && ok) { ring(x, y, k.range, col, false, .8); ring(x, y, k.sight, col, true, .6); }
}

function drawUnit(u, a, now) {
  const [x, y] = posOf(u, a), size = Sprites.size(u.name) * sx, lift = u.flying ? 0.45 * sy : 0;
  const px = X(x), py = Y(y), team = u.team;
  const bob = !u.tower && !u.building && prevById.has(u.id) && (Math.abs(u.x - prevById.get(u.id).x) + Math.abs(u.y - prevById.get(u.id).y) > 1e-3)
    ? Math.sin(now / 90 + u.id) * 0.04 * sy : 0;
  ctx.save();
  if (u.flying) {   // a shadow on the ground, the body up in the air
    ctx.fillStyle = "rgba(0,0,0,.28)"; ctx.beginPath(); ctx.ellipse(px, py + size * .12, size * .32, size * .12, 0, 0, 7); ctx.fill();
  } else if (!u.tower) {   // the team disc under a ground unit
    ctx.fillStyle = C.team[team]; ctx.globalAlpha = .35; ell(x, y, bodyR(u) * .95, bodyR(u) * .55); ctx.fill();
    ctx.globalAlpha = .9; ctx.strokeStyle = C.team[team]; ctx.lineWidth = 1.5 * dpr; ctx.stroke(); ctx.globalAlpha = 1;
  }
  if (u.deploy > 0 || !u.targetable || Sprites.ghostly(u.name)) ctx.globalAlpha = .55;
  const flash = (flashUntil.get(u.id) || 0) > now;
  const sy0 = u.tower ? py : py - size * .22;
  Sprites.draw(ctx, u.name, team, px, sy0 - lift + bob, size, flash);
  ctx.globalAlpha = 1;
  if (u.deploy > 0) {   // deploy time left, as a closing arc
    const m = deployMax.get(u.id) || u.deploy;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5 * dpr; ctx.beginPath();
    ctx.arc(px, sy0 - lift, size * .45, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * (u.deploy / m)); ctx.stroke();
  }
  if (u === hovUnit) { ctx.strokeStyle = C.hover; ctx.lineWidth = 2 * dpr; ctx.strokeRect(px - size / 2, sy0 - lift - size / 2, size, size); }
  ctx.restore();
  return [px, sy0 - lift, size];
}

function drawBars(u, px, cy, size) {
  const full = u.max_hp || maxSeen.get(u.id) || u.hp || 1, f = clamp(u.hp / full, 0, 1);
  const w = u.tower ? size * .7 : clamp(size * .8, .8 * sx, 2.4 * sx), h = (u.tower ? 6 : 4) * dpr;
  const top = cy - size / 2 - h - 2 * dpr;
  ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(px - w / 2 - dpr, top - dpr, w + 2 * dpr, h + 2 * dpr);
  ctx.fillStyle = f < .25 ? "#ffcf4a" : C.team[u.team]; ctx.fillRect(px - w / 2, top, w * f, h);
  if (u.tower) {
    ctx.font = `800 ${12 * dpr}px ui-monospace,monospace`; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.strokeText(Math.round(u.hp), px, top - 2 * dpr);
    ctx.fillStyle = "#fff"; ctx.fillText(Math.round(u.hp), px, top - 2 * dpr);
  } else if (opt.labels && sx / dpr >= 13) {
    const lab = Sprites.label(u.name);
    ctx.font = `700 ${clamp(sx / dpr * .38, 8, 11) * dpr}px ui-monospace,monospace`; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const ly = cy + size * .42, tw = ctx.measureText(lab).width + 6 * dpr, th = clamp(sx / dpr * .38, 8, 11) * dpr + 3 * dpr;
    ctx.globalAlpha = .85; ctx.fillStyle = u.team === 0 ? "#173a7a" : "#7a1f18"; ctx.fillRect(px - tw / 2, ly - dpr, tw, th);
    ctx.globalAlpha = 1; ctx.fillStyle = "#fff"; ctx.fillText(lab, px, ly);
  }
}
const maxSeen = new Map();

function frame(now) {
  draw(now);
  requestAnimationFrame(frame);
}
function draw(now) {
  if (!W) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (terrain) ctx.drawImage(terrain, 0, 0);
  if (!B) return;
  const a = paused || !S ? 1 : clamp((now - tS) / (tickMs() / speed), 0, 1);
  const c = sel !== null && S && S.hand[sel] ? S.hand[sel] : null;
  if (c) for (let y = 0; y < NY(); y++) for (let x = 0; x < NX(); x++) {   // the selected card's legal tiles, this tick
    if (deadAt(x, y)) continue;
    ctx.fillStyle = legalAt(c, x, y) ? C.legal : C.illegal; ctx.fillRect(...cellRect(x, y));
  }
  if (hov) {
    const ok = !c || legalAt(c, hov.x, hov.y);
    ctx.fillStyle = "rgba(255,255,255,.22)"; ctx.fillRect(...cellRect(hov.x, hov.y));
    ctx.strokeStyle = ok ? C.hover : C.theirs; ctx.lineWidth = 2.5 * dpr; ctx.strokeRect(...cellRect(hov.x, hov.y));
  }
  if (lastPlay) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr; ctx.setLineDash([6 * dpr, 5 * dpr]); ctx.strokeRect(...cellRect(lastPlay.x, lastPlay.y)); ctx.setLineDash([]); }
  if (S) {
    hovUnit = unitAt(hov, a);
    const spells = [], ground = [], air = [], proj = [];
    for (const u of S.units) (isProj(u) ? proj : isSpell(u) ? spells : u.flying ? air : ground).push(u);
    for (const u of spells) {
      const [x, y] = posOf(u, a), k = CAT.cards[u.card_id], r = k.spell_radius || 1;
      Sprites.spell(ctx, u.name, X(x), Y(y), r * sx, r * sy, (now - (bornAt.get(u.id) || now)) / 1000, u.team);
      boxLabel((Sprites.SPELLS[u.name] || [, , u.name])[2], X(x) - 14 * dpr, Y(y), "#fff", u.team === 0 ? "#173a7acc" : "#7a1f18cc", 10);
    }
    const byDepth = (p, q) => q.y - p.y;   // farther up the board first
    const drawn = [];
    for (const u of ground.sort(byDepth)) drawn.push([u, drawUnit(u, a, now)]);
    for (const u of air.sort(byDepth)) drawn.push([u, drawUnit(u, a, now)]);
    for (const u of proj) {   // projectiles in flight, with a short trail
      const [x, y] = posOf(u, a), t = trail.get(u.id);
      if (t) { ctx.strokeStyle = C.team[u.team]; ctx.globalAlpha = .6; ctx.lineWidth = 2 * dpr; ctx.beginPath(); ctx.moveTo(X(t[0]), Y(t[1])); ctx.lineTo(X(x), Y(y)); ctx.stroke(); ctx.globalAlpha = 1; }
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(X(x), Y(y), .17 * sx, 0, 7); ctx.fill();
      ctx.fillStyle = C.team[u.team]; ctx.beginPath(); ctx.arc(X(x), Y(y), .11 * sx, 0, 7); ctx.fill();
    }
    for (const [u, [px, cy, size]] of drawn) drawBars(u, px, cy, size);
    if (hovUnit && !isSpell(hovUnit)) {   // its reach, and what it's locked onto
      const [x, y] = posOf(hovUnit, a), r = reachOf(hovUnit), col = C.team[hovUnit.team];
      if (r) { ring(x, y, r.range, col, false); if (r.sight > r.range + 0.05) ring(x, y, r.sight, col, true, .7); }
      const t = S.units.find(v => v.id === hovUnit.target);
      if (t) {
        const [tx, ty] = posOf(t, a);
        ctx.save(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * dpr; ctx.setLineDash([4 * dpr, 4 * dpr]);
        ctx.beginPath(); ctx.moveTo(X(x), Y(y)); ctx.lineTo(X(tx), Y(ty)); ctx.stroke(); ctx.restore();
      }
    }
  }
  // effects: puffs where something died or a shot landed, floating damage
  puffs = puffs.filter(p => now - p.t0 < p.life);
  for (const p of puffs) {
    const k = (now - p.t0) / p.life;
    ctx.save(); ctx.globalAlpha = 1 - k; ctx.strokeStyle = p.col; ctx.lineWidth = (p.big ? 4 : 2) * dpr;
    ell(p.x, p.y, p.r * (0.4 + k)); ctx.stroke();
    if (p.big) { ctx.fillStyle = p.col; ctx.globalAlpha = (1 - k) * .25; ctx.fill(); }
    ctx.restore();
  }
  floats = floats.filter(f => now - f.t0 < 850);
  if (opt.dmg) for (const f of floats) {
    const k = (now - f.t0) / 850;
    ctx.save(); ctx.globalAlpha = 1 - k * k;
    ctx.font = `800 ${(f.big ? 14 : 11) * dpr}px ui-monospace,monospace`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const px = X(f.x) + f.dx * sx, py = Y(f.y) - (0.6 + k * 0.9) * sy;
    ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(0,0,0,.8)"; ctx.strokeText(f.text, px, py);
    ctx.fillStyle = f.col; ctx.fillText(f.text, px, py); ctx.restore();
  }
  if (pending) {
    ghost(pending.card, pending.x, pending.y, true, .9);
    boxLabel(paused ? "queued · Z takes back" : "queued", X(pending.x) + sx, Y(pending.y) - sy, "#1d1606", C.accent, 10);
  }
  if (hov && c) ghost(c, hov.x, hov.y, legalAt(c, hov.x, hov.y), 1);
  if (hov && hovUnit) {
    const u = hovUnit, full = u.max_hp || maxSeen.get(u.id) || 0;
    boxLabel(isSpell(u) ? u.name : `${u.name} ${Math.round(u.hp).toLocaleString()} / ${full.toLocaleString()}`,
      X(posOf(u, a)[0]) - 60 * dpr, Y(posOf(u, a)[1]) - Sprites.size(u.name) * sy * .75 - 14 * dpr, "#fff", u.team === 0 ? "#1d3f80e6" : "#7a1f18e6");
  } else if (hov) boxLabel(`(${hov.x}, ${hov.y})`, (hov.x + 1) * sx + 4 * dpr, H - (hov.y + 1) * sy, "#fff", "#000b");
}

// ---------------------------------------------------------------- events from one tick to the next
const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function clockText(st) {
  if (!st || !B) return "";
  const left = B.regulation_end - st.clock;
  return left > 0 ? mmss(Math.ceil(left)) : "OT+" + mmss(-left);
}
function logEvent(who, text, cls = "") {
  const li = document.createElement("li");
  li.className = (who === "You" ? "you" : who === "Them" ? "them" : "sys") + (cls ? " " + cls : "");
  li.innerHTML = `<span class="muted">${clockText(S) || "–"}</span><span><span class="who">${who === "·" ? "" : who + " "}</span>${text}</span>`;
  const ol = $("log"); ol.prepend(li);
  while (ol.children.length > 120) ol.lastChild.remove();
}
const towerName = u => (Math.abs(u.x - B.center_x) < 1 ? "King" : u.x < B.center_x ? "left Princess" : "right Princess") + " Tower";

function onTick(prev, cur, now) {
  const before = new Map(prev.units.map(u => [u.id, u])), after = new Set(cur.units.map(u => u.id));
  let melee = 0, impact = 0, towerHit = 0, deaths = 0, spells = 0, theirDeploy = 0;
  for (const u of cur.units) {
    if (!maxSeen.has(u.id) || u.hp > maxSeen.get(u.id)) maxSeen.set(u.id, u.hp);
    if (u.deploy > (deployMax.get(u.id) || 0)) deployMax.set(u.id, u.deploy);
    const p = before.get(u.id);
    if (isProj(u)) { if (p) trail.set(u.id, [p.x, p.y]); continue; }
    if (!p) {   // new on the board
      bornAt.set(u.id, now);
      if (isSpell(u)) spells++;
      if (u.team === 1 && u.card_id >= 0 && !initialIds.has(u.id)) {   // their play: one line per card per tick
        const key = u.card_id + "@" + cur.tick;
        if (!oppSeen.has(key)) {
          oppSeen.add(key); theirDeploy++;
          logEvent("Them", `${CAT.cards[u.card_id] ? CAT.cards[u.card_id].name : u.name} at (${Math.round(u.x)}, ${Math.round(u.y)})`);
        }
      }
      continue;
    }
    const d = u.hp - p.hp;
    if (!d || !u.targetable || !u.name) continue;
    const spread = (Math.random() - .5) * .5;
    if (d < 0) {
      floats.push({x: u.x, y: u.y, dx: spread, text: String(-d), col: u.team === 0 ? "#ff9a8a" : "#ffffff", t0: now, big: u.tower});
      flashUntil.set(u.id, now + 110);
      if (u.tower) towerHit++;
      else {
        const att = cur.units.find(v => v.target === u.id && v.team !== u.team), r = att && reachOf(att);
        if (att && r && r.range < 2) melee++; else impact++;
      }
    } else floats.push({x: u.x, y: u.y, dx: spread, text: "+" + d, col: "#7fe08a", t0: now});
  }
  for (const p of prev.units) {
    if (after.has(p.id)) continue;
    if (isProj(p)) { puffs.push({x: p.x, y: p.y, r: .5, col: C.team[p.team], t0: now, life: 260}); trail.delete(p.id); continue; }
    if (isSpell(p)) continue;
    if (p.tower) {
      Sfx.play("towerDown");
      puffs.push({x: p.x, y: p.y, r: 3, col: "#ffb03a", t0: now, life: 900, big: true});
      logEvent("·", `<b>${p.team === 0 ? "Your" : "Their"} ${towerName(p)} is down</b>`, p.team === 0 ? "bad" : "");
    } else if (p.targetable !== false) {
      deaths++;
      puffs.push({x: p.x, y: p.y, r: Math.max(.6, bodyR(p)), col: "#f4f4f4", t0: now, life: 380});
    }
    flashUntil.delete(p.id); deployMax.delete(p.id); maxSeen.delete(p.id); bornAt.delete(p.id);
  }
  if (spells) Sfx.play("spell");
  if (towerHit) Sfx.play("towerHit");
  if (impact) Sfx.play("impact");
  if (melee) Sfx.play("melee");
  if (deaths) Sfx.play("death");
  if (theirDeploy) Sfx.play("deployThem");
  // the clock: 1 min left, the last 10 s, double / triple elixir, overtime
  const l0 = B.regulation_end - prev.clock, l1 = B.regulation_end - cur.clock;
  if (l0 > 60 && l1 <= 60) { Sfx.play("minute"); logEvent("·", "1:00 left"); }
  if (l1 > 0 && l1 <= 10 && Math.ceil(l0) !== Math.ceil(l1)) Sfx.play("tick");
  if (prev.clock < B.double_at && cur.clock >= B.double_at) { Sfx.play("double"); logEvent("·", "Double elixir"); }
  if (prev.clock < B.triple_at && cur.clock >= B.triple_at) { Sfx.play("double"); logEvent("·", "Triple elixir"); }
  if (l0 > 0 && l1 <= 0) logEvent("·", "Overtime");
  // a full bar leaks: a cue when it fills, again every 4 s while it stays full
  if (cur.elixir >= 10 - 1e-6) { if (prev.elixir < 10 - 1e-6 || now - leakAt > 4000) { Sfx.play("leak"); leakAt = now; } }
}

function onPlays(plays) {   // your plays as the server recorded them (landed, or refused and why)
  for (const p of plays.slice(nPlays)) {
    if (p.landed) { Sfx.play("deploy"); logEvent("You", `${p.card} → (${p.x}, ${p.y})`); }
    else { Sfx.play("refused"); logEvent("You", `${p.card} → (${p.x}, ${p.y}) refused${p.why ? ": " + p.why : ""}`, "bad"); }
  }
  nPlays = plays.length;
}

function setState(st, now) {
  Sp = S; S = st; tS = now;
  prevById = new Map((Sp ? Sp.units : []).map(u => [u.id, u]));
  if (Sp) onTick(Sp, S, now);
  updateHud();
}

// ---------------------------------------------------------------- the hand, elixir, clock, inspector
function cardIcon(canvas, name, spell) {
  const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  canvas.width = Math.round((r.width || 52) * d); canvas.height = Math.round((r.height || 52) * d);
  const g = canvas.getContext("2d"), s = canvas.width;
  g.clearRect(0, 0, s, s);
  if (!name) return;
  if (spell) Sprites.spell(g, name, s / 2, s / 2, s * .42, s * .42, 0.3, 0);
  else Sprites.draw(g, name, 0, s / 2, s / 2, s * .95);
}
const cardBtns = [0, 1, 2, 3].map(i => {
  const b = document.createElement("button"); b.className = "cardbtn"; b.type = "button";
  b.innerHTML = `<canvas></canvas><span class="nm"></span><span class="key">${i + 1}</span><span class="cost"></span><span class="fill"></span>`;
  b.addEventListener("pointerdown", ev => { ev.preventDefault(); Sfx.unlock(); pick(i); });   // on press: a click straddling an update can't be lost
  $("hand").appendChild(b); return b;
});
const nextBox = $("next");
nextBox.innerHTML = `<canvas></canvas><span></span>`;
function pick(i) {
  if (!S || S.done || !S.hand[i] || S.hand[i].cost > S.elixir + 1e-6) return;
  sel = sel === i ? null : i; Sfx.play("select"); updateHud(); tileText();
}
function updateHud() {
  cardBtns.forEach((b, i) => {
    const c = S && S.hand[i];
    const id = c ? String(c.id) : "";
    if (b.dataset.id !== id) {
      b.dataset.id = id;
      b.querySelector(".nm").textContent = c ? c.name : "";
      b.querySelector(".cost").textContent = c ? c.cost : "";
      b.querySelector(".cost").hidden = !c;
      cardIcon(b.querySelector("canvas"), c && c.name, c && c.spell);
    }
    const short = !c || S.done || c.cost > S.elixir + 1e-6;
    b.disabled = !c || S.done; b.classList.toggle("low", !!c && short);
    b.querySelector(".fill").style.width = c ? (clamp(S.elixir / c.cost, 0, 1) * 100) + "%" : "0";
    b.classList.toggle("sel", sel === i);
  });
  const n = S && S.next;
  if ((nextBox.dataset.id || "") !== (n ? String(n.id) : "")) {
    nextBox.dataset.id = n ? String(n.id) : "";
    nextBox.querySelector("span").textContent = n ? `${n.name} · ${n.cost}` : "–";
    const k = n && CAT && CAT.cards[n.id];
    cardIcon(nextBox.querySelector("canvas"), n && n.name, k && k.spell);
  }
  $("queued").textContent = pending ? `${pending.card.name} queued` : "";
  if (!S) return;
  $("elx").style.width = (S.elixir * 10) + "%"; $("elxn").textContent = S.elixir.toFixed(1);
  $("oelx").style.width = (S.opp_elixir * 10) + "%"; $("oelxn").textContent = S.opp_elixir.toFixed(1);
  $("ebarYou").classList.toggle("full", S.elixir >= 10 - 1e-6 && !S.done);
  if (S.clock != null && B) {   // counts down to 3:00 of regulation, then overtime
    const left = B.regulation_end - S.clock;
    $("clock").textContent = left > 0 ? mmss(Math.ceil(left)) : "OT +" + mmss(-left);
    $("clock").classList.toggle("late", left <= 10);
    const ph = S.clock >= B.triple_at ? "3× elixir" : S.clock >= B.double_at ? "2× elixir" : "";
    $("phase").textContent = ph; $("phase").hidden = !ph;
  }
  status();
  inspect();
}
let flashMsgUntil = 0;
function flash(msg) { $("status").textContent = msg; flashMsgUntil = performance.now() + 2000; }
function status() {
  if (performance.now() < flashMsgUntil) return;
  let t = !S ? "pick a situation and press New (N)" : S.done ? "finished" : `${(+S.t).toFixed(1)} s in · ${speed}×`;
  if (S && !S.done && paused) t += " · paused (Space)";
  if (scoring) t = "scoring: the scripted defence and the bot replay the same start…";
  $("status").textContent = t;
}
function tileText() {
  if (!hov) { $("tile").textContent = "hover a tile"; return; }
  const c = sel !== null && S && S.hand[sel] ? S.hand[sel] : null;
  let t = `tile (${hov.x}, ${hov.y})`;
  if (c) t += legalAt(c, hov.x, hov.y) ? ` · ${c.name} ✓` : ` · ${c.name}: not here`;
  $("tile").textContent = t;
}
let inspected = "";
function inspect() {
  const u = hovUnit, box = $("inspect");
  if (!u || !S) { if (inspected !== "") { box.innerHTML = "Hover a unit for its HP, target and reach."; inspected = ""; } return; }
  const k = cardOf(u), r = reachOf(u), full = u.max_hp || maxSeen.get(u.id) || u.hp || 1;
  const t = S.units.find(v => v.id === u.target);
  const rows = [["Side", u.team === 0 ? "yours" : "theirs"],
    ["Kind", u.tower ? "crown tower" : isSpell(u) ? "spell" : u.building ? "building" : u.flying ? "air troop" : "ground troop"]];
  if (!isSpell(u)) rows.push(["Target", t ? `${t.name || "?"}${t.tower ? "" : ` (${t.team === u.team ? "own" : t.team === 0 ? "yours" : "theirs"})`}` : "none"]);
  if (r) rows.push(["Reach", `range ${r.range} · sight ${r.sight}${r.air ? " · hits air" : " · ground only"}`]);
  if (isSpell(u) && k) rows.push(["Radius", k.spell_radius]);
  if (k && k.cost) rows.push(["Cost", k.cost]);
  if (u.deploy > 0) rows.push(["Deploying", `${(u.deploy * B.tick_s).toFixed(1)} s`]);
  if (!u.targetable && !isSpell(u)) rows.push(["Targetable", "no"]);
  const html = `<b>${u.name}</b>` + (isSpell(u) ? "" : `<div class="hp"><div style="width:${clamp(u.hp / full, 0, 1) * 100}%;background:${C.team[u.team]}"></div></div><div>${Math.round(u.hp).toLocaleString()} / ${full.toLocaleString()} HP</div>`)
    + `<dl>${rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("")}</dl>`;
  if (html !== inspected) { box.innerHTML = html; inspected = html; }
}

// ---------------------------------------------------------------- input
function cellAt(ev) {   // the tile under the pointer, and the continuous engine point
  const r = cv.getBoundingClientRect();
  const fx = (ev.clientX - r.left) / r.width * NX(), fy = (r.bottom - ev.clientY) / r.height * NY();
  return {x: clamp(Math.floor(fx), 0, NX() - 1), y: clamp(Math.floor(fy), 0, NY() - 1), ex: fx - 0.5, ey: fy - 0.5};
}
cv.addEventListener("pointermove", ev => { hov = cellAt(ev); tileText(); });
cv.addEventListener("pointerleave", () => { hov = null; hovUnit = null; tileText(); inspect(); });
setInterval(inspect, 150);
cv.addEventListener("pointerdown", ev => {
  Sfx.unlock();
  if (!S || sel === null || S.done) return;
  const {x, y} = cellAt(ev), c = S.hand[sel];
  if (!legalAt(c, x, y)) { flash(c.spell ? "spells can't go on that tile" : `${c.name} can't go on tile (${x}, ${y})`); Sfx.play("refused"); return; }
  pending = {slot: sel, x, y, card: c}; sel = null; updateHud(); tileText();
});

function setSpeed(s) {
  speed = s; save("speed", s);
  for (const b of document.querySelectorAll("#speed button")) b.classList.toggle("on", +b.dataset.s === s);
  status();
}
for (const b of document.querySelectorAll("#speed button")) b.onclick = () => setSpeed(+b.dataset.s);
function togglePause() {
  paused = !paused; $("pause").textContent = paused ? "Resume" : "Pause"; $("pause").classList.toggle("on", paused); status();
}
$("pause").onclick = togglePause;
function soundUi() { $("mute").textContent = Sfx.muted ? "🔇" : "🔊"; $("vol").value = Sfx.volume; }
$("mute").onclick = () => { Sfx.setMuted(!Sfx.muted); soundUi(); };
$("vol").oninput = ev => { Sfx.setVolume(ev.target.value); if (Sfx.muted) Sfx.setMuted(false); soundUi(); };
$("vol").onchange = () => Sfx.play("select");
$("helpbtn").onclick = () => { $("help").open = !$("help").open; };
function toggle(k, msg) { opt[k] = !opt[k]; save(k, opt[k]); if (k === "coords") buildTerrain(); flash(`${msg} ${opt[k] ? "on" : "off"}`); }

document.addEventListener("keydown", ev => {
  Sfx.unlock();
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;   // leave Cmd-R and friends to the browser
  if (ev.target.tagName === "SELECT" || ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA") return;
  const k = ev.key;
  if (["1", "2", "3", "4"].includes(k)) pick(+k - 1);
  else if (k === "Escape") { sel = null; pending = null; updateHud(); tileText(); }
  else if (k === " ") togglePause();
  else if (k === "z" || k === "Z" || k === "Backspace") {
    if (pending && paused) { logEvent("You", `took back ${pending.card.name} (never sent)`, ""); pending = null; updateHud(); }
    else if (pending) flash("pause first (Space) to take a play back");
    else return;
  }
  else if (k === "r" || k === "R") $("restart").click();
  else if (k === "n" || k === "N") $("start").click();
  else if (k === "[") setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(speed) - 1)]);
  else if (k === "]") setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(speed) + 1)]);
  else if (k === "m" || k === "M") $("mute").click();
  else if (k === "l" || k === "L") toggle("labels", "unit labels");
  else if (k === "g" || k === "G") toggle("coords", "tile numbers");
  else if (k === "d" || k === "D") toggle("dmg", "damage numbers");
  else if (k === "?" || k === "h" || k === "H") $("helpbtn").click();
  else return;
  ev.preventDefault();
});

// ---------------------------------------------------------------- the game
async function loop(g) {
  while (g === gen && S && !S.done) {
    const t0 = performance.now();
    if (!paused) {
      const act = pending; pending = null;
      const slow = setTimeout(() => { scoring = true; status(); }, 400);   // the last tick waits for the two replays
      const out = await post("/step", act ? {slot: act.slot, x: act.x, y: act.y} : {});
      clearTimeout(slow); scoring = null;
      if (g !== gen) return;                     // a New / Restart happened while this step was in flight
      if (act) lastPlay = act;
      onPlays(out.plays || []);
      setState(out.state, performance.now());
      if (out.result) { showResult(out.result); break; }
    }
    await sleep(Math.max(0, tickMs() / speed - (performance.now() - t0)));
  }
}

async function begin(seed) {
  const g = ++gen;                                 // cancels any running loop
  sel = null; pending = null; lastPlay = null; paused = false; nPlays = 0; scoring = null;
  maxSeen.clear(); flashUntil.clear(); deployMax.clear(); bornAt.clear(); trail.clear(); oppSeen.clear(); floats = []; puffs = [];
  $("pause").textContent = "Pause"; $("pause").classList.remove("on");
  $("result").innerHTML = ""; $("resultbox").hidden = true; $("banner").hidden = true; $("log").innerHTML = "";
  const out = await post("/new", {template: $("tpl").value, seed});
  if (g !== gen) return;
  lastSeed = out.seed; lastTpl = $("tpl").value; lastThreat = out.threat.join(" + ") || "nothing yet";
  sitName = $("tpl").selectedOptions[0].textContent;
  $("sitname").textContent = `${sitName} · seed ${out.seed}`;
  B = out.board || B; resize();
  S = null; Sp = null; prevById = new Map();
  initialIds = new Set(out.state.units.map(u => u.id));
  setState(out.state, performance.now());
  for (const u of S.units) { maxSeen.set(u.id, u.hp); if (u.deploy) deployMax.set(u.id, u.deploy); }
  $("setup").innerHTML = `Threat <b>${lastThreat}</b>${out.plan ? ` · their plan <b>${PLAN[out.plan] || out.plan}</b>` : ""} · your elixir <b>${out.start_elixir[0]}</b> · theirs <b>${out.start_elixir[1]}</b>`;
  logEvent("·", `Start: ${lastThreat} · you ${out.start_elixir[0]} · them ${out.start_elixir[1]}`);
  const cnt = $("count");
  for (const n of [3, 2, 1]) {                     // the game waits; you can already queue a card
    if (g !== gen) return;
    cnt.textContent = n; cnt.hidden = false; flash(`starting in ${n}`); Sfx.play("count");
    await sleep(1000);
  }
  if (g !== gen) return;
  cnt.hidden = true; Sfx.play("go"); flashMsgUntil = 0;
  loop(g);
}

function showResult(R) {
  const rows = [["You", R.you], ["Scripted defence", R.scripted], ["Bot", R.model]];
  const best = Math.max(...rows.map(r => r[1].score)), worst = Math.min(R.scripted.score, R.model.score);
  const fmt = r => `<td class="score">${(r.score >= 0 ? "+" : "") + r.score.toFixed(2)}</td><td>${Math.round(r.tower_hp_lost)}</td><td>${r.elixir_spent}</td><td>${r.unresolved ? "alive" : "handled"}</td><td>${r.seconds}s</td>`;
  $("result").innerHTML = `<table class="res"><tr><th></th><th>score</th><th>tower HP lost</th><th>elixir</th><th>threat</th><th>time</th></tr>
    ${rows.map(([n, r]) => `<tr class="${r.score >= best - 1e-9 ? "best" : ""}"><td>${n}</td>${fmt(r)}</tr>`).join("")}</table>
    <div class="muted small" style="margin-top:6px">Same start for all three; the opponent clone plays on, so its plays can differ.</div>`;
  $("resultbox").hidden = false;
  const you = R.you.score, top = Math.max(R.scripted.score, R.model.score);
  const [cls, head, snd] = you >= top - 1e-9 ? ["win", "Defended", "win"] : you >= worst - 1e-9 ? ["even", "In between", "even"] : ["lose", "They did better", "lose"];
  const bn = $("banner");
  bn.className = "banner " + cls;
  bn.innerHTML = `<b>${head}</b>you ${(you >= 0 ? "+" : "") + you.toFixed(2)} · scripted ${R.scripted.score.toFixed(2)} · bot ${R.model.score.toFixed(2)}<div class="small" style="opacity:.75;margin-top:4px">R restart · N new</div>`;
  bn.hidden = false;
  Sfx.play(snd);
  logEvent("·", `Finished: you ${you.toFixed(2)}, scripted ${R.scripted.score.toFixed(2)}, bot ${R.model.score.toFixed(2)}`);
}

const PLAN = {tank_back: "tank in the back", bridge_rush: "bridge rush", support_back: "support in the back",
  wait_defend: "wait and defend", defend: "defend what crosses", punish: "punish an overcommit",
  tempo: "defend what crosses, punish a commitment, push at a full bar"};
$("start").onclick = () => { Sfx.unlock(); begin(null); };
$("restart").onclick = () => { Sfx.unlock(); if (lastSeed !== null) { $("tpl").value = lastTpl; begin(lastSeed); } };

async function loadSaved() {
  const list = await (await fetch("/saved")).json(), s = $("savedlist");
  s.innerHTML = `<option value="">Saved starts (${list.length})…</option>` + list.map((x, i) =>
    `<option value="${i}">${x.template.replace("_", " ")} · ${x.threat} · seed ${x.seed}${x.note ? " · " + x.note : ""}</option>`).join("");
  s.dataset.list = JSON.stringify(list);
}
$("save").onclick = async () => {
  if (lastSeed === null) return;
  await post("/save", {template: lastTpl, seed: lastSeed, note: $("note").value, threat: lastThreat});
  $("note").value = ""; $("save").textContent = "Saved"; setTimeout(() => $("save").textContent = "Save", 1200);
  loadSaved();
};
$("savedlist").onchange = ev => {
  const list = JSON.parse(ev.target.dataset.list || "[]"), x = list[+ev.target.value];
  if (!x) return;
  $("tpl").value = x.template; ev.target.value = ""; begin(x.seed);
};

// ---------------------------------------------------------------- start up
readColors(); soundUi(); setSpeed(speed);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); buildTerrain(); });
new ResizeObserver(resize).observe(cv);
Promise.all([fetch("/board").then(r => r.json()), fetch("/catalog").then(r => r.json())]).then(([b, cat]) => {
  B = b; CAT = cat;
  for (const k of Object.values(cat.cards)) if (!byName.has(k.name)) byName.set(k.name, k);
  resize(); updateHud();
});
loadSaved(); updateHud();
requestAnimationFrame(frame);
})();
