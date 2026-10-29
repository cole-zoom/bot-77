"use strict";
// The play page's sounds, synthesised with Web Audio (oscillators and filtered noise; no audio files, nothing from
// the game). The browser only allows sound after a click or a key, so the context starts on the first one.
// Volume and mute are remembered in localStorage (wrapped: a private window or blocked storage just forgets).
(function () {
const KEY = "bot77-play:";
const load = (k, d) => { try { const v = localStorage.getItem(KEY + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } };
const save = (k, v) => { try { localStorage.setItem(KEY + k, JSON.stringify(v)); } catch (e) { /* not remembered */ } };

let ac = null, master = null, noiseBuf = null;
let volume = load("volume", 0.6), muted = load("muted", false);
const last = new Map();   // name -> last start (s), so a burst of hits in one tick doesn't stack into noise

function ensure() {
  if (!ac) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return false;
    ac = new AC();
    master = ac.createGain(); master.connect(ac.destination);
    noiseBuf = ac.createBuffer(1, ac.sampleRate, ac.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    apply();
  }
  if (ac.state === "suspended") ac.resume();
  return true;
}
function apply() { if (master) master.gain.value = muted ? 0 : volume * volume; }

function env(node, t, gain, attack, dur) {
  node.gain.setValueAtTime(0.0001, t);
  node.gain.exponentialRampToValueAtTime(Math.max(gain, 0.0002), t + attack);
  node.gain.exponentialRampToValueAtTime(0.0001, t + dur);
}
function tone(freq, dur, type = "square", gain = 0.2, slideTo = null, delay = 0) {
  const t = ac.currentTime + delay, o = ac.createOscillator(), v = ac.createGain();
  o.type = type; o.frequency.setValueAtTime(freq, t);
  if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
  env(v, t, gain, 0.005, dur);
  o.connect(v); v.connect(master); o.start(t); o.stop(t + dur + 0.05);
}
function noise(dur, gain = 0.3, freq = 1000, type = "lowpass", delay = 0, q = 1) {
  const t = ac.currentTime + delay, s = ac.createBufferSource(), f = ac.createBiquadFilter(), v = ac.createGain();
  s.buffer = noiseBuf; f.type = type; f.frequency.value = freq; f.Q.value = q;
  env(v, t, gain, 0.004, dur);
  s.connect(f); f.connect(v); v.connect(master); s.start(t, Math.random() * 0.5); s.stop(t + dur + 0.05);
}
const arp = (notes, step, dur, type, gain) => notes.forEach((f, i) => tone(f, dur, type, gain, null, i * step));

const SOUNDS = {
  select: [0.03, () => tone(1300, 0.035, "square", 0.05)],
  deploy: [0.05, () => { noise(0.12, 0.18, 700, "lowpass"); tone(330, 0.09, "triangle", 0.22, 660); tone(660, 0.1, "triangle", 0.12, null, 0.06); }],
  deployThem: [0.05, () => { noise(0.12, 0.14, 500, "lowpass"); tone(260, 0.12, "triangle", 0.18, 170); }],
  refused: [0.1, () => { tone(180, 0.09, "square", 0.12); tone(140, 0.12, "square", 0.12, null, 0.1); }],
  melee: [0.045, () => { noise(0.05, 0.3, 2200, "bandpass", 0, 2); tone(160, 0.05, "square", 0.08, 90); }],
  impact: [0.045, () => { noise(0.08, 0.26, 900, "lowpass"); tone(110, 0.07, "sine", 0.2, 60); }],
  spell: [0.12, () => { noise(0.4, 0.35, 600, "lowpass"); tone(140, 0.35, "sawtooth", 0.1, 40); }],
  death: [0.06, () => { tone(520, 0.16, "square", 0.08, 110); noise(0.1, 0.1, 3000, "highpass"); }],
  towerHit: [0.08, () => { tone(220, 0.06, "square", 0.08, 170); noise(0.05, 0.16, 1400, "bandpass", 0, 3); }],
  towerDown: [0.5, () => { noise(1.1, 0.55, 380, "lowpass"); tone(110, 0.9, "sawtooth", 0.2, 30); tone(70, 1.1, "sine", 0.3, 28); }],
  leak: [0.5, () => { tone(880, 0.08, "sine", 0.16); tone(660, 0.1, "sine", 0.16, null, 0.1); }],
  minute: [1, () => arp([523, 659, 784], 0.12, 0.16, "triangle", 0.2)],
  tick: [0.3, () => tone(1050, 0.05, "square", 0.1)],
  double: [1, () => arp([392, 523, 659, 784, 1046], 0.08, 0.14, "triangle", 0.2)],
  count: [0.3, () => tone(660, 0.1, "sine", 0.22)],
  go: [0.3, () => tone(990, 0.22, "sine", 0.22)],
  win: [1, () => { arp([523, 659, 784, 1046], 0.11, 0.18, "triangle", 0.22); tone(1046, 0.5, "triangle", 0.16, null, 0.45); }],
  even: [1, () => arp([523, 587, 523], 0.13, 0.16, "triangle", 0.18)],
  lose: [1, () => { arp([523, 466, 392], 0.16, 0.2, "triangle", 0.2); tone(311, 0.6, "triangle", 0.16, 260, 0.48); }],
};

window.Sfx = {
  unlock: ensure,
  play(name) {
    if (muted || !SOUNDS[name] || !ensure()) return;
    const [gap, fn] = SOUNDS[name], now = ac.currentTime;
    if (now - (last.get(name) ?? -9) < gap) return;
    last.set(name, now);
    try { fn(); } catch (e) { /* a sound never breaks the game */ }
  },
  get volume() { return volume; },
  get muted() { return muted; },
  setVolume(v) { volume = Math.max(0, Math.min(1, +v)); save("volume", volume); apply(); },
  setMuted(m) { muted = !!m; save("muted", muted); apply(); },
};
})();
