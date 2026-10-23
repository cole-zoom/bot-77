"""One-time seed of the models registry (bot77.models, 2026-10-22) with every model version on disk and its history
from decisions.md (scores quoted as written there; `notes` gives the lines). From now on train_moves.py /
train_place.py register what they make. Re-running replaces these rows.

  uv run python scripts/seed_models.py
"""
import pickle
from pathlib import Path

from bot77.models import register

M = "data/models"
D = f"{M}/bot77_detector"


def feats(pkl):  # the feature list saved with a card / placement model
    try:
        return pickle.loads(Path(pkl).read_bytes())["features"]
    except Exception:  # noqa: BLE001
        return []


ROWS = [
    # ---- unit detector (the log's V-numbers differ from the pod run names: V3 = v4l, V4 = v5l, V5 = v6l)
    ("detector:v1", "detector", "v1", f"{D}/v1/weights/best.pt", "detector", "superseded", "2026-09-29",
     "V1: YOLO11s at 896 px, 12 epochs on synthetic v1 (KataCR generator, 12,000 images + 3 new classes)",
     {"KataCR 800 real frames mAP50": "0.70-0.73", "52-play benchmark fully right": "22 (42%)",
      "benchmark 2024 + v1 ensemble": "32 (62%)"}, "decisions.md 223-248, 1371-1382"),
    ("detector:v2_smoke", "detector", "v2_smoke", f"{D}/v2_smoke/weights/best.pt", "detector", "not adopted", "2026-10-09",
     "smoke test of train_v2.py (1 epoch, 3% of data)", {}, "decisions.md 1364-1368"),
    ("detector:v2m", "detector", "v2m", f"{D}/v2m/best.pt", "detector", "not adopted", "2026-10-10",
     "run 1: yolo11m on real_v2 + synth_v1 (219 classes); best = epoch 3", {"val mAP50-95 peak (e3)": "0.409",
      "benchmark best (e3) Alina / Cole / friend": "25/52 / 3/34 / 0/25"}, "decisions.md 1385-1425; e3 later pseudo-labelled real_v3"),
    ("detector:v3l", "detector", "v3l", f"{D}/v3l/epoch15.pt", "detector", "superseded", "2026-10-10",
     "run 2: yolo11l on real_v3 x3 + synth_v2; epoch 15 = the start of V3",
     {"dev e10 (v1 play logic)": "67/473 fully right (14.2%)", "benchmark e15 rules v2": "43/111"}, "decisions.md 1474-1584"),
    ("detector:v4l", "detector", "v4l", f"{D}/v4l/best.pt", "detector", "superseded", "2026-10-11",
     "V3: from run 2 e15, 30 epochs on real x3 + synth_v3 (game-art sprites + 3D renders)",
     {"dev current play code e10 / e15 / e20": "27.9% / 30.4% / 30.7%"}, "decisions.md 1585-1653; e20 = the start of V4"),
    ("detector:v5l", "detector", "v5l", f"{D}/v5l/epoch15.pt", "detector", "superseded", "2026-10-11",
     "V4 (epoch 15): from V3 e20, real_v3 x3 + real_v4 + synth_v3",
     {"dev e15 clock logic": "42.3%", "replay test (30) final 10-16": "806 / 95 / 243 (66.8%)"},
     "decisions.md 1654-2260. Still the source of unit_detections_v5l_e15 (replay boards for A / placement) and held-back rule tuning"),
    ("detector:v6l", "detector", "v6l", f"{D}/v6l/best.pt", "detector_live", "adopted", "2026-10-14",
     "V5 (best = epoch 11): from V4, + 7,478 real_v5r (replay clock-spot labels)",
     {"val mAP50 / mAP50-95": "0.742 / 0.434", "dev (with spells)": "59.5 / 40 / 9.2",
      "replay test (30) with spells": "59.8 / 141 / 6.5", "live test": "148 / 13 / 51 (51.0%, 24% fake)"},
     "decisions.md 1966-2362. The live bot's detector and C's POV default ('the default detector from here', 10-14)"),
    ("detector:v7l", "detector", "v7l", f"{D}/v7l/best.pt", "detector", "not adopted", "2026-10-16",
     "V7: 4 epochs from V5 on real + mined x10, no synth", {"dev": "421 / 48 / 228", "held-back": "1146 / 192 / 251"},
     "decisions.md 2200-2204: dropping synth cost more than the mined frames gained"),
    ("detector:v8l", "detector", "v8l", f"{D}/v8l/best.pt", "detector", "not adopted", "2026-10-18",
     "V8: 12 epochs from V5 on real + synth + mined x5 + live x3 + replays 10-17; best epoch 7",
     {"dev": "429 / 40 / 190", "live test": "147 / 19 / 44 (50.7%)", "replay test (30)": "781 / 105 / 148"},
     "decisions.md 2264-2284: no gain over V5 on dev (pre-registered rule)"),
    # ---- spell clip classifiers (components of the spell mixes)
    ("spell:spell_v1", "spell", "spell_v1", f"{M}/spell_v1/best.pt", "spell_component", "superseded", "2026-10-15",
     "R(2+1)D-18 clip classifier on 12,597 clips from 350 replays", {"held-back clips spells named right": "91%",
      "replay test (30) V5": "46.5 / 191 / 6.2 -> 59.8 / 141 / 6.5"}, "decisions.md 2008-2040"),
    ("spell:spell_v2", "spell", "spell_v2", f"{M}/spell_v2/best.pt", "spell_component", "in use", "2026-10-15",
     "+ ability positives (13,489 clips, 15 ability classes)", {"held-back clips spells named right": "88%",
      "benchmark (116)": "60 / 7 / 30 -> 65 / 5 / 28"}, "decisions.md 2082-2101: part of every adopted mix"),
    ("spell:spell_v3", "spell", "spell_v3", f"{M}/spell_v3/best.pt", "spell_component", "superseded", "2026-10-15",
     "16,337 clips incl. 2,848 from Cole's POV", {"best val macro-F1": 0.778}, "decisions.md 2135-2139: only inside spell_mix"),
    ("spell:spell_v4", "spell", "spell_v4", f"{M}/spell_v4/best.pt", "spell_component", "not adopted", "2026-10-16",
     "v3 data + 13 relabelled clips", {"val macro-F1": 0.804, "held-back / dev right v2+v4": "1221 / 428"},
     "decisions.md 2205-2207: worse on whole games in every mix"),
    ("spell:spell_v5", "spell", "spell_v5", f"{M}/spell_v5/best.pt", "spell_component", "in use", "2026-10-18",
     "17,107 clips (+770 from Cole's live games)", {"val macro-F1": 0.781}, "decisions.md 2261-2270: in spell_mix25 and spell_mix256"),
    ("spell:spell_v6", "spell", "spell_v6", f"{M}/spell_v6/best.pt", "spell_component", "in use", "2026-10-21",
     "19,488 clips (+2,381 from 71 games)", {"val macro-F1": 0.791}, "decisions.md 2331-2336: in spell_mix256"),
    # ---- spell probability mixes (what C actually uses)
    ("spell_mix:spell_mix", "spell_mix", "spell_mix", f"{M}/spell_mix/spell_probs", "spell_pov", "superseded", "2026-10-15",
     "v2 + v3 averaged (Goblin Curse, Vines, abilities from v2)", {"dev": "430 / 47 / 215 (58.3%)",
      "replay test (30)": "780 / 120 / 196 (64.6%)"}, "decisions.md 2140-2151"),
    ("spell_mix:spell_mix25", "spell_mix", "spell_mix25", f"{M}/spell_mix25/spell_probs", "spell_pov", "adopted", "2026-10-18",
     "v2 + v5", {"dev": "425 / 46 / 165", "held-back": "1206 / 160 / 218"}, "decisions.md 2268-2270, 2334: POV default"),
    ("spell_mix:spell_mix256", "spell_mix", "spell_mix256", f"{M}/spell_mix256/spell_probs", "spell_replay", "adopted", "2026-10-21",
     "v2 + v5 + v6", {"held-back": "1224 / 152 / 201", "replay test V5 dets": "785 / 110 / 134 -> 774 / 117 / 128"},
     "decisions.md 2334-2343: replay default (REPLAY_SPELL_PROBS)"),
    ("spell_mix:spell_mix26", "spell_mix", "spell_mix26", f"{M}/spell_mix26/spell_probs", "spell_replay", "not adopted", "2026-10-21",
     "v2 + v6", {"held-back": "1219 / 159 / 208", "dev right": "418"}, "decisions.md 2334-2335"),
    ("spell_mix:spell_mix235", "spell_mix", "spell_mix235", f"{M}/spell_mix235/spell_probs", "spell_mix", "not adopted", "2026-10-18",
     "v2 + v3 + v5 (experiment; not in the decision log)", {}, "not in decisions.md"),
    ("spell_mix:spell_mix25ab", "spell_mix", "spell_mix25ab", f"{M}/spell_mix25ab/spell_probs", "spell_mix", "not adopted", "2026-10-18",
     "v2 + v5 variant (experiment; not in the decision log)", {}, "not in decisions.md"),
    ("spell_mix:spell_mix25avg", "spell_mix", "spell_mix25avg", f"{M}/spell_mix25avg/spell_probs", "spell_mix", "not adopted", "2026-10-18",
     "v2 + v5 plain average (experiment; not in the decision log)", {}, "not in decisions.md"),
    ("spell_probs:probs_new_v2", "spell_probs", "probs_new_v2", f"{M}/probs_new_v2", "spell_inputs", "reference", "2026-10-18",
     "spell_v2 probabilities for the streamed games (inputs to the mixes)", {}, "decisions.md 2267, 2333"),
    ("spell_probs:probs_new_v3", "spell_probs", "probs_new_v3", f"{M}/probs_new_v3", "spell_inputs", "reference", "2026-10-18",
     "spell_v3 probabilities (inputs to the mixes)", {}, "decisions.md 2267, 2333"),
    ("spell_probs:probs_new_v5", "spell_probs", "probs_new_v5", f"{M}/probs_new_v5", "spell_inputs", "reference", "2026-10-18",
     "spell_v5 probabilities (inputs to the mixes)", {}, "decisions.md 2267, 2333"),
    # ---- fake-play filter
    ("fake_filter:fake_filter", "fake_filter", "fake_filter", f"{M}/fake_filter/model.pkl", "fake_filter", "superseded", "2026-10-16",
     "v1: gradient-boosted P(fake) per play, 31 features, 332 replays", {"held-back AUC": 0.868, "dev AUC": 0.648},
     "decisions.md 2227-2235"),
    ("fake_filter:fake_filter_v2", "fake_filter", "fake_filter_v2", f"{M}/fake_filter_v2/model.pkl", "fake_filter", "superseded", "2026-10-18",
     "+ 10 live training games", {"held-back AUC": 0.869, "dev AUC": "0.648 -> 0.695", "live test": "148 / 13 / 51 (24% fake)"},
     "decisions.md 2273-2279"),
    ("fake_filter:fake_filter_v3", "fake_filter", "fake_filter_v3", f"{M}/fake_filter_v3/model.pkl", "fake_filter", "not adopted", "2026-10-18",
     "+ own-play signal features", {"dev AUC": "0.697 vs 0.695"}, "decisions.md 2297-2298: v2 stays"),
    ("fake_filter:fake_filter_v4", "fake_filter", "fake_filter_v4", f"{M}/fake_filter_v4/model.pkl", "fake_filter", "adopted", "2026-10-20",
     "replays + 33 live games, POV weight 3", {"held-back AUC": 0.865, "dev AUC": "0.695 -> 0.724",
      "live test": "148 / 14 / 29 (15% fake)"}, "decisions.md 2318-2321, 2338-2340: default at 0.6 (POV and replays)"),
    ("fake_filter:fake_filter_v5", "fake_filter", "fake_filter_v5", "", "fake_filter", "not adopted", "2026-10-21",
     "383 rescored replays + 45 live games (not kept on disk)", {"held-back AUC": 0.878, "dev AUC": "0.709-0.711"},
     "decisions.md 2337-2341: v4 stays"),
    # ---- crop classifier
    ("crop_clf:crop_clf", "crop_clf", "crop_clf.pt", f"{M}/crop_clf.pt", "crop_clf", "adopted", "2026-10-02",
     "head on DINOv2 embeddings; KataCR + our verified crops (208 classes)", {"held-out ours / KataCR": "70% / 88%",
      "at >=0.9 confidence": "92.5% right on 67% of crops"}, "decisions.md 819-1017: ranks review pages and the dataset cleaner"),
    ("crop_clf:crop_clf_nocand", "crop_clf", "crop_clf_nocand.pt", f"{M}/crop_clf_nocand.pt", "crop_clf", "reference", "2026-10-01",
     "the fair-test variant with candidate frames held out (the log doesn't name the file)",
     {"Cole's 234 verdicts p>=0.3": "right 18% -> 60%, wrong 40% -> 14%"}, "decisions.md 831-832"),
    # ---- third party
    ("third_party:katacr", "third_party", "katacr", f"{M}/katacr", "third_party", "superseded", "2026-09-28",
     "KataCR's 2024 detectors (Wu et al. 2025, arXiv:2504.04783)", {"52-play benchmark fully right": "31 (60%)"},
     "decisions.md 91-96, 240, 248: stopped relying on it 09-28"),
    ("third_party:sam", "third_party", "sam", f"{M}/sam/sam2.1_t.pt", "labelling", "reference", "2026-09-28",
     "SAM 2.1 tiny: sprite cut-outs, click boxes, tracking", {}, "decisions.md 121"),
    ("third_party:yolo11s", "third_party", "yolo11s.pt", f"{M}/yolo11s.pt", "base_weights", "reference", "2026-09-28",
     "YOLO11s base weights (detector v1)", {}, "decisions.md 227, 1394"),
]
for mid, fam, ver, path, role, status, date, what, metrics, notes in ROWS:
    register(mid, fam, ver, path, role, what=what, metrics=metrics, notes=notes, trained_at=date, status=status,
             adopt=status == "adopted")

# the card and placement models: with their training data and feature lists
register("card:moves_v1", "card", "moves_v1", f"{M}/moves_v1/model.pkl", "card",
         what="A v1: ranks the 4 hand cards (hand, elixir, clock, history, response rates)",
         data={"table": "move_samples", "train_decisions": 19506, "games": 471}, features=feats(f"{M}/moves_v1/model.pkl"),
         metrics={"heldback top-1": 0.466, "heldback top-2": 0.760, "test top-1": 0.440, "test top-2": 0.729,
                  "test top ladder top-1": 0.420},
         notes="decisions.md 2352-2357. The live bot's card model", trained_at="2026-10-21", status="adopted", adopt=True)
register("card:moves_v2", "card", "moves_v2", f"{M}/moves_v2/model.pkl", "card",
         what="A v2: + board features (board_states from unit_detections_v5l_e15)",
         data={"table": "move_samples + board_states", "train_decisions": 19506}, features=feats(f"{M}/moves_v2/model.pkl"),
         metrics={"heldback top-1": 0.484, "test top-1": 0.440, "test top-1, attacker on our half": 0.489},
         notes="decisions.md 2413-2424: no gain on test, not adopted. Trained with the older card roles",
         trained_at="2026-10-22", status="not adopted")
register("place:place_v1", "place", "place_v1", f"{M}/place_v1/model.pkl", "place",
         what="placement model: ranks 32 fixed + board-anchored spots", data={"table": "placements", "train_plays": 11424},
         features=feats(f"{M}/place_v1/model.pkl"),
         metrics={"heldback median tiles": 4.1, "heldback same lane": 0.74, "test median tiles": 4.1, "test same lane": 0.77,
                  "test within 2 tiles": 0.27},
         notes="decisions.md 2425-2438. Trained with the older card roles", trained_at="2026-10-22", status="superseded")
register("place:place_v2", "place", "place_v2", f"{M}/place_v2/model.pkl", "place",
         what="place_v1 retrained through bot77.features with the current card roles",
         data={"table": "placements", "train_plays": 11424}, features=feats(f"{M}/place_v2/model.pkl"),
         metrics={"heldback median tiles": 3.9, "heldback same lane": 0.74, "heldback within 2 tiles": 0.31},
         notes="decisions.md 2453-2455. The bot's shadow placer (the rules still choose the tap)",
         trained_at="2026-10-22", status="adopted", adopt=True)

from bot77.models import list_models  # noqa: E402
rows = list_models()
print(len(rows), "models;", sum(r["adopted"] for r in rows), "adopted:", ", ".join(f"{r['role']}={r['version']}" for r in rows if r["adopted"]))
print("missing files:", [r["model_id"] for r in rows if r["path"] and not Path(r["path"]).exists()])
