"""Click-anchored auto-label candidates for entities still under the bar, for Cole to approve.

  build:  uv run python scripts/candidates.py build <coverage.log> [video,ids]   -> data/reviews/candidates.html
  import: uv run python scripts/candidates.py import <export.json>             -> real_v1/manifest_reviewed.json

Anchoring on Cole's clicked crops raised yield ~5x on his recordings but about half the boxes were wrong, so these
never go straight into the training set (decisions.md, 2026-10-01)."""
import json, random, shutil, sys
from pathlib import Path
from bot77.autolabel import OUT, AutoLabeler, collect_items
from bot77.spotcheck import write_spotcheck

CAND = Path("data/external/real_v1_candidates")
if sys.argv[1] == "build":
    need = {l.split()[0] for l in open(sys.argv[2]) if "need" in l}
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    have = {x["name"] for f in OUT.glob("manifest_*.json") for x in json.loads(f.read_text())}
    items = [i for i in collect_items() if i["entity"] in need and (only is None or i["video_id"] in only)]
    al, rng, out = AutoLabeler(out=CAND, click_anchors=True), random.Random(0), []
    for i, it in enumerate(items):
        try:
            out += [x for x in al.label(it, rng) if x and x["name"] not in have]
        except Exception as e:  # noqa: BLE001
            print("ERR", it["video_id"], it["t_video"], e, flush=True)
    # filter with the crop classifier (bot77/cropclf.py): keep p(entity) >= REVIEW_MIN_P, or unscorable entities
    import cv2
    from bot77.cropclf import REVIEW_MIN_P, CropClassifier
    clf, kept = CropClassifier(emb=al.anchor.emb), []
    for x in out:
        im = cv2.imread(str(CAND / "images" / f"{x['name']}.jpg")); H, W = im.shape[:2]
        _, cx, cy, w, h = map(float, (CAND / "labels" / f"{x['name']}.txt").read_text().splitlines()[0].split())
        crop = im[max(int((cy - h / 2) * H), 0):int((cy + h / 2) * H), max(int((cx - w / 2) * W), 0):int((cx + w / 2) * W)]
        pe = clf.p_entity(crop, x["entity"])
        if pe is None or pe >= REVIEW_MIN_P:
            kept.append({**x, "clf_p": pe})
    print(f"classifier filter kept {len(kept)} of {len(out)}", flush=True)
    out = kept
    (CAND / "manifest_0.json").write_text(json.dumps(out, indent=1))
    n = write_spotcheck(CAND, Path("data/reviews/candidates.html"), n=len(out), seed=0)
    print(len(items), "plays,", n, "candidate frames", dict(al.stats), flush=True)
else:
    verdict = {f["id"]: f.get("verdict") for f in json.loads(Path(sys.argv[2]).read_text())["frames"]}
    keep = [x for x in json.loads((CAND / "manifest_0.json").read_text()) if verdict.get(x["name"]) == "right"]
    for x in keep:
        for sub, ext in (("images", "jpg"), ("labels", "txt")):
            shutil.copy(CAND / sub / f"{x['name']}.{ext}", OUT / sub / f"{x['name']}.{ext}")
    prev = OUT / "manifest_reviewed.json"
    old = json.loads(prev.read_text()) if prev.exists() else []
    names = {x["name"] for x in old}
    prev.write_text(json.dumps(old + [{**x, "how": "candidate-approved"} for x in keep if x["name"] not in names], indent=1))
    print(len(keep), "approved of", len(verdict))
