"""V4 data engine, step 2: detect the extracted arena crops (mine_v4_extract.py) on a GPU, then import the detections.

On the pod (crops are already the detector's 576x896 input; team logic is bot77.detect.yolo's, run on the crop):
  python3 mine_v4_detect.py detect <weights.pt> <mine_v4 dir>      -> <dir>/detections.jsonl (crop coordinates)
Back on the Mac:
  uv run python scripts/mine_v4_detect.py import <detections.jsonl> <table>   -> LanceDB, frame coordinates
"""
import json, sys
from pathlib import Path

CONF = 0.35
BADGES = {"bar-level", "bar"}


def badge_team(img, box):
    import cv2
    x0, y0, x1, y1 = (int(v) for v in box)
    patch = img[max(y0, 0):y1, max(x0, 0):x1]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 130) & (hsv[..., 2] > 110)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 168)) & sat).mean()
    blue = ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 118) & sat).mean()
    if max(red, blue) < 0.08:
        return None
    return "enemy" if red > blue else "friendly"


def detect(weights, root):
    import cv2
    from ultralytics import YOLO
    root = Path(root)
    model = YOLO(weights)
    man = [json.loads(l) for l in open(root / "manifest.jsonl")]
    out = open(root / "detections.jsonl", "w")
    B = 32
    for i in range(0, len(man), B):
        batch = man[i:i + B]
        imgs = [cv2.imread(str(root / m["file"])) for m in batch]
        res = model.predict(imgs, imgsz=896, conf=CONF, verbose=False, half=True)
        for k, (m, img, r) in enumerate(zip(batch, imgs, res)):
            names = r.names
            raw = [(names[int(c)], float(p), tuple(float(v) for v in b)) for b, c, p in
                   zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            badges = [(b, badge_team(img, b)) for n, _, b in raw if n in BADGES]
            dets = []
            for name, conf, b in raw:
                team = badge_team(img, b) if name in BADGES else None
                if team is None and name not in BADGES:
                    cx, top, w, h = (b[0] + b[2]) / 2, b[1], b[2] - b[0], b[3] - b[1]
                    near = [(abs((bb[0] + bb[2]) / 2 - cx) + abs((bb[1] + bb[3]) / 2 - top), t) for bb, t in badges
                            if t and b[0] - w * 0.3 <= (bb[0] + bb[2]) / 2 <= b[2] + w * 0.3
                            and top - h * 0.6 <= (bb[1] + bb[3]) / 2 <= top + h * 0.5]
                    if near:
                        team = min(near)[1]
                if team is None:
                    team = "enemy" if (b[1] + b[3]) / 2 < 448 else "friendly"
                dets.append([name, team, round(conf, 4), *[round(v, 1) for v in b]])
            out.write(json.dumps({"i": i + k, "dets": dets}) + "\n")
        if i % 3200 == 0:
            print(i, "/", len(man), flush=True)
    out.close()
    print("DETECT_DONE", len(man))


def do_import(det_file, table):
    import lancedb
    import pyarrow as pa
    from bot77.detect.run import SCHEMA
    root = Path(det_file).parent
    man = [json.loads(l) for l in open(root / "manifest.jsonl")]
    rows = []
    for l in open(det_file):
        r = json.loads(l)
        m = man[r["i"]]
        for name, team, conf, x0, y0, x1, y1 in r["dets"]:
            rows.append({"match_id": m["match_id"], "video_id": m["video_id"], "t_video": m["t_video"], "cls": name,
                         "team": team, "conf": conf, "x0": m["ox"] + x0 * m["sx"], "y0": m["oy"] + y0 * m["sy"],
                         "x1": m["ox"] + x1 * m["sx"], "y1": m["oy"] + y1 * m["sy"]})
    db = lancedb.connect("data/lancedb")
    if table in db.table_names(limit=10_000):
        db.drop_table(table)
    db.create_table(table, pa.Table.from_pylist(rows, schema=SCHEMA))
    print("imported", len(rows), "detections into", table)


if __name__ == "__main__":
    if sys.argv[1] == "detect":
        detect(sys.argv[2], sys.argv[3])
    else:
        do_import(sys.argv[2], sys.argv[3])
