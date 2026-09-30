"""Score detector v1 on the opponent benchmark (Alina match 1, 52 plays Cole labelled) and
compare with the 2024 KataCR detector."""
import json
from pathlib import Path

from bot77.detect.run import detect_video
from bot77.detect.yolo import Bot77Detector
from bot77.opponent import opponent_events
from bot77.review import score_opponent

db = Path("data/lancedb")
truth = json.loads(Path("data/ground_truth/opponent_truth_uz4VVzGlOjE_m01.json").read_text())
detect_video("uz4VVzGlOjE", db, matches=[1], detector=Bot77Detector(), table="unit_detections_v1")
opponent_events("uz4VVzGlOjE", db, matches=[1], det_table="unit_detections_v1", out_table="opponent_events_v1")
old = score_opponent(db, truth)
new = score_opponent(db, truth, table="opponent_events_v1")
print(json.dumps({"katacr_2024": old, "bot77_v1": new}, indent=1))
Path("data/logs/eval_detector_v1.json").write_text(json.dumps({"katacr_2024": old, "bot77_v1": new}, indent=1))
