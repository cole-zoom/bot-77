"""The models registry from the command line (bot77.models, 2026-10-22).

  uv run python scripts/models.py list [--role card]
  uv run python scripts/models.py show card:moves_v1
  uv run python scripts/models.py adopt card:moves_v1     (the bot loads the adopted model of each role)
"""
import argparse, json

from bot77.models import adopt_model, list_models

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["list", "show", "adopt"]); ap.add_argument("model_id", nargs="?")
ap.add_argument("--role")
a = ap.parse_args()
if a.cmd == "list":
    for r in list_models(a.role):
        m = json.loads(r["metrics"] or "{}")
        key = ", ".join(f"{k} {v}" for k, v in list(m.items())[:2])
        print(f"{'*' if r['adopted'] else ' '} {r['model_id']:32s} {r['role']:16s} {r['status']:12s} {r['trained_at']}  {key}")
    print("* = adopted for its role")
elif a.cmd == "show":
    r = next(x for x in list_models() if x["model_id"] == a.model_id)
    for k, v in r.items():
        print(f"{k:14s} {v}")
else:
    adopt_model(a.model_id)
    print("adopted", a.model_id)
