"""docs/features.md from the feature registry (bot77.features.registry), 2026-10-22.

  uv run python scripts/features_doc.py
"""
from collections import defaultdict
from pathlib import Path

from bot77.features import FEATURES

by = defaultdict(list)
for f in FEATURES:
    by[f.group].append(f)
lines = ["# Features", "", "Every feature the models see, generated from `src/bot77/features/registry.py` by "
         "`scripts/features_doc.py` (tests/test_features.py keeps the registry in step with the code). Level: "
         "`decision` = one value per moment, `card` = per card in hand, `spot` = per candidate spot. All positions are "
         "in the deciding player's frame (`bot77.features.frames`).", "",
         f"{len(FEATURES)} features.", ""]
for g, fs in by.items():
    lines += [f"## {g}", "", "| Feature | Level | Meaning | Since |", "|---|---|---|---|"]
    lines += [f"| `{f.name}` | {f.level} | {f.doc} | {f.since} |" for f in fs] + [""]
Path("docs/features.md").write_text("\n".join(lines))
print(len(FEATURES), "features -> docs/features.md")
