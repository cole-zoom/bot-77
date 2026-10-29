"""The defence gate (docs/plan-elixir-spells.md, "Don't lose the defence"): a checkpoint's exam on the 10 defence
templates against the defence school v2 epoch 3's, per template, at the same calibration and n.

  uv run --no-sync python scripts/defence_gate.py NEW_EXAM.txt [--base data/models/bc_defence_v2/exam_epoch03.txt]
      [--col 3] [--base-col 3]

FAIL if the defence-template mean drops more than 0.05 below the base's, or any one template drops more than 0.15.
Reads situation_exam.py's text output (a column per player: nothing, rush, defend, then each --ckpt).
"""
import argparse
import re

DEFENCE = ["hog_rider", "battle_ram", "melee", "supported_push", "arrows_value", "fireball_value", "emergency",
           "safe_opening", "defend_counter", "dont_overcommit"]
MEAN_DROP, ONE_DROP = 0.05, 0.15


def read(path: str, col: int) -> dict[str, float]:
    out = {}
    for line in open(path):
        parts = line.split()
        if parts and parts[0] in DEFENCE:
            vals = re.findall(r"([+-]\d+\.\d+) \(hp", line)
            out[parts[0]] = float(vals[col])
    return out


def gate(new: dict, base: dict) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for t in DEFENCE:
        d = new[t] - base[t]
        bad = d < -ONE_DROP - 1e-9
        ok &= not bad
        lines.append(f"  {t:16s} {base[t]:+.2f} -> {new[t]:+.2f} ({d:+.2f}){'  FAIL (> 0.15 drop)' if bad else ''}")
    mb, mn = sum(base[t] for t in DEFENCE) / len(DEFENCE), sum(new[t] for t in DEFENCE) / len(DEFENCE)
    bad = mn - mb < -MEAN_DROP - 1e-9
    ok &= not bad
    lines.append(f"  {'mean':16s} {mb:+.3f} -> {mn:+.3f} ({mn - mb:+.3f}){'  FAIL (> 0.05 drop)' if bad else ''}")
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("--base", default="data/models/bc_defence_v2/exam_epoch03.txt")
    ap.add_argument("--col", type=int, default=3)
    ap.add_argument("--base-col", type=int, default=3)
    a = ap.parse_args()
    ok, lines = gate(read(a.new, a.col), read(a.base, a.base_col))
    print(f"defence gate: {'PASS' if ok else 'FAIL'}  ({a.new} col {a.col} vs {a.base} col {a.base_col})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
