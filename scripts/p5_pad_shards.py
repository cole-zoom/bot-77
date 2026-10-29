"""Plan P5: the pro imitation shards with the opponent-elixir input, everything else as school v2 / v3 saw it.

  uv run --no-sync python scripts/p5_pad_shards.py --old data/cache/bc/shards_t01 --new data/cache/bc/shards_p5_t01 \
      --out data/cache/bc/shards_p5pad_t01

shards_t01 (2026-10-27) predates the engine's client stat audit (2026-10-28), so a rebuild today differs in the
spatial channels too, not only in the elixir scalars. To change one thing at a time, this takes shards_t01's
observations and puts in the rebuilt shards' (build_bc_dataset.py on the P5 engine) extra scalars 1-2 (both sides'
spend, now set from the bars) and the new scalar 10 (the opponent's estimate). Rows, labels and weights must match
exactly, or it stops.
"""
import argparse
from pathlib import Path

import numpy as np

import clash_royale_env as cre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = cre.ClashRoyaleEnv.EXTRA_SCALARS_START
    assert cre.ClashRoyaleEnv.NUM_EXTRA_SCALARS == 11, "needs the P5 engine build"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.old).glob("*.npz"))
    for i, f in enumerate(files, 1):
        old, new = np.load(f), np.load(Path(a.new) / f.name)
        for k in old.files:
            if k != "obs" and not np.array_equal(old[k], new[k]):
                raise SystemExit(f"{f.name}: {k} differs between the old and the rebuilt shard")
        o, n = old["obs"], new["obs"]
        assert o.shape[1] + 1 == n.shape[1] and o.shape[0] == n.shape[0]
        obs = np.concatenate([o[:, :E + 10], n[:, E + 10:E + 11], o[:, E + 10:]], axis=1)
        obs[:, E + 1:E + 3] = n[:, E + 1:E + 3]
        assert np.array_equal(obs[:, E + 11:], o[:, E + 10:])
        np.savez_compressed(out / f.name, **{k: (obs if k == "obs" else old[k]) for k in old.files})
        if i % 100 == 0:
            print(f"{i}/{len(files)}", flush=True)
    print(f"{len(files)} shards -> {out}")


if __name__ == "__main__":
    main()
