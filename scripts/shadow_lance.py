"""Cole's shadow-mode moments as a LanceDB table (2026-11-10): one row per decision of every game scripts/shadow_play.py
recorded, with what Cole did there and what the model guesses, so his plays can be queried and searched by similarity.

  uv run --no-sync python scripts/shadow_lance.py [--ckpt data/models/bc_school_v7/epoch03.pth]
  uv run --no-sync python scripts/shadow_lance.py --similar 2026-11-10_203307 84.3 [--k 10] [--by embed|vector]
  -> data/lancedb/shadow_moments   (rebuilt whole each run, from data/cache/bc/shadow_t01{,_held}/*.npz)

The model's guesses are recomputed offline from the saved observations: each game is one sequence from its first
decision with a fresh LSTM, through train_bc.py's sequence_loss (forward_sequence, bit-identical to the live per-step
loop), so they match what the model would have said live up to the decisions shadow_play didn't record. Columns:
p (5: the 4 slots + wait, softmax of the masked card logits, as the live policy's), p_wait, model_card (top of the 4
slots, as train_bc's card-at-plays), model_cell (the placement head given model_card), model_cell_his (given Cole's
card, at his plays), p_his (the chance it gave his card, wait included), plus `model` (the checkpoint). Slots are
named from the observation's hand one-hots (engine names): hand, cole_card_name, model_card_name; elixir likewise.

Search: `vector` is the raw 13977-dim observation (~99% zeros; the nearest moments by L2 are mostly ones with the same
hand and elixir), `embed` the LSTM's 256-dim state at that decision (what the heads read: the game so far as the model
sees it). --similar searches `embed` by default (cosine), keeping only moments where Cole played, other games first.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bc import BOARD_W, sequence_loss  # noqa: E402  (also puts the fork on sys.path)
from python_ai.models.policy_io import load_net  # noqa: E402

SRC = {False: ROOT / "data/cache/bc/shadow_t01", True: ROOT / "data/cache/bc/shadow_t01_held"}
DB = ROOT / "data/lancedb"
TABLE = "shadow_moments"
OBS_DIM, EMB_DIM = 13977, 256


@torch.no_grad()
def guesses(net, obs: np.ndarray, card: np.ndarray) -> dict:
    """The model over one whole game (fresh LSTM at the first decision)."""
    hx = []
    hook = net.lstm.register_forward_hook(lambda m, i, o: hx.append(o[0]))
    try:
        o = torch.tensor(obs, dtype=torch.float32).unsqueeze(1)  # (T, 1, D)
        c = torch.tensor(card).unsqueeze(1)
        ones = torch.ones_like(c, dtype=torch.bool)
        w = torch.ones_like(c, dtype=torch.float32)
        # pass 1, conditioned on Cole's card: card logits, the LSTM state, the cell it'd pick for his card
        _, _, (cl, pl_his, _, _) = sequence_loss(net, o, c, torch.full_like(c, -1), ones, w)
        emb = torch.stack(hx).squeeze(1)
        p = torch.softmax(cl[:, 0], -1)
        top = cl[:, 0, :net.hand_size].argmax(-1)
        # pass 2, conditioned on its own top card: where it'd put it
        _, _, (_, pl_top, _, _) = sequence_loss(net, o, top.unsqueeze(1), torch.full_like(c, -1), ones, w)
    finally:
        hook.remove()
    return {"p": p.numpy(), "emb": emb.numpy(), "top": top.numpy(), "cell_top": pl_top[:, 0].argmax(-1).numpy(),
            "cell_his": pl_his[:, 0].argmax(-1).numpy()}


def hand_names(net, obs: np.ndarray) -> tuple[list[list[str]], np.ndarray]:
    """The 4 hand cards (engine names, from the observation's one-hots) and elixir at each decision."""
    import clash_royale_env as cre
    sc = obs[:, net.spatial_size:]
    n = cre.ClashRoyaleEnv.NUM_CARD_IDS
    ids = sc[:, 1 + 4:1 + 4 + 4 * n].reshape(-1, 4, n).argmax(-1)
    names = {i: cre.get_card_info(int(i))["name"] for i in np.unique(ids)}
    return [[names[i] for i in r] for r in ids], (sc[:, 0] * 10).astype(np.float32)


def fsl(a: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(pa.array(a.astype(np.float32).ravel()), dim)


def build(a):
    import lancedb
    t0 = time.time()
    net = load_net(a.ckpt, "cpu", verbose=False)
    model = Path(a.ckpt).parent.name + "/" + Path(a.ckpt).name
    tables = []
    for held, d in SRC.items():
        for f in sorted(d.glob("cole_shadow_*.npz")):
            z = np.load(f)
            obs, card, cell = z["obs"].astype(np.float32), z["card"].astype(np.int64), z["cell"].astype(np.int64)
            g = guesses(net, obs, card)
            T, play = len(card), card < 4
            p_his = g["p"][np.arange(T), card]
            hand, elixir = hand_names(net, obs)
            name = lambda k: [h[j] if j < 4 else "wait" for h, j in zip(hand, k)]  # noqa: E731
            tables.append(pa.table({
                "game": [f.stem.removeprefix("cole_shadow_")] * T, "held": [held] * T, "i": np.arange(T, dtype=np.int32),
                "t": z["t"].astype(np.float32), "cole_card": card.astype(np.int8), "cole_cell": cell.astype(np.int16),
                "cole_card_name": name(card), "is_play": play, "hand": hand,
                "elixir": elixir, "result": [str(z["result"])] * T,
                "p": fsl(g["p"], 5), "p_wait": g["p"][:, 4].astype(np.float32),
                "model_card": g["top"].astype(np.int8), "model_card_name": name(g["top"]),
                "model_cell": g["cell_top"].astype(np.int16),
                "model_cell_his": np.where(play, g["cell_his"], -1).astype(np.int16),
                "p_his": p_his.astype(np.float32), "model": [model] * T,
                "vector": fsl(obs, OBS_DIM), "embed": fsl(g["emb"], EMB_DIM)}))
            hit = (g["top"][play] == card[play]).mean() if play.any() else float("nan")
            print(f"  {f.name}: {T} moments, {int(play.sum())} plays, model's top card his {hit:.3f}", flush=True)
    if not tables:
        sys.exit("no shadow games yet")
    tab = pa.concat_tables(tables)
    lancedb.connect(DB).create_table(TABLE, tab, mode="overwrite")
    print(f"{DB / TABLE}: {tab.num_rows} rows from {len(tables)} games ({model}) in {time.time() - t0:.1f} s")


def similar(a):
    import lancedb
    tab = lancedb.connect(DB).open_table(TABLE)
    q = tab.search().where(f"game = '{a.similar[0]}'").select(["i", "t", a.by]).limit(100000).to_arrow().to_pylist()
    if not q:
        sys.exit(f"no game {a.similar[0]}")
    row = min(q, key=lambda r: abs(r["t"] - float(a.similar[1])))
    me = tab.search().where(f"game = '{a.similar[0]}' AND i = {row['i']}").limit(1).to_arrow().to_pylist()[0]
    show(me, "query")
    t0 = time.time()
    hits = (tab.search(row[a.by], vector_column_name=a.by).distance_type("cosine" if a.by == "embed" else "l2")
            .where(f"is_play AND NOT (game = '{me['game']}' AND abs(t - {me['t']}) < 3)", prefilter=True)
            .select(["game", "t", "elixir", "hand", "cole_card", "cole_card_name", "cole_cell", "model_card",
                     "model_card_name", "model_cell", "p", "p_his", "_distance"])
            .limit(a.k).to_arrow().to_pylist())
    print(f"{len(hits)} nearest moments where Cole played (by {a.by}, {(time.time() - t0) * 1e3:.0f} ms):")
    for h in hits:
        show(h, f"{h['_distance']:.3f}")


def cell_str(c):
    return "-" if c < 0 else f"({c % BOARD_W},{c // BOARD_W})"


def show(r, tag):
    his = "wait" if r["cole_card"] == 4 else f"{r['cole_card_name']} @ {cell_str(r['cole_cell'])}"
    print(f"  {tag:>7}  {r['game']} t={r['t']:6.1f} {r['elixir']:4.1f}e [{', '.join(r['hand'])}]  Cole: {his:<24} "
          f"model: {r['model_card_name']} @ {cell_str(r['model_cell']):<8} p_wait={r['p'][4]:.2f} p_his={r['p_his']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/models/bc_school_v7/epoch03.pth", help="the model whose guesses are stored")
    ap.add_argument("--similar", nargs=2, metavar=("GAME", "T"), help="the k nearest moments where Cole played")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--by", choices=["embed", "vector"], default="embed")
    ap.add_argument("--threads", type=int, default=2, help="torch threads (live inference shares this laptop)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    similar(a) if a.similar else build(a)


if __name__ == "__main__":
    main()
