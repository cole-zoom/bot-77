"""Train the spell-stage clip classifier on the pod (decisions.md, 2026-10-14).

R(2+1)D-18 (Kinetics-400 weights, torchvision) over 16 arena frames at 10 fps; classes = spells with enough training
clips + "__none". Clips come from scripts/spell_clips.py (JPEG strips of 28 frames starting 0.6 s before the card
left the hand). A positive crop starts 0.4 s before to 0.2 s after the play, so at inference a window starting at s
means a play near s + 0.1.

  python3 train_spell.py --data /root/spell_clips --out /workspace/spell [--epochs 14]
"""
import argparse, json, math, random, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

T = 16
POS_OFF = (2, 8)  # crop start frame for positives (frame 6 = the play)
MIN_TRAIN = 25
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


class Clips(Dataset):
    def __init__(self, root, rows, classes, train):
        self.root, self.rows, self.cls, self.train = Path(root), rows, {c: i for i, c in enumerate(classes)}, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        strip = cv2.imread(str(self.root / "clips" / f"{r['name']}.jpg"))[:, :, ::-1]
        h = strip.shape[0] // 28
        frames = strip.reshape(28, h, strip.shape[1], 3)
        pos = r["label"] != "__none"
        if self.train:
            o = random.randint(*POS_OFF) if pos else random.randint(0, 28 - T)
        else:
            o = 5 if pos else 6
        x = torch.from_numpy(np.ascontiguousarray(frames[o:o + T])).permute(3, 0, 1, 2).float() / 255  # C,T,H,W
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
            x = (x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)).clamp(0, 1)
        return (x - MEAN) / STD, self.cls[r["label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((Path(a.data) / "index.json").read_text())
    rows = idx["clips"]
    n_train = Counter(r["label"] for r in rows if r["split"] == "train")
    classes = ["__none"] + sorted(c for c, n in n_train.items() if c != "__none" and n >= MIN_TRAIN)
    rows = [r for r in rows if r["label"] in classes]
    tr = [r for r in rows if r["split"] == "train"]; va = [r for r in rows if r["split"] == "val"]
    print("classes", classes, "train", len(tr), "val", len(va), flush=True)
    cnt = Counter(r["label"] for r in tr)
    # half the draws "no spell", the rest spread over spells by sqrt of frequency
    spell_mass = sum(math.sqrt(cnt[c]) for c in classes[1:])
    w = [0.5 / cnt["__none"] if r["label"] == "__none" else 0.5 * math.sqrt(cnt[r["label"]]) / spell_mass / cnt[r["label"]]
         for r in tr]
    dl = DataLoader(Clips(a.data, tr, classes, True), batch_size=a.batch, num_workers=a.workers, drop_last=True,
                    sampler=WeightedRandomSampler(w, num_samples=len(tr), replacement=True), persistent_workers=True)
    dv = DataLoader(Clips(a.data, va, classes, False), batch_size=64, num_workers=a.workers)
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.cuda().to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl:
            x, y = x.cuda(non_blocking=True).to(memory_format=torch.channels_last_3d), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        model.eval(); cm = np.zeros((len(classes), len(classes)), int)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in dv:
                p = model(x.cuda().to(memory_format=torch.channels_last_3d)).argmax(1).cpu()
                for a_, b_ in zip(y.tolist(), p.tolist()):
                    cm[a_, b_] += 1
        rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
        prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
        f1 = 2 * rec * prec / np.maximum(rec + prec, 1e-9)
        spell_acc = cm.diagonal()[1:].sum() / max(cm[1:].sum(), 1)  # spell clips named right
        none_fp = cm[0, 1:].sum() / max(cm[0].sum(), 1)  # quiet clips called a spell
        score = f1[1:].mean()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.3f} {time.time() - t0:.0f}s  spell acc {spell_acc:.3f} "
              f"none->spell {none_fp:.3f} macro-F1 {score:.3f}", flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "classes": classes, "T": T, "fps": idx["fps"],
                        "w": idx["w"], "h": idx["h"], "epoch": ep + 1, "macro_f1": float(score)}, out / "best.pt")
            (out / "val_report.json").write_text(json.dumps(
                {"epoch": ep + 1, "classes": classes, "confusion": cm.tolist(), "recall": rec.round(3).tolist(),
                 "precision": prec.round(3).tolist(), "spell_acc": float(spell_acc), "none_fp": float(none_fp)}, indent=1))
    print("TRAIN_DONE best macro-F1", round(best, 3), flush=True)


if __name__ == "__main__":
    main()
