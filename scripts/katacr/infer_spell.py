"""Spell-stage inference over whole-game streams (scripts/spell_streams.py) on the pod: class probabilities for a
16-frame window starting every STRIDE frames. Peak picking and merging with the troop plays happen locally
(bot77.spells), so thresholds can be tuned without the GPU.

  python3 infer_spell.py --model /workspace/spell/best.pt --streams /root/spell_streams --out /workspace/spell_probs
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

STRIDE = 3
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).cuda()
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--streams", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu", weights_only=False)  # our own checkpoint
    model = r2plus1d_18(); model.fc = nn.Linear(model.fc.in_features, len(ck["classes"]))
    model.load_state_dict(ck["state_dict"]); model = model.cuda().eval().to(memory_format=torch.channels_last_3d)
    T, out = ck["T"], Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for d in sorted(Path(a.streams).iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        if (out / f"{meta['match_id']}.npz").exists():
            continue
        fr = np.concatenate([cv2.imread(str(p))[:, :, ::-1] for p in sorted(d.glob("*.jpg"))])
        h = ck["h"]; fr = fr.reshape(-1, h, fr.shape[1], 3)[:meta["n"]]
        x_all = torch.from_numpy(np.ascontiguousarray(fr)).cuda().permute(3, 0, 1, 2)  # C,N,H,W uint8
        starts = list(range(0, len(fr) - T + 1, STRIDE))
        probs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(0, len(starts), 64):
                x = torch.stack([x_all[:, s:s + T] for s in starts[b:b + 64]]).float() / 255
                x = ((x - MEAN) / STD).to(memory_format=torch.channels_last_3d)
                probs.append(model(x).float().softmax(1).cpu().numpy())
        t = meta["t_start"] + np.array(starts) / meta["fps"]
        np.savez_compressed(out / f"{meta['match_id']}.npz", t=t, p=np.concatenate(probs), classes=np.array(ck["classes"]),
                            set=meta["set"])
        print(meta["match_id"], len(starts), "windows", flush=True)
    print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
