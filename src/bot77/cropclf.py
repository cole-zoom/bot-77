"""Score crops with the classifier head trained on our verified labels + KataCR's crops
(`scripts/crop_classifier.py`). Use it to *filter and rank* review pages, never to accept labels on its own.

Tested against Cole's 234 candidate verdicts with every candidate frame held out of training (2026-10-02):
keeping p(entity) >= 0.3 cut the pages to 42 frames, wrong boxes went from 40% to 14% and right ones from 18% to
60%, but 17 of the 42 frames Cole approved were dropped. Entities with no training crops get p = None, so a caller
can keep them rather than drop them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MODEL = Path("data/models/crop_clf.pt")
REVIEW_MIN_P = 0.3


class CropClassifier:
    def __init__(self, path: Path = MODEL, emb=None):
        from bot77.mine import Embedder

        ck = torch.load(path)
        self.classes = ck["classes"]
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        self.scale = ck.get("scale", 20.0)
        n = len(self.classes)
        self.head = torch.nn.Sequential(torch.nn.Linear(384, 512), torch.nn.GELU(), torch.nn.Dropout(0.3),
                                        torch.nn.Linear(512, n))
        self.head.load_state_dict(ck["state"]); self.head.eval()
        self.emb = emb or Embedder()

    @torch.no_grad()
    def probs(self, crops: list[np.ndarray]) -> np.ndarray:
        e = np.asarray(self.emb(crops), np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return torch.softmax(self.head(torch.tensor(e) * self.scale), 1).numpy()

    def p_entity(self, crop: np.ndarray, entity: str) -> float | None:
        if entity not in self.cidx or crop.size == 0:
            return None
        return float(self.probs([crop])[0, self.cidx[entity]])
