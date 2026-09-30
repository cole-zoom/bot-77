# KataCR (third-party, MIT)

From [KataCR](https://github.com/wty-yy/KataCR) by Tianyang Wu — see `LICENSE` (MIT) — and the
paper [Wu et al. 2025, arXiv:2504.04783](https://arxiv.org/abs/2504.04783) (`docs/references.bib`).

What bot-77 uses:
- **Detector weights** `detector1_v0.7.13` / `detector2_v0.7.13` (YOLOv8, 2024-05-01), downloaded from
  the links in KataCR's README and converted once to tensor-only files
  (`data/models/katacr/*.safe.pt`, not in git) so they load with `torch.load(weights_only=True)`.
  Before converting, the original pickles were checked to reference only torch, ultralytics and
  katacr classes.
- **The detection output format and NMS**: the last "class" channel is the unit's faction
  (friendly / enemy); `bot77.detect.units` ports the logic of `katacr/yolov8/custom_utils.py:
  non_max_suppression` and the two-detector merge of `katacr/yolov8/combo_detect.py`.
- **The class list** (`label_list.py`), read from the weights' `names`.
- **The arena crop** the detector was trained on (`split_bbox_params['part2_2.16']`, 576×896).
