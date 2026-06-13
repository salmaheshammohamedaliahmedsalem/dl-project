# Best Model Handoff - CMR Multi Challenge

This is the GitHub-safe handoff file for the current best model package.

Large binaries are **not** committed here because the checkpoint files exceed GitHub's normal 100 MB file limit:

- Task 1 checkpoint: `182.72 MB`
- Task 2 checkpoint: `220.88 MB`

Use the local share folder for the full artifact package:

`/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT/best model`

## Recommended Submission

Upload this zip first:

`/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT/SUBMISSIONS_ORGANIZED/UPLOAD_THIS/TRY_TASK1_LOCALBEST_TASK2_LGE_SAVED_BEST.zip`

The same zip is copied inside the share folder:

`/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT/best model/outputs/submissions/RECOMMENDED_UPLOAD_TRY_TASK1_LOCALBEST_TASK2_LGE_SAVED_BEST.zip`

It contains:

- Task 1 Cine: local-best selected masks from the strongest trained Task 1 candidates.
- Task 2 LGE: predictions from the saved improved quick LGE model.
- Verified CodaBench structure and prediction shapes.

## Best Checkpoints

Task 1 primary checkpoint in the share package:

`/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT/best model/checkpoints/task1_ovr256_epoch1_best_model.pth`

Task 2 primary checkpoint in the share package:

`/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT/best model/checkpoints/task2_lge_saved_best_model.pth`

## Architecture

Core model:

`MultiHeadResUNetPP2D`

Source:

`best model/source_code/model_2d_resunetpp_multihead.py`

The model is a 2.5D multi-head ResUNet++ style segmentation network:

- Input uses three adjacent slices as channels: previous slice, current slice, next slice.
- A shared encoder extracts features across views.
- Separate decoder heads handle incompatible label spaces for `2CH`, `4CH`, `SAX`, and `RAS`.
- Residual convolution blocks stabilize optimization.
- Squeeze-and-Excitation blocks recalibrate channels.
- ASPP bottleneck captures multi-scale context.
- Attention-gated skip connections suppress irrelevant encoder features.
- One-vs-rest foreground logits are decoded into integer label maps with deterministic postprocessing.

Task 1 heads:

- `2ch`: background plus LV cavity and LV myocardium.
- `4ch`: background plus LV cavity, LV myocardium, RV cavity, RA, LA.
- `sax`: background plus LV myocardium, LV cavity, RV cavity.

Task 2 heads:

- `2ch`: background plus LV cavity, LV myocardium, scar.
- `4ch`: background plus LV cavity, LV myocardium, scar, RV cavity.
- `sax`: background plus LV cavity, LV myocardium, scar, RV cavity.
- `ras`: background plus RA.

## Training Setup

Task 1 training script:

`src/cmr_models/train_2d_task1.py`

Task 2 training script:

`src/cmr_models/train_2d_task2_lge.py`

Important training choices:

- Multi-source round-robin loader to balance view-specific batches.
- Volume-level train/validation split to avoid slice leakage.
- Weighted sampling to increase foreground and rare-class examples.
- BCE + Dice + focal-style losses.
- Shape-preserving inference and deterministic view-specific decoding.

## Local Metrics Snapshot

Task 1 local-best package:

- Local Task 1 macro Dice: `0.8975`
- 2CH Dice: `0.8883`
- 4CH Dice: `0.9015`
- SAX Dice: `0.9027`

Task 2 saved improved package:

- Local Task 2 mean Dice: `0.6639`
- 2CH Dice: `0.5792`
- 4CH Dice: `0.7157`
- SAX Dice: `0.5623`
- RAS Dice: `0.7986`

## Artifact Manifest

The full local share folder includes hashes in:

`/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT/best model/artifact_manifest.json`

Key hashes:

- `checkpoints/task1_ovr256_epoch1_best_model.pth`: `6d47e9c21a7bfee1...`
- `checkpoints/task2_lge_saved_best_model.pth`: `aec4d73be35c3612...`
- `outputs/submissions/RECOMMENDED_UPLOAD_TRY_TASK1_LOCALBEST_TASK2_LGE_SAVED_BEST.zip`: `6fdcd62a9a80cb54...`

## Rebuild Recommended Submission

From `/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT`:

```bash
python3.11 CMR_multi_model/package_task2_lge_quick_submission.py \
  --checkpoint OUTPUTS_ORGANIZED/outputs/task2_lge_quick_ovr2d/best_model.pth \
  --task1-source-zip SUBMISSIONS_ORGANIZED/UPLOAD_THIS/TRY_TASK1_LOCALBEST_SELECT_TASK2_LGE.zip \
  --zip-path SUBMISSIONS_ORGANIZED/UPLOAD_THIS/TRY_TASK1_LOCALBEST_TASK2_LGE_SAVED_BEST.zip \
  --device cpu
```

To rebuild the local-best Task 1 selection:

```bash
python3.11 CMR_multi_model/build_task1_localbest_task2_lge_submission.py
```

## Current Live Experiment

A safer Task 1 fine-tune may still be running from:

`/Users/salmaheshamsalem/Desktop/DEEP_LEARNING_PROJECT/OUTPUTS_ORGANIZED/outputs/task1_ovr256_safe_finetune/best_model.pth`

Use it only if validation beats the protected Task 1 checkpoint above.
