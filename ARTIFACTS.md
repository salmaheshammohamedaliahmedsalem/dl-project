# Artifact Contents

This repo contains the shareable, GitHub-safe files for the current CMR Multi challenge model package.

## Included in Git

- `README.md`: model handoff, architecture, metrics, and rebuild commands.
- `source_code/`: model architecture and training scripts with neutral filenames.
- `submissions/recommended_submission.zip`: current recommended CodaBench upload file.
- `reports/`: local validation metrics snapshots.
- `training_logs/`: training logs for Task 1 and Task 2.
- `logs/`: live experiment log snapshot.
- `checkpoints/README.md`: where to get the model weights.

## Included as GitHub Release Assets

- `task1_best_model.pth`
- `task2_best_model.pth`

Reason: each checkpoint is larger than GitHub's normal 100 MB Git object limit.
