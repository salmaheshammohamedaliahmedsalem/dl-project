# CMR Competition Presentation Package

## Main Files

- `CMR_Competition_Model_Report.ipynb` — notebook with methodology, architecture, results, graphs, and inference overlays.
- `submissions/FINAL_MAIN_TASK1_TASK2.zip` — fresh final upload candidate.
- `reports/task1_selection_report.json` — Task 1 local-selection details and metrics.
- `training_logs/` — raw training logs used for curves.
- `notebook_assets/` — generated figures and qualitative inference overlays.

## Current Recommendation

Use `submissions/FINAL_MAIN_TASK1_TASK2.zip` as the clean final package. It rebuilds the current best-scoring strategy:

- Task 1: local-selected mask package from strongest trained candidates.
- Task 2: saved LGE model inference.

Known public score group for this strategy: `0.64`.
