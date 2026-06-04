# SNN reset research experiments

Large studies for protocol screening and reset control diagnostics. These
scripts import `cl1_snn_reset` and write artifacts under `results/` in this
directory. They do not modify `src/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `full_grid_search.py` | Calibrated coarse protocol grid (e.g. 10k neurons, many protocols × seeds) |
| `control_checks.py` | Reset vs no-reset and noise/actuator diagnostics on representative protocols |

## Running

```bash
.venv-uv/bin/python experiments/snn_reset/full_grid_search.py
.venv-uv/bin/python experiments/snn_reset/control_checks.py
```

Use `--resume` on `full_grid_search.py` to continue an interrupted grid into the same
`results/<run_id>/` folder.

## Results

Generated outputs live in `experiments/snn_reset/results/`. A typical full-grid
run produces:

| File | Contents |
|------|----------|
| `metadata.json` | Git commit, machine info, grid definition, job counts |
| `raw_trials.csv` | One row per protocol×seed trial (resumable) |
| `summary.csv` | Protocol-level aggregates across seeds |
| `ranked.csv` | Scalar `reset_score` sort for quick inspection |
| `pareto.csv` | Nondominated protocols on erasure/health/cost objectives |
| `report.md` | Human-readable summary of the run |

Published notes for the calibrated 10k grid are in
[`docs/snn_reset/full_grid_10k_calibrated_20260602.md`](../../docs/snn_reset/full_grid_10k_calibrated_20260602.md).
The control-checks study is documented in
[`docs/snn_reset/control_checks.md`](../../docs/snn_reset/control_checks.md).

## Regression

Fast smoke, benchmark, and learned inverse-reset controllability entrypoints live
in `experiments/regression1/`.
