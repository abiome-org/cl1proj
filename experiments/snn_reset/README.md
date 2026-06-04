# SNN reset research experiments

Large studies for protocol screening and learned inverse reset. These scripts
import `cl1_snn_reset` (and reporting helpers from `cl1_snn_reset.inverse_control`
where noted) and write artifacts under `results/` in this directory. They do not
modify `src/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `full_grid_search.py` | Calibrated coarse protocol grid (e.g. 10k neurons, many protocols × seeds) |
| `control_checks.py` | Reset vs no-reset and noise/actuator diagnostics on representative protocols |
| `learned_inverse_reset.py` | Dataset → forward model → stim-program optimization → validation |
| `configs/*.yaml` | Inverse-reset run presets (`output_dir` points at `experiments/snn_reset/results`) |

## Running

```bash
.venv-uv/bin/python experiments/snn_reset/full_grid_search.py
.venv-uv/bin/python experiments/snn_reset/control_checks.py
.venv-uv/bin/python experiments/snn_reset/learned_inverse_reset.py --config experiments/snn_reset/configs/inverse_reset_smoke.yaml
```

Use `--resume` on `full_grid_search.py` to continue an interrupted grid into the same
`results/<run_id>/` folder.

## Results

All generated outputs live in `experiments/snn_reset/results/` (gitignored). A typical
full-grid run produces:

| File | Contents |
|------|----------|
| `metadata.json` | Git commit, machine info, grid definition, job counts |
| `raw_trials.csv` | One row per protocol×seed trial (resumable) |
| `summary.csv` | Protocol-level aggregates across seeds |
| `ranked.csv` | Scalar `reset_score` sort for quick inspection |
| `pareto.csv` | Nondominated protocols on erasure/health/cost objectives |
| `report.md` | Human-readable summary of the run |

Inverse-reset runs add `dataset/`, `models/`, `candidates/`, and validation tables
under a timestamped subdirectory of `results/`.

Published notes for the calibrated 10k grid are in
`docs/snn_reset/full_grid_10k_calibrated_20260602.md`.

## Regression

Fast smoke and benchmark entrypoints moved to `experiments/regression/`.
