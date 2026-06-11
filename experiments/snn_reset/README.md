# SNN reset grid search

`run_grid.py` runs the task suite in-process: for each task it trains every
task/seed once, lets the trained state settle, then clones that same state across
the protocol grid. Per-task results land in one subfolder each and are
aggregated into `all_task_summary.csv`.

## Scripts

| Script | Role |
|--------|------|
| `tasks.py` | All task definitions in one place (`TASK_BUILDERS` registry); run one task with `tasks.py --task <name>` |
| `run_grid.py` | Suite runner: picks tasks, gives each its own output dir, aggregates summaries |
| `common.py` | Shared CLI args, resume logic, and `run_task_grid` (one task's protocol × seed grid) |
| `figlib.py` | Figure presentation support: palette, style, labels, plot helpers (data loading is `cl1_snn_reset.load_suite`) |
| `figures/` | One script per paper figure (see AGENTS.md); each imports `figlib` |
| `relearning_analysis.py` | Pareto analysis for forgetting versus relearning savings |

The five tasks (`evoked_channel_response`, `conditioned_electrode_association`,
`delayed_conditioned_response`, `pattern_discrimination`,
`temporal_order_discrimination`) live as builder functions in `tasks.py`. Their
task-identity choices (criterion, training length, channel wiring) are the
experiment's decisions; the mechanism they configure lives in `cl1_snn_reset`.

## Reading the results

`all_task_summary.csv` has one row per protocol × task (averaged over seeds).
Key columns:

| Column | Meaning |
|--------|---------|
| `baseline_score`, `trained_score` | Task score before / after training (learning check: 0 → 1 for a real learned task) |
| `naive_weight_control_score` | Score after overwriting trained weights with naive weights — the privileged positive control; ~0 confirms the readout is weight-sensitive |
| `reset_score`, `no_reset_score` | Post-protocol score for the reset branch vs the matched plasticity-on rest branch |
| `reset_minus_no_reset_score` | Apparent behavioral reset; ≤ 0 means the protocol did not lower the score below drift |
| `weight_erasure_reset` | 1 − ‖W_reset − W_naive‖ / ‖W_trained − W_naive‖; >0 moves weights toward naive, <0 moves them away |
| `erasure_projection_reset_vs_no_reset` | Fraction of the reset displacement aligned with the training axis |
| `relearn_savings` | 1 − relearn_trials / initial_trials; >0 means relearning was faster (trace persisted), ≤ 0 is genuine functional reset |
| `reset_window_neuron_spikes_delta` | Extra neuron spikes the protocol evokes vs the rest branch (acute perturbation, not erasure) |

## Validated Grid

The current default protocol grid has 60 protocols. Tasks are

- `conditioned_electrode_association`
- `pattern_discrimination`

`evoked_channel_response` is a sensory responsiveness control. The delayed and
temporal-order tasks remain available as task-viability checks, but they were not
part of the validated learned-task grid.

## Smoke Run

```bash
.venv-uv/bin/python experiments/snn_reset/run_grid.py \
  --neurons 192 \
  --seeds 1 \
  --limit-protocols 1 \
  --training-repetitions 1 \
  --eval-repetitions 2 \
  --input-current-uA 120 \
  --target-current-uA 120
```

## Relearning Round

Use `--measure-relearning` to retrain each reset branch after the immediate
post-reset assay and record relearning trials plus savings. The strict front
uses criterion-level forgetting: no-reset at or above criterion and reset below
criterion. A separate score-drop front is also emitted for partial performance
drops that do not cross criterion.

```bash
.venv-uv/bin/python experiments/snn_reset/run_grid.py \
  --tasks conditioned_electrode_association pattern_discrimination \
  --measure-relearning \
  --relearn-repetitions 80
```

`--relearn-only-if-forgot` is an optional screening shortcut. Do not use it for
the final savings run unless the analysis explicitly treats savings as
conditional on criterion-level forgetting.
