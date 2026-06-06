# SNN reset grid search

Each task is a separate script; `run_grid.py` calls those
task scripts and collects summaries under one result folder.

The runner trains each task/seed once, lets the trained state settle, and then
clones that same state across the protocol grid.

## Scripts

| Script | Task |
|--------|------|
| `task_evoked_channel_response.py` | Direct evoked channel response |
| `task_conditioned_electrode_association.py` | A to B conditioned association |
| `task_delayed_conditioned_response.py` | Delayed conditioned response |
| `task_pattern_discrimination.py` | Two-pattern target discrimination |
| `task_temporal_order_discrimination.py` | Same electrodes, different temporal order |
| `run_grid.py` | Central runner that launches the task scripts |
| `figures.py` | Figure generator for completed modular grid outputs |
| `relearning_analysis.py` | Pareto analysis for forgetting versus relearning savings |

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
