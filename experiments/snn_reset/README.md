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
