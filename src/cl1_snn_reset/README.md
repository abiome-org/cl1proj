# `cl1_snn_reset`

Installable library for the CL1-style reset simulator: spatial E/I culture, 64-channel
MEA stimulation and recording, colored pulse protocols, train-reset-relearn trials,
metrics, trace probes, and optional batch sweeps.

## Public API

Import from the package root only (no private `_` symbols, no experiment scripts):

```python
from cl1_snn_reset import (
    CultureConfig,
    ExperimentConfig,
    ResetProtocol,
    PhaseSnapshot,
    capture_phase,
    build_trial_artifacts,
    run_trial,
    run_sweep,
    pareto_front,
)
```

Plot helpers live in `cl1_snn_reset.analysis` (`plot_protocol_scatter`,
`plot_pareto_summary`). Learned inverse control lives in `cl1_snn_reset.inverse_control`.

## Reference docs

- [`docs/snn_reset/simulator.md`](../../docs/snn_reset/simulator.md) — the forward
  simulation engine and the train-reset-relearn trial flow.
- [`docs/snn_reset/metrics.md`](../../docs/snn_reset/metrics.md) — metrics, scoring,
  and multi-objective protocol selection.
- [`inverse_control/README.md`](inverse_control/README.md) — learned inverse reset
  control.

## Runnable studies

Grid searches, regression smoke tests, and inverse-reset pipelines are under
`experiments/` in the repository root. They consume this package but are not part
of the wheel.
