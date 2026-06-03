# SNN Reset Experiments

This folder contains runnable benchmark and sweep entrypoints for the reset
simulator.  The main experiment trains a channel-to-channel response, applies a
colored pulse-reset protocol, then measures hidden weight erasure, behavior,
savings, trace detectability, health, and stimulation cost.

Run the calibrated full 10k-neuron coarse grid:

```bash
.venv-uv/bin/python experiments/snn_reset/full_grid_search.py
```

The benchmark entrypoint is:

```bash
.venv-uv/bin/python experiments/snn_reset/benchmarks/benchmark_snn_reset.py
```

Write generated outputs under `experiments/snn_reset/results/`; that directory
is ignored by Git.

The completed calibrated 10k grid screened 540 protocols across 3 seeds.  See
`docs/snn_reset/full_grid_10k_calibrated_20260602.md` for the result summary.
