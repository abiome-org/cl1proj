# SNN Reset Experiments

This folder contains runnable benchmark and sweep entrypoints for the reset
simulator.  The main experiment trains a channel-to-channel response, applies a
colored pulse-reset protocol, then measures hidden weight erasure, behavior,
savings, trace detectability, health, and stimulation cost.

Run the calibrated full 10k-neuron coarse grid:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp/xdg-cache \
  .venv-uv/bin/python experiments/snn_reset/full_grid_search.py
```

The benchmark entrypoint is:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp/xdg-cache \
  .venv-uv/bin/python experiments/snn_reset/benchmarks/benchmark_snn_reset.py
```

Write generated outputs under `experiments/snn_reset/results/`; that directory
is ignored by Git.

The completed calibrated 10k grid ran 540 protocols across 3 seeds on an Apple
M4 Max with 64 GiB unified memory.  It completed 1,620 jobs in 7,103.18 seconds
with 8 process workers, then replicated in a neutral temp checkout in 7,083.15
seconds.  Aggregate outputs matched exactly.  See
`docs/snn_reset/full_grid_10k_calibrated_20260602.md` for the full report.
