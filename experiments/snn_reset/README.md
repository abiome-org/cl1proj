# SNN Reset Experiments

This folder contains runnable benchmark and sweep entrypoints for the reset
simulator.

The benchmark entrypoint is:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp/xdg-cache \
  .venv-uv/bin/python experiments/snn_reset/benchmarks/benchmark_snn_reset.py
```

Write generated outputs under `experiments/snn_reset/results/`; that directory
is ignored by Git.
