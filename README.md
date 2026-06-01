# CL1 Reset Experiments

This repository is organized around CL1-style SNN reset experiments.  The CL SDK
runtime is vendored as a compatibility layer, while the reset simulator and SDK
bridge live in separate packages.

## Layout

- `src/cl1_snn_reset/`: core SNN reset simulator, including the spatial E/I
  culture model, MEA electrode interface, colored pulse protocols,
  train-reset-relearn loop, metrics, trace probe, and sweep helpers.
- `src/cl1_clsdk_bridge/`: adapters that connect experiment packages to the CL
  SDK simulator runtime.
- `src/cl/`: vendored CL SDK runtime surface.  Keep SDK compatibility work here;
  keep experiment code outside this package.
- `experiments/snn_reset/`: benchmark and sweep entrypoints.
- `tests/snn_reset/`: reset simulator tests.
- `tests/clsdk_bridge/`: SDK bridge tests.
- `docs/snn_reset/` and `docs/clsdk_bridge/`: notes for the local experiment
  packages.

Compatibility imports are kept for existing notebooks and SDK call sites:
`cl.snn_reset` re-exports `cl1_snn_reset`, and `cl.twin.ResetSNNAdapter`
re-exports the adapter from `cl1_clsdk_bridge`.

## Running

Run the benchmark:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp/xdg-cache \
  .venv-uv/bin/python experiments/snn_reset/benchmarks/benchmark_snn_reset.py
```

Run focused reset tests:

```bash
.venv-uv/bin/python -m pytest tests/snn_reset tests/clsdk_bridge -q
```

Enable the reset SNN inside the simulator:

```bash
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_DYNAMICS=snn_reset
```
