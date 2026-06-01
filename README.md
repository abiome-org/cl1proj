# CL1 SNN Reset Workspace

This repository vendors the CL SDK plus an experimental SNN reset screening
platform for CL1-style multielectrode-array simulations.

The reset platform is packaged with the SDK so it can run both as an offline
sweep tool and as a selectable biological twin backend.

## Components

- `cl-sdk/src/cl/snn_reset/`: spatial E/I culture model, electrode interface,
  colored stimulation protocols, train-reset-relearn experiments, metrics,
  trace probe utilities, sweep execution, and analysis helpers.
- `cl-sdk/src/cl/twin/`: biological twin components used by the SDK simulator.
- `cl-sdk/scripts/benchmark_snn_reset.py`: benchmark and screening entrypoint.
- `cl-sdk/tests/cl/test_snn_reset.py`: focused reset-platform tests.

## Running

Run the benchmark:

```bash
cd cl-sdk
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp/xdg-cache \
  .venv-uv/bin/python scripts/benchmark_snn_reset.py
```

Run focused reset tests:

```bash
cd cl-sdk
.venv-uv/bin/pytest tests/cl/test_snn_reset.py -q
```

Enable the reset SNN inside the simulator:

```bash
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_DYNAMICS=snn_reset
```
