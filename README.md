# CL1 SNN Reset Workspace

This repository vendors the CL SDK plus the experimental SNN reset platform.

## Where The Reset Code Lives

The implementation follows the requested `snn_reset/` outline under the SDK
package namespace so it can be imported by, and wired into, the CL simulator:

| Outline module | Implemented path |
| --- | --- |
| `snn_reset/config.py` | `cl-sdk/src/cl/snn_reset/config.py` |
| `snn_reset/network.py` | `cl-sdk/src/cl/snn_reset/network.py` |
| `snn_reset/electrodes.py` | `cl-sdk/src/cl/snn_reset/electrodes.py` |
| `snn_reset/task.py` | `cl-sdk/src/cl/snn_reset/task.py` |
| `snn_reset/noise.py` | `cl-sdk/src/cl/snn_reset/noise.py` |
| `snn_reset/protocols.py` | `cl-sdk/src/cl/snn_reset/protocols.py` |
| `snn_reset/metrics.py` | `cl-sdk/src/cl/snn_reset/metrics.py` |
| `snn_reset/trace_probe.py` | `cl-sdk/src/cl/snn_reset/trace_probe.py` |
| `snn_reset/experiment.py` | `cl-sdk/src/cl/snn_reset/experiment.py` |
| `snn_reset/sweep.py` | `cl-sdk/src/cl/snn_reset/sweep.py` |
| `snn_reset/analysis.py` | `cl-sdk/src/cl/snn_reset/analysis.py` |

The CL SDK integration is in:

- `cl-sdk/src/cl/twin/reset_adapter.py`
- `cl-sdk/src/cl/twin/surrogate.py`
- `cl-sdk/src/cl/_twin_producer.py`

Run the benchmark with:

```bash
cd cl-sdk
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp/xdg-cache \
  .venv-uv/bin/python scripts/benchmark_snn_reset.py
```

Run the focused reset tests with:

```bash
cd cl-sdk
.venv-uv/bin/pytest tests/cl/test_snn_reset.py -q
```
