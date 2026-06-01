# SNN Reset Simulator

`cl1_snn_reset` is the core experiment package for CL1-style reset screening.

It owns the spatial E/I culture model, MEA stimulation and recording layer,
colored pulse-event reset protocols, train-reset-relearn experiments, metrics,
trace probe utilities, sweep execution, and ranking helpers.

Use this package directly for offline experiments:

```python
from cl1_snn_reset import CultureConfig, ResetProtocol, run_trial
```
