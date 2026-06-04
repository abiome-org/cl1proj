# CL SDK Bridge

`cl1_clsdk_bridge` contains thin adapters that connect the reset simulator
library to the vendored CL SDK surrogate twin (`ResetSNNAdapter`, dynamics env
helpers).

Keep SDK-facing wiring here so `cl1_snn_reset` stays a standalone library and
`src/cl/` changes stay minimal. Runnable studies live under `experiments/`.
