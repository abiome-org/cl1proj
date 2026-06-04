# CL SDK Bridge

`cl1_clsdk_bridge` contains thin adapters that connect the reset simulator
library to the vendored CL SDK surrogate twin (`ResetSNNAdapter`, dynamics env
helpers).

Keep SDK-facing wiring here so `cl1_snn_reset` stays a standalone library and
`src/cl/` changes stay minimal. Runnable studies live under `experiments/`.

See [`docs/clsdk_bridge/adapter.md`](../../docs/clsdk_bridge/adapter.md) for the
`ResetSNNAdapter`, the `CL_SDK_SIM_MODE` / `CL_SDK_TWIN_DYNAMICS` selection
contract, and the twin frame/stim mapping.
