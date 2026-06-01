# CL SDK Bridge Notes

The bridge package is implemented in `src/cl1_clsdk_bridge`.

It currently exposes `ResetSNNAdapter`, which lets the CL SDK simulator select
the reset SNN backend through:

```bash
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_DYNAMICS=snn_reset
```

Compatibility re-exports remain available from `cl.twin.ResetSNNAdapter`.
