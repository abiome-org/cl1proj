# Vendored SDK Notes

The vendored SDK runtime lives in `src/cl`.

Files in this package should be treated as the CL API and simulator support
surface.  Experiment-specific models, sweeps, metrics, and CL1 reset screening
code should live outside `src/cl` unless a small compatibility shim is needed.
