# Experiments

Runnable studies for this repository. Each subdirectory has its own `README.md`
describing scripts and where results are written.

| Directory | Role |
|-----------|------|
| [`regression/`](regression/) | Fast smoke and benchmark gates (public library API only) |
| [`snn_reset/`](snn_reset/) | Protocol grids, control checks, learned inverse reset |

## Rules

- Do not edit `src/` from experiment work; change the library in a normal code change.
- Import only public symbols from `cl1_snn_reset` / `cl1_clsdk_bridge` (package root or documented subpackages such as `inverse_control`). No private `_` names.
- Write outputs only under `experiments/<name>/results/` (gitignored).

Library unit tests remain in `tests/` at the repository root.
