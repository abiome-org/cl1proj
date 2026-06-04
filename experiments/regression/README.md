# Regression experiments

Fast checks and learned inverse-reset controllability experiments that exercise
the installable libraries. These scripts only import public APIs from
`cl1_snn_reset` and `cl1_clsdk_bridge`; they do not modify anything under
`src/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `smoke.py` | One small train-reset-relearn trial plus a 64-channel `ResetSNNAdapter` render |
| `benchmark.py` | Network build/advance timing and a short protocol×seed sweep with optional parallelism |
| `learned_inverse_reset.py` | Dataset → forward model → controllability analysis → stim-program optimization → validation |
| `configs/*.yaml` | Learned inverse-reset run presets (`output_dir` points at `experiments/regression/results`) |

## Running

```bash
.venv-uv/bin/python experiments/regression/smoke.py
.venv-uv/bin/python experiments/regression/benchmark.py
.venv-uv/bin/python experiments/regression/learned_inverse_reset.py --config experiments/regression/configs/inverse_reset_smoke.yaml
```

## Results

Outputs are written under `experiments/regression/results/`:

- `benchmark_sweep.csv` — per-job sweep rows from the benchmark
- `benchmark_summary.json` — machine metadata, timing, parallel speedup, top protocols
- `inverse_reset_*/` — learned inverse-reset datasets, forward models, controllability reports, candidates, and validation tables

Use `benchmark.py --output` to override the sweep CSV path.

## CI

`smoke.py` is suitable for a quick gate. `benchmark.py` is heavier and is intended
for local perf regression, not every commit.
