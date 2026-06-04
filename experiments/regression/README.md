# Regression experiments

Fast checks that the installable libraries still work. These scripts only import
public APIs from `cl1_snn_reset` and `cl1_clsdk_bridge`; they do not modify
anything under `src/`.

## Scripts

| Script | Purpose |
|--------|---------|
| `smoke.py` | One small train-reset-relearn trial plus a 64-channel `ResetSNNAdapter` render |
| `benchmark.py` | Network build/advance timing and a short protocol×seed sweep with optional parallelism |

## Running

```bash
.venv-uv/bin/python experiments/regression/smoke.py
.venv-uv/bin/python experiments/regression/benchmark.py
```

## Results

Outputs are written under `experiments/regression/results/` (gitignored):

- `benchmark_sweep.csv` — per-job sweep rows from the benchmark
- `benchmark_summary.json` — machine metadata, timing, parallel speedup, top protocols

Use `benchmark.py --output` to override the sweep CSV path.

## CI

`smoke.py` is suitable for a quick gate. `benchmark.py` is heavier and is intended
for local perf regression, not every commit.
