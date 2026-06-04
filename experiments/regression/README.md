# Regression experiments

Fast checks and the learned inverse-reset controllability pipeline that exercise
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
.venv-uv/bin/python experiments/regression/learned_inverse_reset.py \
    --config experiments/regression/configs/inverse_reset_smoke.yaml
```

Use `benchmark.py --output` to override the sweep CSV path.

---

## Learned inverse-reset pipeline

`learned_inverse_reset.py` orchestrates the library package
`cl1_snn_reset.inverse_control` into one end-to-end study. For what the pipeline
stages do internally, see the module README at
[`src/cl1_snn_reset/inverse_control/README.md`](../../src/cl1_snn_reset/inverse_control/README.md);
this section covers how to *run* it — the CLI, the config schema, and the output
layout.

### What a run does

A `full` run executes four stages in order, emitting one newline-delimited JSON
progress event per stage and writing artifacts as it goes:

1. **dataset** — sample stim programs, roll them through the simulator on top of
   trained states, record causal deltas vs. a no-reset control → `dataset/`.
2. **controllability** — fit the forward delta model, score it, and run the
   reachability/controllability analysis → `models/`, `reports/`.
3. **optimize** — search the program space against the inverse objective for
   ranked candidate protocols → `candidates/`.
4. **validate** — re-run top candidates through the full simulator paired
   against no-reset, with optional bootstrap effect sizes → `validation/`,
   `reports/inverse_reset_report.md`.

`metadata.json` (config + git/dependency/env provenance + diagnostics) is
written at start and refreshed after each stage.

### CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--config` | *(required)* | path to a preset YAML |
| `--mode` | `full` | one of `dataset`, `controllability`, `optimize`, `validate`, `full` |
| `--output-dir` | `<output_dir>/<run-id>` | override the run directory |
| `--run-id` | UTC timestamp | name the run (becomes the results subdir) |
| `--dataset-dir` | run dir | reuse a prior `dataset/` (for partial modes) |
| `--model-path` | run dir | reuse a prior fitted model |
| `--candidates-path` | run dir | reuse prior candidates (for `validate`) |

Single-stage modes reload upstream inputs from disk, so you can run
`dataset` → `controllability` → `optimize` → `validate` separately, or re-run a
later stage against a cached earlier one.

### Presets

All presets share the schema below and point `run.output_dir` at
`experiments/regression/results`. They differ mainly in scale and the question
they probe:

| Config | Scale / optimizer | Purpose |
|--------|-------------------|---------|
| `inverse_reset_smoke.yaml` | tiny (12 programs/state, CMA-ES, 48 evals) | fast CI gate exercising the whole chain |
| `inverse_reset_mvp.yaml` | 10k-neuron net, 500 programs/state, `ridge_then_ensemble`, 1000 bootstrap samples | the reference "real" run with statistics |
| `inverse_reset_controllability.yaml` | 800 programs/state, CMA-ES | the controllability regression |
| `inverse_reset_reachability.yaml` | 28 programs/state, random search | reachability probe (which state dims are attainable) |
| `inverse_reset_actuator_causality.yaml` | 6 programs/state, random search | actuator → state causality check |
| `inverse_reset_closed_loop.yaml` | 500 programs/state, CMA-ES | closed-loop / MPC-style setup |

### Config schema

| Block | Controls |
|-------|----------|
| `run` | name, `output_dir`, `random_seed`, optional `resume`, `warmup_s` |
| `network` | culture size and backend (`n_neurons`, degrees, `n_electrodes`, `backend`) |
| `task` | input/target channel, learning criterion, train/relearn/eval trial counts |
| `seeds` | `train` and `heldout` seed lists (heldout used for validation) |
| `state` | projector + feature options (observable/privileged, normalization, PCA dims) |
| `target` | target `mode` (e.g. `trace_removed`), task projection, health source |
| `stim_sampling` | blocks to include, amplitude/duration/delay grids, programs per state |
| `constraints` | hardware safety limits (amplitude, charge, cooldown, energy, electrodes) |
| `model` | forward-model type, train/val/test split, feature caps, ridge alphas |
| `optimizer` | search `type` (`random` / `cma_es`), candidates per state, eval budget |
| `loss_weights` | per-term weights for the inverse objective (optional override) |
| `validation` | paired no-reset, bootstrap samples, candidate limit |

The driver hard-fails if a `validation` block enables an unsupported control
(`energy_matched_random`, `untrained_control`, `high_dose_positive_control`) —
these keys exist in presets but are declared off, so a run cannot silently claim
a control it did not execute.

### Inverse-reset output layout

Each run writes a self-contained directory under `results/<run-id>/`:

```
metadata.json                       # config + provenance + diagnostics
dataset/
  examples.csv                      # one row per rollout example
  states.npz                        # state vectors
  stim_programs.jsonl               # the sampled programs
  state_vector_spec.json            # feature names / groups / masks
models/linear_delta_model.pkl       # fitted forward model
reports/
  controllability_report.md         # human-readable controllability writeup
  controllability_summary.json
  controllability_arrays.npz        # Jacobian / reachable component
  inverse_reset_report.md           # final ranked-candidate report
candidates/optimized_protocols.csv  # ranked CandidateProtocols (+ .jsonl)
validation/
  candidate_validation.csv
  paired_no_reset_validation.csv    # candidate vs. no-reset control
  bootstrap_candidate_effects.csv   # when validation.bootstrap_samples > 0
```

---

## Results

Outputs are written under `experiments/regression/results/` and are tracked in
git (per the repository rule never to ignore experiment results or generated
reports):

- `benchmark_sweep.csv` — per-job sweep rows from the benchmark
- `benchmark_summary.json` — machine metadata, timing, parallel speedup, top protocols
- `inverse_reset_*/` — learned inverse-reset datasets, forward models, controllability reports, candidates, and validation tables

## CI

`smoke.py` and the `inverse_reset_smoke.yaml` preset are suitable for a quick
gate. `benchmark.py` is heavier and is intended for local perf regression, not
every commit.

## Tests

Library-level unit tests for the inverse-reset package live in
[`tests/snn_reset/inverse_control/`](../../tests/snn_reset/inverse_control/),
one module per pipeline component.
