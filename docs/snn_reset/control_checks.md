# Reset control checks

Companion study to the calibrated full grid
(`docs/snn_reset/full_grid_10k_calibrated_20260602.md`). Where the grid screens
many protocols for the best reset score, this study takes three representative
protocols and runs the controls needed to decide whether any apparent reset
effect is real: reset versus no-reset, trained versus untrained cultures, and
noise/actuator diagnostics that characterize the stimulation channel itself.

Script: `experiments/snn_reset/control_checks.py`. It imports `cl1_snn_reset`
and writes artifacts under `experiments/snn_reset/results/`. It does not modify
`src/`.

## Scientific question

The full grid reported negative weight-erasure (post-reset weights moved
*farther* from the naive state), so before attributing any of that to the reset
protocol we need controls that separate three confounds:

1. **Protocol vs. free-running drift.** A trained culture that simply keeps
   running with plasticity on will drift in weight space on its own. The
   reset-vs-no-reset arm holds everything fixed except the stimulation pulses,
   so any difference is attributable to the protocol rather than to elapsed
   time under spontaneous activity.
2. **Erasing learning vs. perturbing a network.** The untrained arm applies the
   same protocols to a culture that never learned the task. If "reset" metrics
   look similar on a naive network, the protocol is perturbing generic network
   state, not undoing task-specific learning.
3. **Actuator fidelity.** The noise diagnostics confirm the colored-noise
   generator produces the requested spectral slope and quantify the per-spatial
   mode pulse burden, so downstream effects (or their absence) can be read
   against a known actuator, not an uncharacterized one.

## Protocols under test

Three fixed protocols chosen to span the grid's burden range
(`selected_protocols()`):

| `protocol_id`      | beta | duration | current | schedule      | spatial mode  |
| ------------------ | ---: | -------: | ------: | ------------- | ------------- |
| `low_burden_0.75s` |    2 |    0.75s |   0.8uA | `epoch_pause` | `independent` |
| `mid_burden_1.5s`  |    0 |    1.5s  |   0.8uA | `static`      | `shared`      |
| `high_burden_3s`   |   -2 |    3.0s  |   2.6uA | `static`      | `shared`      |

## Control arms

Each arm builds a fresh 10k-neuron network per seed (`build_workers=1` inside
the worker), warms up for `warmup_s`, then captures phase snapshots and weight
norms. The `reset` and `no_reset` modes share the same code path; the only
difference is what happens during the protocol window:

- `reset` calls `apply_reset_protocol`, delivering the stimulation pulses
  (seeded at `seed + 10_000`).
- `no_reset` calls `net.advance(duration_s, [], plasticity=True, record=True)` —
  the network free-runs for the same wall-clock duration with plasticity on but
  zero stimulation pulses.

### Trained reset vs. no-reset (`run_trained_control`)

Baseline snapshot → `train_to_criterion` → trained snapshot → protocol window →
post snapshot → `evaluate_task` → `train_to_criterion` (relearn). Per
seed×protocol it computes the standard trial metrics (`compute_trial_metrics`)
plus weight-norm deltas (`trained_delta_norm`, `post_delta_norm`,
`post_minus_trained_norm`, `baseline_weight_norm`) and
`reset_window_neuron_spikes`. `summarize_reset_vs_no_reset` then inner-joins the
two modes on `(source_protocol_id, seed)` and emits per-metric
`*_reset`, `*_no_reset`, and `*_delta` columns for `weight_erasure`,
`residual_performance`, `savings`, `trace_auc_proxy`, `health`,
`post_delta_norm`, and `reset_window_neuron_spikes`.

### Untrained reset controls (`run_untrained_control`)

Same protocols and modes applied to a culture that is **never** trained.
Captures baseline and post snapshots plus `evaluate_task` before and after, and
reports `weight_drift_norm`, `weight_drift_rel_to_baseline`, `pre_behavior`,
`post_behavior`, path strength (`path0`, `path_post`, `path_delta`), and
`trace_auc_proxy`. This is the "does the protocol do anything to a naive
network" baseline.

### Noise / actuator diagnostics (`noise_diagnostics`)

Independent of the cultures. For each `beta` in `[-2, -1, 0, 1, 2]` it draws
`colored_noise` repeats and fits a power-law slope (`estimate_power_slope` via
`np.fft.rfft`, log-log polyfit over the 2%–85% frequency band) to check the
generator hits the target slope `-beta`. For each beta it also expands every
`spatial_mode` (`shared`, `independent`, `correlated`, `phase_shifted`) over 64
channels and counts `protocol_events`: `event_count`, `total_pulses`,
`active_channels`, and `mean_pulses_per_active_channel`.

## CLI

Run with the project venv. There is no `--resume` flag (the full-grid script
has one; this script does not — each run writes a self-contained directory):

```bash
.venv-uv/bin/python experiments/snn_reset/control_checks.py
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--neurons` | `10000` | Culture size |
| `--mean-out-degree` | `64` | Mean synaptic out-degree |
| `--max-out-degree` | `96` | Max synaptic out-degree |
| `--local-candidate-multiplier` | `6` | Local connectivity candidate pool multiplier |
| `--build-workers` | `1` | Workers for network construction (forced to 1 inside each job) |
| `--spontaneous-rate-hz` | `0.12` | Background spontaneous firing rate |
| `--seeds` | `1 3 4` | Network seeds (nargs `+`) |
| `--workers` | `1` | Process pool size for the trial arms (`ProcessPoolExecutor`); `<=1` runs serially |
| `--input-channels` | `8` | Task input channel(s) |
| `--target-channels` | `9` | Task target channel(s) |
| `--max-trials` | `120` | Training trial cap |
| `--eval-interval-trials` | `8` | Criterion evaluation interval |
| `--eval-trials` | `8` | Trials per evaluation block |
| `--inter-trial-ms` | `70.0` | Inter-trial interval (ms) |
| `--criterion-response-probability` | `0.875` | Learning criterion |
| `--readout-window-s` | `1.5` | Readout window for phase capture |
| `--warmup-s` | `0.5` | Pre-task warmup |
| `--noise-repeats` | `32` | Colored-noise repeats per beta in the spectral fit |
| `--noise-samples` | `8192` | Samples per colored-noise draw |
| `--output-dir` | auto | Output dir; defaults to `experiments/snn_reset/results/control_checks_<UTC timestamp>` |

Parallelism is per-job: trained and untrained arms are submitted as
`seeds × protocols × {reset, no_reset}` jobs to a `ProcessPoolExecutor` sized by
`--workers`. The reference run below used `workers=6` and finished in ~56s.

## Output artifacts

From `experiments/snn_reset/results/control_checks_10k_20260602/`:

| File | Contents |
| --- | --- |
| `metadata.json` | Start/finish UTC, `neurons`, `seeds`, `workers`, `protocol_ids`, `elapsed_s` |
| `noise_diagnostics.csv` | One row per `(beta, spatial_mode)`: target vs. estimated power slope (mean/std), `event_count`, `total_pulses`, `active_channels`, `mean_pulses_per_active_channel` |
| `trained_controls.csv` | One row per `(protocol, seed, mode)` for trained cultures: full trial metrics plus `control_mode`, `source_protocol_id`, weight-norm deltas, `reset_window_neuron_spikes` |
| `reset_vs_no_reset.csv` | Trained reset−no_reset comparison per `(source_protocol_id, seed)`: `*_reset`, `*_no_reset`, `*_delta` for weight erasure, residual performance, savings, trace AUC, health, `post_delta_norm`, reset-window spikes |
| `untrained_controls.csv` | One row per `(protocol, seed, mode)` for naive cultures: `weight_drift_norm`, `weight_drift_rel_to_baseline`, pre/post behavior, path strength deltas, `trace_auc_proxy`, reset-window spikes |
| `report.md` | Human-readable summary: setup, noise spectra, trained reset-vs-no-reset, reset-minus-no-reset, untrained controls |

## Headline finding

In the reference 10k run (seeds 1, 3, 4), **every reset-minus-no-reset delta is
exactly 0** across all three protocols and all seven compared metrics
(weight erasure, residual performance, savings, trace AUC, health,
`post_delta_norm`, reset-window spikes). The reset and no-reset arms produced
identical trial metrics and weight trajectories; the only column that differed
was `total_pulses` (e.g. `low_burden_0.75s` ≈ 19, `mid_burden_1.5s` ≈ 465,
`high_burden_3s` ≈ 1514 vs. 0 for no-reset). The electrode stimulation
delivered pulses but moved no downstream metric relative to free-running drift.

This reinforces the grid's falsification result: weight-erasure values are
negative for all three protocols (`-0.437`, `-0.854`, `-1.866` for low/mid/high
burden), i.e. post-reset weights drifted farther from naive, and that drift is
indistinguishable from leaving the trained network to free-run for the same
duration. The untrained arm shows the protocols do perturb a naive network
(weight drift 0.07–0.31 relative to baseline, scaling with duration) but, like
the trained arm, the `reset` and `no_reset` rows are identical metric-for-metric
apart from pulse count. The noise diagnostics confirm the actuator is sound: the
estimated power slope tracks the target `-beta` to within ~0.01 (e.g. beta −2 →
1.999 vs. target 2.0), so the null result is not an artifact of a miscalibrated
stimulation channel.

For the broader interpretation and next steps (varying training depth,
consolidation delay, homeostatic parameters, stronger protocol families), see
`docs/snn_reset/full_grid_10k_calibrated_20260602.md`.
