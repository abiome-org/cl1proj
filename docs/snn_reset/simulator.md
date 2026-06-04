# Reset simulator engine

The forward simulator is the core of `cl1_snn_reset`: a spatial E/I culture on a
64-channel MEA that can be trained on an electrode-to-electrode task, perturbed
by a colored reset protocol, and asked to relearn. One call to `run_trial`
drives this whole train-reset-relearn loop and hands the resulting weight and
activity snapshots to the metrics layer.

This doc covers the simulation engine only. Scoring (`metrics.py`,
`trace_probe.py`), ranking/plots (`analysis.py`), batch sweeps (`sweep.py`), and
the learned `inverse_control` subpackage are documented elsewhere.

## Modules

| Module | Role |
|--------|------|
| `config.py` | `CultureConfig`, `TaskConfig`, `ExperimentConfig` dataclasses, `load_experiment_config`, `to_dict` |
| `network.py` | `build_network`, the `CorticalCultureNetwork` (numpy) and `Brian2CultureNetwork` engines, `NetworkSnapshot` |
| `electrodes.py` | `ElectrodeArray`, `StimEvent`, `ChannelActivity` — the MEA interface |
| `noise.py` | `colored_noise`, `generate_colored_events`, the `beta` temporal-color parameter |
| `protocols.py` | `ResetProtocol`, `protocol_events`, `coarse_protocol_grid`, `shift_stim_events`, `stim_events_energy_uC` |
| `task.py` | `train_to_criterion`, `evaluate_task`, `TrainingResult` |
| `experiment.py` | `run_trial`, `capture_phase`, `apply_reset_protocol`, `build_trial_artifacts`, `record_spontaneous_activity`, `PhaseSnapshot`, `TrialResult` |
| `artifacts.py` | `TrialArtifacts` — the metrics input bundle |

All public symbols are re-exported from the package root (see
`src/cl1_snn_reset/__init__.py`). The package README is the API quick-start;
this doc is the engine internals.

## The culture network

`build_network(cfg: CultureConfig, seed: int = 1)` dispatches on `cfg.backend`
and returns the engine object. Both engines share an `ElectrodeArray` and the
same public surface: `advance`, `weights_vector`, `set_weights`, `snapshot`,
`path_strength`, `channel_connectivity_matrix`, and `synapse_count`.

`CorticalCultureNetwork` is the production engine. It is a sparse LIF network
with STDP and homeostasis built directly on numpy:

- Neurons are scattered uniformly across a `field_size_mm` square. A
  `scipy.spatial.cKDTree` finds local candidates; each source draws
  `mean_out_degree` synapses weighted by `exp(-distance / connection_length_mm)`,
  plus a few `long_range_prob` random long-range edges, capped at
  `max_out_degree`. Excitatory cells (`excitatory_fraction`) get weights from
  `excitatory_weight_range`, inhibitory from `inhibitory_weight_range`. The sign
  of every synapse is frozen at build time in `self.signs`.
- `advance(duration_ms, events, *, plasticity=True, record=True)` is the time
  step. Each `dt_ms` step decays synaptic and stimulation currents, integrates
  the membrane toward `v_rest_mv` with Gaussian `background_noise_mv`, fires
  cells over `v_threshold_mv` (plus a `spontaneous_rate_hz` floor), delivers
  synaptic current to targets, applies STDP, resets to `v_reset_mv`, and holds
  `refractory_ms`.
- STDP is pair-based with `stdp_a_plus` / `stdp_a_minus` over `stdp_tau_ms`,
  clipped to `[w_min, w_max]` per sign. Every `homeostasis_interval_ms`, an EMA
  of population rate is nudged toward `target_rate_hz` by scaling excitatory
  weights at rate `homeostasis_rate`.
- The hidden weight vector is private to the engine; protocols and tasks only
  ever stimulate and record through electrodes. `weights_vector` / `snapshot`
  expose the hidden state for metrics, and `channel_connectivity_matrix`
  averages positive synapse weights into a 64×64 channel coupling matrix (cached
  by weight-array identity). `path_strength(input_channels, target_channels)`
  reads the mean input→target block of that matrix — the scalar the task uses as
  a behavioral proxy.

`Brian2CultureNetwork` is a Brian2-backed LIF/STDP build for smaller exact runs.
It reuses the numpy engine's geometry, connectome, and electrode array, then
mirrors them into a Brian2 `NeuronGroup` + `Synapses` with `method="exact"` and
`prefs.codegen.target = cfg.brian2_codegen_target`. Its `advance` runs the
Brian2 network when plasticity is on and falls back to the numpy engine (synced
via `set_weights`) for non-plastic recording. Sweeps default to numpy because
repeated Brian2 graph runs carry higher overhead; `brian2` is opt-in and raises
`ModuleNotFoundError` if the package is absent.

### Backend selection

`cfg.backend` is one of `"numpy"` (aliases `"fast"`, `"sparse"`) or `"brian2"`.
`cfg.brian2_codegen_target` (default `"numpy"`) sets Brian2's codegen target when
that backend is active. Any other backend string raises `ValueError`.

## Configuration

`load_experiment_config(path)` parses a YAML file into an `ExperimentConfig`,
coercing nested `culture`/`task` blocks into their dataclasses (unknown keys are
dropped) and re-tupling `input_channels`, `target_channels`, and
`response_window_ms`. It requires PyYAML. `to_dict(value)` is the inverse
direction for provenance: it walks nested frozen dataclasses into plain dicts and
turns tuples into lists.

`ExperimentConfig` is one replicate: a `CultureConfig`, a `TaskConfig`, plus
trial-level knobs.

| Field | Default | Meaning |
|-------|--------:|---------|
| `culture` | `CultureConfig()` | network and electrode parameters |
| `task` | `TaskConfig()` | the conditioned-response task |
| `readout_window_s` | `1.5` | spontaneous-activity window recorded at each phase |
| `warmup_s` | `0.5` | non-plastic settling run before the baseline phase |
| `seed` | `1` | replicate seed (network build + RNG) |
| `keep_snapshots` | `False` | retain full `NetworkSnapshot`s and per-phase activities on the `TrialResult` |

`CultureConfig` (`n_neurons=10_000`, `n_electrodes=64`, `field_size_mm=3.0`,
`backend="numpy"`, …) holds the geometry, membrane/synapse constants, STDP and
homeostasis rates, electrode kernel exponents (`stim_kernel_gamma`,
`record_kernel_gamma`), and the stim/synaptic gains. `n_electrodes` must be a
perfect square (the array is laid out as a square grid).

`TaskConfig` defines the electrode-to-electrode conditioned response:
`input_channels=(8,)` paired with `target_channels=(55,)` at `pair_delay_ms`,
with `criterion_response_probability=0.65`, `max_trials=120`, and
`eval_trials` / `eval_interval_trials` controlling evaluation cadence.

## The electrode interface

`ElectrodeArray` is the only path in and out of the hidden network. Built from
config, it precomputes two distance kernels over electrode/neuron positions:

- `stim_kernel` (exponent `stim_kernel_gamma`, per-electrode max-normalized)
  spreads a channel pulse onto nearby cells. `stimulate(event)` sums the kernels
  for the event's channels and scales by current, a phase-balance factor, and a
  pulse-width factor relative to the 160 µs reference.
- `record_kernel` (exponent `record_kernel_gamma`, column-normalized) and its
  argmax `nearest_channel` assign each hidden spike to one channel. `record(...)`
  projects fired neurons into a `ChannelActivity`.

`StimEvent(time_us, channels, current_uA, pulse_width_us=160, phases=(-1.0, 1.0))`
is one charge-balanced channel pulse. `ChannelActivity` holds
`spike_times_ms`, `channels`, `counts`, `duration_ms`, `total_neuron_spikes`,
with a `binned_counts(bin_ms, channel_count)` helper.

## Colored reset protocols

A `ResetProtocol` is the perturbation applied after training:

| Field | Meaning |
|-------|---------|
| `beta` | temporal color of the stimulation (see below) |
| `duration_s` | total protocol length |
| `current_uA`, `pulse_width_us` | per-pulse amplitude and width |
| `schedule` | `static`, `alternating_blue_red`, `epoch_pause`, or `ramp` |
| `spatial_mode` | `shared`, `independent`, `correlated`, or `phase_shifted` |
| `burst_rate_hz`, `epoch_s`, `pause_s` | optional schedule modifiers |
| `protocol_id` | optional label; otherwise `.id` is derived from the fields |

`protocol_events(protocol, *, n_channels, rng)` compiles a protocol into a list
of `StimEvent`s. `static` emits one colored event stream; `alternating_blue_red`
splices a blue (`beta=-1.0`) first half with a red (`beta=2.0`) second half;
`epoch_pause` repeats active epochs separated by silent pauses; `ramp` scales
amplitude up over time. `shift_stim_events(events, offset_us)` time-shifts a
stream and is how the multi-segment schedules are stitched together. Unknown
schedules raise `ValueError`.

`coarse_protocol_grid()` is the screening grid: the product of
`beta ∈ {-2,-1,0,1,2}`, four spatial modes, three schedules, three durations,
and three currents. `stim_events_energy_uC(events)` totals delivered charge
(µC, with a factor of 2 for the biphasic pulse) for the cost axis used in
ranking; `ResetProtocol.total_charge_uC(total_pulses)` is the per-protocol
equivalent.

### The `beta` temporal-color parameter

`beta` is the central protocol knob. `colored_noise(beta, n, rng)` returns
zero-mean unit-variance noise whose power spectrum approximates `1/f**beta`:

| `beta` | Color | Spectral character |
|-------:|-------|--------------------|
| `-2` | violet | power rises steeply with frequency |
| `-1` | blue | power rises with frequency |
| `0` | white | flat spectrum |
| `1` | pink | `1/f` |
| `2` | red / brown | power concentrated at low frequency |

Negative `beta` produces fast, decorrelated, high-frequency drive; positive
`beta` produces slow, clustered, low-frequency drive. The coarse grid sweeps
`beta` from `-2` to `2`. `generate_colored_events` turns a `beta` value into a
pulse-event stream: it builds a `colored_noise` envelope over 1 ms bins,
exponentiates and mean-normalizes it into an intensity envelope, samples event
bins against `rate_hz` times that envelope, and (per spatial mode) chooses which
channels co-activate in each event. `_default_rate(beta)` picks a base rate when
`burst_rate_hz` is unset — higher for bluer protocols, lower for redder ones —
so temporal color shapes both event timing and event density.

## The conditioned-response task

`task.py` trains the input→target association by paired electrode stimulation.

- `evaluate_task(net, cfg, *, trials=None)` stimulates the input channels with
  plasticity off and measures the fraction of trials with a target-channel spike
  inside `response_window_ms` — the response probability.
- `train_to_criterion(net, cfg)` evaluates, then repeats `paired_training_trial`
  (input pulse, then target pulse `pair_delay_ms` later, plasticity on),
  re-evaluating every `eval_interval_trials`, until response probability reaches
  `criterion_response_probability` or `max_trials` is hit. It returns a
  `TrainingResult(trials_to_criterion, response_probability, reached_criterion,
  history)`.

The same routine runs twice per trial: once to establish the baseline skill and
once to measure how fast the culture relearns after the reset.

## One train-reset-relearn trial

`run_trial(cfg: ExperimentConfig, protocol: ResetProtocol, seed=None)` is the
top-level entry point. Data flow:

1. **build** — `build_network(cfg.culture, seed)` constructs the engine; if
   `warmup_s > 0` it advances non-plastically to settle activity.
2. **baseline phase** — `capture_phase` records the naive weight vector, task
   `path_strength`, and a `readout_window_s` window of spontaneous activity
   (`record_spontaneous_activity`) into a `PhaseSnapshot`.
3. **train** — `train_to_criterion` learns the task; result kept as `initial`.
4. **trained phase** — `capture_phase` again, recording the trained state.
5. **reset** — `apply_reset_protocol(net, protocol, seed=trial_seed + 10_000)`
   compiles the protocol via `protocol_events` and advances the network with
   plasticity on, returning the reset-window activity and the total pulse count.
6. **post-reset phase** — `capture_phase` records the perturbed state, and
   `evaluate_task` measures residual `post_behavior`.
7. **relearn** — `train_to_criterion` runs again; result kept as `relearn`.
8. **metrics handoff** — `build_trial_artifacts` assembles the baseline,
   trained, and post `PhaseSnapshot`s plus both `TrainingResult`s, the protocol,
   path strengths, and a `trace_auc_proxy` into a `TrialArtifacts`, which
   `compute_trial_metrics` (from `metrics.py`) scores.

The phases of a trial:

| Phase | Captured state | Used for |
|-------|----------------|----------|
| baseline (`W0` / `A0`) | naive weights, path, spontaneous activity | reference for erasure and health |
| trained (`Wtrained`) | post-training weights and path | what the reset is meant to undo |
| post-reset (`Wpost` / `Apost`) | weights, path, activity, residual behavior | reset effect and culture health |
| relearn | second training curve | savings / relearning speed |

`run_trial` returns a `TrialResult(metrics, initial, relearn, snapshots,
activities)`. `snapshots` and `activities` are populated only when
`cfg.keep_snapshots` is true (full `NetworkSnapshot`s for `W0`/`Wtrained`/
`Wpost`/`Wrelearn` and the four phase activities, including the reset window);
otherwise both are `None` and only the scored metrics and training results are
returned.

`PhaseSnapshot` carries `weights`, `path_strength`, optional `activity`, and an
optional full `NetworkSnapshot`. `TrialArtifacts` (in `artifacts.py`) is the
frozen bundle of everything the metrics layer consumes:
`W0`/`Wtrained`/`Wpost`, `A0`/`Apost`, `initial`/`relearn`, `post_behavior`,
`protocol`, `seed`, `total_pulses`, `trace_auc_proxy`, and the three path
strengths.

## How it is exercised

The engine is driven by scripts under `experiments/`, which import the public
API only and never modify `src/`:

- `experiments/snn_reset/full_grid_search.py` runs the calibrated coarse grid
  (`coarse_protocol_grid` × seeds) at scale via `run_sweep`;
  `control_checks.py` runs reset-vs-no-reset and noise/actuator diagnostics on
  representative protocols.
- `experiments/regression/smoke.py` runs one small train-reset-relearn trial
  (a fast CI gate); `benchmark.py` times network build/advance and a short
  protocol×seed sweep. The learned inverse-reset pipeline
  (`learned_inverse_reset.py` and the `inverse_reset_*.yaml` presets) sits on top
  of this engine but is documented with the `inverse_control` subpackage.

The headline study finding for the calibrated 10k grid is recorded in
`docs/snn_reset/full_grid_10k_calibrated_20260602.md`.
