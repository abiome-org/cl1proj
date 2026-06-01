# Biological Digital Twin

The CL SDK now has a constrained extension point for moving the simulator from
passive replay toward a biological digital twin of the CL1.  The implementation
is intentionally contained in:

- `src/cl/_twin_producer.py`
- `src/cl/twin/`
- `tests/cl/test_twin_producer.py`

The public `cl.open()` API is unchanged.  The twin writes the same shared-memory
frames, spikes, and stims as the replay producer, so closed-loop applications,
recording, analysis, and visualization continue to use the existing SDK surface.

## Selecting a Backend

Replay remains the default simulator mode.

```shell
CL_SDK_SIM_MODE=replay
```

The first twin backend can be enabled with any of these aliases:

```shell
CL_SDK_SIM_MODE=surrogate
CL_SDK_SIM_MODE=biological
CL_SDK_SIM_MODE=twin
```

Useful twin configuration variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CL_SDK_TWIN_SEED` | `1` | Deterministic random seed. |
| `CL_SDK_TWIN_DIV` | `21` | Days-in-vitro maturation prior for activity and coupling. |
| `CL_SDK_TWIN_BASELINE_RATE_HZ` | `0.15` | Spontaneous per-channel spike rate. |
| `CL_SDK_TWIN_NOISE_STD` | `5.0` | Raw frame noise in sample units. |
| `CL_SDK_TWIN_NOISE_COLOR` | `pink` | Raw frame noise model: `pink` for stateful 1/f-like noise or `white` for independent Gaussian noise. |
| `CL_SDK_TWIN_STIM_COUPLING` | `1.0` | Strength of electrode-to-tissue coupling. |
| `CL_SDK_TWIN_EVOKED_PROBABILITY` | `0.9` | Local probability of stim-evoked spikes. |
| `CL_SDK_TWIN_ARTIFACT_AMPLITUDE` | `1200.0` | Stimulation artifact amplitude in sample units. |
| `CL_SDK_TWIN_PLASTICITY` | `off` | Reserved for `off`, `stp`, `stdp`, and homeostatic modes. |
| `CL_SDK_TWIN_DOPAMINE` | `0.0` | Third-factor gate for surrogate gains and SNN STDP/excitability. |
| `CL_SDK_TWIN_GABA_BLOCK` | `0.0` | Inhibitory synapse blockade in SNN mode. |
| `CL_SDK_TWIN_CULTURE_STATE` | `normal` | Culture regime: `normal`, `quiescent`, `synchronized_burst`, or `auto`. |
| `CL_SDK_TWIN_EXCITABILITY` | `1.0` | Baseline tissue excitability multiplier. |
| `CL_SDK_TWIN_PROFILE_PATH` | unset | Optional JSON culture profile produced by `TwinProfile`. |
| `CL_SDK_TWIN_DYNAMICS` | `off` | Optional recurrent dynamics mode: `population`, `hawkes`, `glm`, or `recurrent`. |
| `CL_SDK_TWIN_RECURRENT_COUPLING` | `0.35` | Strength of recurrent population propagation. |
| `CL_SDK_TWIN_PROPAGATION_DELAY_FRAMES` | `12` | Delay for recurrent population spikes. |
| `CL_SDK_TWIN_REFRACTORY_FRAMES` | `20` | Per-population refractory window. |
| `CL_SDK_TWIN_SNN_NEURON_COUNT` | `256` | Virtual neuron count for Izhikevich SNN mode. |
| `CL_SDK_TWIN_SNN_EXCITATORY_FRACTION` | `0.8` | Fraction of excitatory cells in SNN mode. |
| `CL_SDK_TWIN_SNN_COUPLING` | `1.0` | Electrode-to-cell coupling strength in SNN mode. |
| `CL_SDK_TWIN_SNN_CONNECTION_PROBABILITY` | `0.08` | Base probability for distance-decayed cell synapses. |
| `CL_SDK_TWIN_SNN_LENGTH_CONSTANT_UM` | `250.0` | Spatial decay constant for SNN synaptic probability. |
| `CL_SDK_TWIN_SNN_FIELD_GAMMA` | `1.25` | Electrode field attenuation exponent into virtual tissue. |
| `CL_SDK_TWIN_SNN_SPARSE_THRESHOLD` | `1024` | SNN neuron count where Izhikevich mode switches from dense matrices to sparse adjacency lists. |
| `CL_SDK_TWIN_SNN_MAX_TARGETS_PER_SOURCE` | `64` | Upper bound on outgoing recurrent targets per source cell in sparse SNN mode. |
| `CL_SDK_TWIN_SNN_STDP_LEARNING_RATE` | `0.004` | Cell-synapse learning rate in SNN STDP modes. |
| `CL_SDK_TWIN_SNN_STDP_TAU_FRAMES` | `500.0` | STDP trace decay constant in CL frames. |
| `CL_SDK_TWIN_SNN_STP_RECOVERY_FRAMES` | `1500.0` | Vesicle-resource recovery time for SNN short-term depression. |
| `CL_SDK_TWIN_SNN_STP_FACILITATION_FRAMES` | `500.0` | Facilitation decay time back to baseline release probability. |
| `CL_SDK_TWIN_SNN_STP_DEPRESSION` | `0.15` | Per-spike resource depletion in SNN STP modes. |
| `CL_SDK_TWIN_SNN_STP_FACILITATION` | `0.08` | Per-spike facilitation increment in SNN STP modes. |
| `CL_SDK_TWIN_SNN_REFRACTORY_FRAMES` | `25` | Per-cell refractory interval after a virtual spike. |
| `CL_SDK_TWIN_SNN_MIN_PROPAGATION_DELAY_FRAMES` | `1` | Minimum cell-to-cell recurrent delay. |
| `CL_SDK_TWIN_SNN_MAX_PROPAGATION_DELAY_FRAMES` | `25` | Maximum distance-scaled cell-to-cell recurrent delay. |
| `CL_SDK_TWIN_SPIKE_THRESHOLD_SIGMA` | `4.5` | Negative rolling-threshold multiplier for converting raw EAPs into `SpikeRecord` objects. |
| `CL_SDK_TWIN_SPIKE_REFRACTORY_FRAMES` | `20` | Per-channel detector refractory interval. |
| `CL_SDK_TWIN_SPIKE_ARTIFACT_BLANK_FRAMES` | `4` | Per-channel blanking interval after stimulation artifact onset. |

## Current Implementation

The twin backend is opt-in and bidirectional.  The default twin dynamics use a
calibrated surrogate layer for fast SDK tests; `CL_SDK_TWIN_DYNAMICS=population`
adds MEA-level recurrence, and `CL_SDK_TWIN_DYNAMICS=izhikevich` enables dense
or sparse cell-level SNN dynamics:

1. User code calls `neurons.stim(...)`.
2. `Neurons._queue_stims()` forwards the stim timestamp, channel, and current
   into `BiologicalTwinProducer`.
3. The producer subprocess applies the stim to `SurrogateTwinModel`.
4. The model uses the 8x8 MEA geometry to spread stimulation spatially.
5. Nearby channels receive stimulation artifacts and probabilistic evoked spikes.
6. Frames, spike records, and stim records are written into `SharedDataBuffer`.

This means stimulation now changes future simulated measurements.

The integration is intentionally organized as a capability ladder rather than a
single claim that the simulator is biologically complete. Applications can
inspect that ladder directly:

```python
from cl.twin import describe_twin_capabilities

report = describe_twin_capabilities()
print(report.implemented_count, report.approximated_count, report.roadmap_count)
```

The report marks bidirectional MEA coupling, artifacts, profile calibration,
validation gates, virtual pharmacology, feedback protocols, and learning-curve
evaluation as implemented. Population recurrence, Izhikevich SNN dynamics,
STDP, and STP move from approximated to implemented when their opt-in runtime
modes are enabled. Large sparse 5k-cell Izhikevich cultures are structurally
supported through adjacency-list synapses above `CL_SDK_TWIN_SNN_SPARSE_THRESHOLD`.
Long-horizon training can run without wall-clock sleeps through
`TwinAcceleratedTrainer`; dedicated GPU kernels can still replace internals
later without changing the public SDK boundary.

Formal spikes are now derived from rendered voltage. Surrogate and SNN events
write a 75-sample biphasic EAP template into raw channel frames, and
`RollingThresholdSpikeDetector` emits `SpikeRecord` objects when the negative
trough crosses a robust per-channel noise threshold outside artifact-blanking
windows. User-visible raw frames still include stimulation artifacts; the
detector uses an internal post-blanking signal so evoked spikes can be detected
after the artifact window rather than being overwritten by the artifact tail.

## Culture Profiles

The twin can load a calibrated culture profile:

```shell
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_PROFILE_PATH=/path/to/culture-profile.json
```

Profiles are plain JSON files created by `cl.twin.TwinProfile`.  They currently
capture:

- Per-channel baseline firing rates.
- Per-channel median inter-spike intervals.
- Per-channel burst rate, median burst duration, and mean burst spike count.
- Per-channel raw noise estimates.
- Binned functional connectivity.
- Stim-to-spike response probabilities.
- Stim-to-spike median latency.
- Stim-to-spike pairwise response counts and confidence scores.
- 95% uncertainty intervals for baseline rates and stim-response fields.
- Estimated SNN neuron count and channel-density topology prior from activity,
  stim-response recruitability, and functional connectivity support.
- Per-channel and per-field calibration confidence scores.
- Dead/noisy channel hints.

Example:

```python
from cl.twin import TwinProfile

profile = TwinProfile.from_recording_path("recording.h5")
profile.save("culture-profile.json")
```

At runtime, `SurrogateTwinModel` uses the profile to shape spontaneous firing,
raw noise, connectivity-weighted evoked responses, and calibrated response
latencies.  When burst fields are present, background activity can be emitted as
short packets of spikes rather than only independent Poisson events.  Missing
profile fields fall back to deterministic defaults.

Profiles are schema-versioned.  `TwinProfile.load(...)` migrates older JSON
profiles into the current schema by filling newly introduced fields with neutral
defaults, and rejects profile files from unsupported future schema versions
rather than silently misinterpreting them.

## Plasticity Hooks

The surrogate includes a bounded `PlasticityState` hook for channel-level
response-gain adaptation, while SNN modes implement cell-synapse STDP and STP:

```shell
CL_SDK_TWIN_PLASTICITY=stp
CL_SDK_TWIN_PLASTICITY=stdp
CL_SDK_TWIN_PLASTICITY=stdp_homeostatic
```

When enabled, repeated stimulation can depress local response gain and
stim-following spikes can potentiate it.  The response gain is bounded so
closed-loop training can alter the twin without creating runaway activity.

In Izhikevich mode, `CL_SDK_TWIN_PLASTICITY=stdp` also adapts the actual
cell-to-cell synapse matrix.  Synapses use a target-by-source layout, preserve
their original excitatory or inhibitory sign, and are clipped to bounded
magnitudes.  `stdp_homeostatic` additionally rescales incoming weight norms back
toward their baseline totals.

Virtual dopamine is exposed through `CL_SDK_TWIN_DOPAMINE`.  In surrogate mode
it increases the per-channel STDP-like response-gain update.  In Izhikevich
mode it acts as a bounded third factor: the same pre/post spike timing produces
larger cell-synapse updates, and virtual cells become slightly easier to recruit
without bypassing the electrode coupling or calibration profile.

In the same SNN mode, `CL_SDK_TWIN_PLASTICITY=stp` enables short-term synaptic
depression and facilitation.  Each fired presynaptic cell transiently depletes a
readily releasable resource pool and increases release probability; both states
recover toward baseline over configurable frame constants.  The product of those
states modulates outgoing recurrent transmission, giving the simulated tissue a
fast activity-dependent fatigue mechanism that can dampen runaway loops.
`stdp` and `stdp_homeostatic` also include this fast STP modulation so long-term
learning and short-term fatigue can operate together.

## Accelerated Training

`CL_SDK_ACCELERATED_TIME=1` keeps the normal SDK producer from sleeping on wall
clock.  For offline experiments, `TwinAcceleratedTrainer` runs the same model
path directly in process:

```python
from cl.twin import TwinAcceleratedTrainer, TwinFeedbackProtocol

trainer = TwinAcceleratedTrainer(model=model, protocol=TwinFeedbackProtocol(64, 25_000))
result = trainer.run_trials([False, False, True, True], sensory_channel=3, motor_channel=12)
```

Correct trials are compiled into structured feedback; incorrect trials are
compiled into chaotic stimulation.  The trainer renders biological time in
chunks without sleeping and returns a `TwinTrainingResult` containing simulated
frames, delivered feedback pulses, observed spikes, and a learning-curve report.

## DIV Maturation

`CL_SDK_TWIN_DIV` now feeds a `MaturationState` prior.  Low-DIV cultures are
treated as sparse and weakly coupled: lower spontaneous firing, lower burst
probability, lower electrode/cell recruitment, lower recurrent coupling, fewer
SNN synapses, and slower propagation.  Mature cultures approach the configured
rates, coupling, and delay values.  This is intentionally a bounded prior; a
loaded `TwinProfile` remains the stronger source of culture-specific evidence.

## Culture State

`CL_SDK_TWIN_CULTURE_STATE` selects a coarse runtime regime.  `quiescent`
suppresses baseline firing, burst tendency, coupling, and excitability.
`normal` leaves the configured/profiled values unchanged.  `synchronized_burst`
models hyperexcitable epileptiform tissue by increasing recurrent coupling and
burst scales and by occasionally adding near-simultaneous network-wide burst
packets.  `auto` infers this state from the pharmacology knobs: high
`CL_SDK_TWIN_GABA_BLOCK` becomes synchronized bursting, while very low
`CL_SDK_TWIN_EXCITABILITY` becomes quiescent.

## Closed-Loop Feedback Patterns

`TwinFeedbackProtocol` turns task outcomes into stimulation patterns without
changing the public SDK API.  Correct outcomes produce a predictable structured
sensory-to-motor pair.  Incorrect outcomes produce a broad, high-frequency
chaotic pattern across all MEA channels.  `SurrogateTwinModel.apply_feedback(...)`
then routes both patterns through ordinary electrode stimulation, so artifacts,
evoked spikes, plasticity, and optional SNN dynamics all see the same MEA
coupling path as `neurons.stim(...)`.

`TwinLearningEvaluator` scores task-level learning curves from closed-loop
trials.  It compares early and late accuracy windows, optionally including
response latency, so Pong-like tasks can validate that structured feedback
improves behavior instead of only perturbing spike statistics:

```python
from cl.twin import TaskTrial, TwinLearningEvaluator

trials = [
    TaskTrial(timestamp=i, correct=outcome, response_latency_frames=latency)
    for i, (outcome, latency) in enumerate(task_log)
]
report = TwinLearningEvaluator.evaluate_trials(trials, window_size=20)
assert report.passed
```

## Population Dynamics

The surrogate can now run a live population dynamics layer:

```shell
CL_SDK_TWIN_DYNAMICS=population
```

This is a Hawkes/GLM-like recurrent model at MEA resolution.  Each channel is a
local neural population.  When a population spikes, `PopulationDynamics` uses the
calibrated connectivity matrix to schedule delayed downstream spikes, subject to
bounded coupling, propagation delay, and a refractory window.  This is the first
runtime step beyond purely evoked responses: activity can now propagate through
the virtual culture after the initial stimulation event.

The design is intentionally replaceable.  The producer still only asks
`SurrogateTwinModel.render(...)` for frames and spikes; population, dense
Izhikevich, and sparse Izhikevich engines all sit behind that same internal
dynamics contract.

## Izhikevich SNN Mode

The twin also includes a compact cell-level SNN:

```shell
CL_SDK_TWIN_DYNAMICS=izhikevich
```

`IzhikevichNetwork` simulates virtual neurons with the standard membrane and
recovery equations.  `TissueTopology` places those cells in the same 1000um x
1000um coordinate system as the 8x8 MEA, maps each cell to its nearest
recording electrode, projects electrode stimulation into cells using a
distance-decayed extracellular field kernel, and builds directed
distance-decayed synapses.  Synapses are stored target-by-source; excitatory
source cells create positive outgoing weights and inhibitory source cells create
negative outgoing weights.  Cell spikes are then projected back into MEA-level
spike events.

Recurrent SNN transmission uses distance-scaled propagation delays.  When a cell
spikes, its outgoing synaptic current is scheduled into a small frame-delay
queue instead of being injected immediately.  Cells also have a configurable
refractory interval, which prevents unrealistically rapid repeated spikes after
strong stimulation or recurrent input.

`CL_SDK_TWIN_GABA_BLOCK` scales negative SNN synapses toward zero, letting the
twin emulate inhibitory blockade conditions such as bicuculline-like
disinhibition without changing user code.

The default neuron count is intentionally modest so it can run cheaply in the
SDK producer process, but the same mode can be scaled up with sparse storage.

When `CL_SDK_TWIN_SNN_NEURON_COUNT` is at or above
`CL_SDK_TWIN_SNN_SPARSE_THRESHOLD`, the producer swaps in
`SparseIzhikevichNetwork`.  The sparse engine keeps the same stim/render API but
stores recurrent synapses as per-source adjacency lists with bounded outgoing
targets and delay buckets.  It also skips dense pairwise distance storage in
`TissueTopology`, which keeps 5k-cell cultures feasible inside the constrained
SDK process while preserving distance-decayed E/I synapses, propagation delays,
STP, and bounded local STDP.

## North-Star Architecture

The long-term design keeps the existing producer boundary:

```text
Neurons
  |
  v
ProducerFactory
  |
  +-- DataProducer              passive replay/random simulator
  +-- BiologicalTwinProducer    bidirectional biological twin
        |
        +-- MEAGeometry
        +-- ArtifactModel
        +-- NeuralStateModel
        +-- StimResponseModel
        +-- SpikeDetector
        +-- PlasticityModel
        +-- CalibrationProfile
```

The next model replacement should happen inside `src/cl/twin/`, not by changing
the public API.

## Implemented Architecture

### Phase 1: Bidirectional surrogate

Status: implemented.

- Selectable twin backend with `CL_SDK_SIM_MODE`.
- Spatial MEA coupling.
- Stimulation artifacts in raw frames.
- Stim-evoked spikes.
- Slow latent excitability state.
- Regression tests proving that stimulation causally changes future readouts.

### Phase 2: Calibrated statistical twin

Status: implemented.

Implemented:

- Baseline firing rate per channel.
- ISI and burst statistics as runtime calibration targets.
- Functional connectivity matrix.
- Stim-triggered latency histograms.
- Pairwise stim-to-channel response support and confidence.
- Coarse cell-density/topology priors inferred from per-channel activity,
  stim-response recruitability, and functional connectivity support.
- Per-channel noise, artifact, and dead-channel profiles.
- Profile version migration.
- Confidence/uncertainty estimates for fitted fields.
- JSON profile save/load.
- Runtime profile loading via `CL_SDK_TWIN_PROFILE_PATH`.

The model fits these profiles from `RecordingView` metrics and serializes them
as a stable culture profile.

### Phase 3: Live neural dynamics

Status: implemented.

Implemented:

- `PopulationDynamics` recurrent engine.
- Profile-connectivity-driven spike propagation.
- Configurable coupling, propagation delay, and refractory period.
- Runtime opt-in via `CL_SDK_TWIN_DYNAMICS=population`.
- `IzhikevichNetwork` cell-level SNN engine.
- Runtime opt-in via `CL_SDK_TWIN_DYNAMICS=izhikevich`.
- `SparseIzhikevichNetwork` large-culture SNN engine above
  `CL_SDK_TWIN_SNN_SPARSE_THRESHOLD`.
- Configurable virtual neuron count, excitatory fraction, and SNN coupling.
- `TissueTopology` with 2D cell positions, nearest-electrode projection, and
  electrode-to-cell field attenuation.
- Profile-calibrated SNN neuron count and channel-density-biased cell placement.
- Distance-decayed directed cell synapses.
- Source-correct excitatory/inhibitory synapse signs and GABA blockade scaling.
- Cell-level STDP and homeostatic STDP modes for Izhikevich synapses.
- Cell-level short-term depression/facilitation for recurrent SNN transmission.
- Dopamine-gated SNN STDP and bounded cell recruitment modulation.
- Cell-level refractory gating and distance-scaled recurrent propagation delays.

This phase preserves the producer API and only changes internals under
`src/cl/twin/`.

### Phase 4: Plasticity and closed-loop learning

Status: implemented.

Implemented:

- Short-term depression and facilitation.
- Dopamine as a third-factor gate for surrogate response gains and SNN STDP.
- Structured and chaotic feedback pattern generation routed through MEA stims.
- Learning-curve reports over early versus late task windows.

Plasticity must remain configurable because deterministic SDK testing and
adaptive scientific simulation have different needs.

### Phase 5: Virtual biology and pharmacology

Expose biological condition profiles:

- DIV maturation.
- GABA blockade.
- Excitability shifts.
- Culture-level state transitions between quiescence, normal activity, and
  synchronized network-wide bursting.

Implemented:

- GABA blockade scaling for inhibitory SNN synapses.
- DIV maturation scaling for baseline activity, burst tendency, coupling,
  SNN synapse density, and propagation delay.
- Excitability shifts through config and calibrated profiles.
- Dopamine-like learning-rate modulation in surrogate and SNN plasticity.
- Culture-level quiescent, normal, and synchronized-burst regimes.

These should be profile/config fields, not public API changes.

## Validation Gates

A twin is only credible if it predicts interventions, not just baseline traces.
The implemented validation surface covers:

- Firing statistics.
- ISI distributions.
- Burst structure.
- Functional connectivity.
- Stim-triggered histograms.
- Artifact and blanking behavior.
- Closed-loop task performance curves.

The current implementation starts that validation with
`tests/cl/test_twin_producer.py`, which proves these invariants:

- A stimulation event creates future biological consequences in the readout.
- Replay remains the default simulator mode.
- Culture profiles serialize and change runtime dynamics.
- Culture profiles capture ISI/burst statistics and can drive burst-like
  background spike packets.
- Culture profiles migrate older schemas and expose bounded confidence metadata.
- Stim response profiles include pairwise response support/confidence and the
  runtime attenuates low-confidence response boosts.
- Culture profiles can bias SNN neuron count and virtual cell placement from
  per-channel activity support.
- `CL_SDK_TWIN_PROFILE_PATH` is honored by runtime configuration.
- Plasticity state is opt-in, bounded, and modulates response gain.
- Population dynamics can propagate spikes through calibrated connectivity.
- Izhikevich SNN mode can emit MEA spikes from cell-level stimulation.
- SNN mode uses spatial tissue topology, field coupling, and cell synapses.
- SNN excitatory/inhibitory source signs and GABA blockade are explicit and
  tested.
- SNN STDP changes cell synapses only in opt-in plasticity modes and preserves
  bounded signs.
- SNN short-term plasticity, refractory gating, and delayed recurrent current
  are explicit, tested cell-level mechanisms.
- Validation compares stim-triggered response probability and latency when
  simulated stim events are supplied.
- Validation checks raw-frame artifact/blanking windows when stims and frames
  are supplied.
- Learning validation compares early and late closed-loop task performance.

For runtime or notebook checks, `cl.twin.TwinValidator` can compare simulated
spike events against a `TwinProfile` and return a `TwinValidationReport`.
Current gates include firing-rate MAE, median-ISI MAE, burst-rate MAE, and
optional stim-triggered response probability/latency MAE plus artifact blanking
coverage when raw frames are supplied:

```python
from cl.twin import TwinProfile, TwinValidator

profile = TwinProfile.load("culture-profile.json")
report = TwinValidator.validate_spikes(
    profile=profile,
    spikes=simulated_spikes,
    stims=simulated_stims,
    raw_frames=simulated_frames,
    raw_frame_start=frame_start_timestamp,
)
assert report.passed
```

The tolerances are explicit inputs so experiments can tighten them as the twin
moves from SDK development aid toward biological prediction.
