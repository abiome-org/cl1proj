# CL1 Reset Experiments

This repository studies whether a trained cortical culture can be returned to a
naive, trainable state using stimulation alone.  The long-term target is a
practical reset operation for wetware computing: a protocol that removes a
learned trace, preserves culture health, and does not leave a strong relearning
shortcut behind.

The first stage is a simulator because it can expose the hidden weight matrix.
The simulator is still constrained at the interface: stimulation is delivered as
64-channel multielectrode-array pulse events, and candidate protocols are judged
through channel-level spikes, behavior, health, and trace-probe readouts.

## What The Experiments Ask

The central question is not just whether behavior falls to chance after a
stimulation protocol.  A culture can stop responding because it is silent,
depressed, damaged, saturated, or temporarily decorrelated.  A useful reset
should also reduce the learned synaptic trace and avoid savings when the same
task is trained again.

The experiments therefore separate three outcomes:

- Apparent behavioral reset: post-reset response probability falls.
- Mechanistic reset: trained weights move back toward the naive weight state.
- Functional reset: relearning is not substantially faster than initial
  learning.

A protocol only looks promising if it improves all three while keeping the
network responsive enough to learn afterward.

## Experimental Model

The reset simulator models a scaled dissociated cortical culture on a
multielectrode array:

- Spatially embedded recurrent excitatory/inhibitory spiking network.
- Local-distance-biased connectivity with sparse longer-range recurrence.
- Fixed-sign STDP on excitatory synapses plus slow homeostatic stabilization.
- Electrode stimulation as charge-balanced pulse events.
- Electrode readout as channel-level multi-unit activity.

The simulator keeps full weights and spikes for analysis, but the experiment
runner uses the same channel-level interface that later wetware protocols will
need.

## Trial Loop

Each trial follows a train-reset-relearn sequence:

1. Build a naive 2D culture and record baseline weights and activity.
2. Train an electrode-to-electrode conditioned response task.
3. Apply one reset protocol made of colored channel pulse events.
4. Measure post-reset behavior, activity, health, criticality, and weights.
5. Train the same task again and measure relearning savings.

The task uses paired stimulation: an input electrode pulse followed by a target
electrode pulse.  Native STDP strengthens paths that make the target response
follow the input.  Reset protocols then try to erase that learned structure
without direct access to synapses.

## Protocol Space

The coarse sweep varies the stimulation statistics along three actuator axes:

- Temporal color: spectral slope `beta` from violet/blue through white to
  pink/red event timing.
- Schedule: static stimulation, alternating color regimes, or epoch/pause
  structure.
- Spatial pattern: shared, independent, correlated, or phase-shifted channel
  activity.

Duration, current, pulse width, and pulse count are tracked as cost variables.
The protocol set is intentionally expressed as electrode pulse trains rather
than arbitrary direct current injection into individual neurons.

## Scoring

Protocols are ranked by a bundle of readouts:

- Weight erasure: whether post-reset weights return toward naive weights.
- Path erasure: whether the task-specific input-to-target path weakens.
- Residual behavior: target response probability after reset.
- Savings: relearning trials relative to initial trials.
- Trace AUC: classifier detectability of trained post-reset activity from
  channel readouts.
- Health and criticality: firing rate, active-channel fraction, branching
  ratio, and distance from naive spontaneous activity.
- Energy cost: pulses, current, and duration.

The ranking is a screen, not a proof.  The Pareto front is more important than a
single score because a low-cost protocol, a strong erasure protocol, and a
high-health protocol can be different candidates.

## Current 10k-Neuron Result

The calibrated full grid screened 540 reset protocols across three seeds on a
10,000-neuron, 64-channel simulated culture.

The scientific result is negative for true reset in this configuration.  Every
protocol had negative weight erasure, meaning post-reset weights were farther
from the naive state than the trained weights were.  The best-ranked protocols
are therefore the least damaging/least costly candidates, not successful resets.
The top Pareto candidate was:

```text
b2_epoch_pause_independent_0.75s_0.8uA
weight_erasure=-0.4373
residual_performance=0.5000
savings=-1.3889
trace_auc=0.5307
health=0.0996
energy_cost=0.0048
```

This argues that the current fixed-sign STDP reset actuator is not yet enough to
restore the simulated culture to a naive weight state.  It is still useful:
behavioral suppression and low trace AUC alone would have overstated reset
quality without the hidden-weight and savings checks.

Full run notes are in `docs/snn_reset/full_grid_10k_calibrated_20260602.md`.

## Running

Run the calibrated full 10k-neuron grid:

```bash
.venv-uv/bin/python experiments/snn_reset/full_grid_search.py
```

Run the smaller benchmark:

```bash
.venv-uv/bin/python experiments/snn_reset/benchmarks/benchmark_snn_reset.py
```

Run focused reset and bridge tests:

```bash
.venv-uv/bin/python -m pytest tests/snn_reset tests/clsdk_bridge -q
```

Enable the reset SNN inside the CL SDK simulator:

```bash
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_DYNAMICS=snn_reset
```

## Where Things Live

- `src/cl1_snn_reset/`: reset simulator, protocols, metrics, trace probe, and
  sweep helpers.
- `src/cl1_clsdk_bridge/`: adapter between the reset simulator and the CL SDK
  twin runtime.
- `src/cl/`: CL SDK runtime surface and compatibility imports.
- `experiments/snn_reset/`: benchmark and full-grid entrypoints.
- `docs/snn_reset/`: experiment notes and full-grid reports.
- `tests/snn_reset/` and `tests/clsdk_bridge/`: focused simulator and bridge
  tests.

Compatibility imports are kept for existing notebooks and SDK call sites:
`cl.snn_reset` re-exports `cl1_snn_reset`, and `cl.twin.ResetSNNAdapter`
re-exports the adapter from `cl1_clsdk_bridge`.

## Prior Art

The experiment connects several related threads: Hopfield and Crick-Mitchison
unlearning, coordinated reset stimulation, depotentiation, dissociated-culture
learning, criticality recovery, and machine unlearning.  Useful starting points
include DishBrain, Wagenaar/Pine/Potter culture stimulation, Tass coordinated
reset, depotentiation studies, SNN unlearning, and noise-driven maintenance of
critical dynamics.
