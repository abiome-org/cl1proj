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

## Results

Full run notes are in `docs/snn_reset/full_grid_10k_calibrated_20260602.md`.

## Repository layout

| Path | Role |
|------|------|
| `src/cl1_snn_reset/` | Installable reset simulator library (import in code and notebooks) |
| `src/cl1_clsdk_bridge/` | CL SDK surrogate twin adapter for the reset culture |
| `src/cl/` | Vendored CL SDK runtime (minimal changes; compatibility shims only) |
| `experiments/` | Runnable studies; see `experiments/README.md` |
| `tests/` | Fast library and bridge unit tests (`pytest`) |
| `docs/snn_reset/` | Archived run reports |

Experiments consume the library through `import cl1_snn_reset`. They do not
modify `src/`. Each experiment folder has a `README.md` for scripts and results
paths.

## Running

Install the package in the project virtualenv (editable install is typical):

```bash
uv sync   # or pip install -e .
```

Regression smoke (quick):

```bash
.venv-uv/bin/python experiments/regression/smoke.py
```

Regression benchmark (timing / short sweep):

```bash
.venv-uv/bin/python experiments/regression/benchmark.py
```

Calibrated full 10k-neuron protocol grid:

```bash
.venv-uv/bin/python experiments/snn_reset/full_grid_search.py
```

Library and bridge tests:

```bash
.venv-uv/bin/python -m pytest tests/snn_reset tests/clsdk_bridge -q
```

Enable the reset SNN inside the CL SDK simulator:

```bash
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_DYNAMICS=snn_reset
```

## Prior Art

The experiment connects several related threads: Hopfield and Crick-Mitchison
unlearning, coordinated reset stimulation, depotentiation, dissociated-culture
learning, criticality recovery, and machine unlearning.  Useful starting points
include DishBrain, Wagenaar/Pine/Potter culture stimulation, Tass coordinated
reset, depotentiation studies, SNN unlearning, and noise-driven maintenance of
critical dynamics.

Compatibility imports are kept for existing notebooks and SDK call sites:
`cl.snn_reset` re-exports `cl1_snn_reset`, and `cl.twin.ResetSNNAdapter`
re-exports the adapter from `cl1_clsdk_bridge`.
