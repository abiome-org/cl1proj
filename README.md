# CL1 Reset Experiments

This repository tests whether a trained cortical culture can be driven back
toward a naive, trainable state using stimulation alone.  The practical target
is a reset operation for wetware computing: not merely suppressing behavior for a
moment, but reducing the learned trace enough that relearning no longer shows a
large shortcut.

The project starts in simulation, where hidden synaptic weights are observable,
then preserves the same channel-level interface needed for later CL1 wetware
validation.

## Research Question

Can electrode stimulation with carefully chosen temporal and spatial statistics
erase a learned response in a fixed-sign STDP network, while keeping the culture
healthy enough to learn again?

The key distinction is between looking reset and being reset.  A culture can
fall to chance performance because it is silent, damaged, globally depressed, or
temporarily decorrelated.  A useful reset should also reduce savings: after the
protocol, the network should not relearn the old task substantially faster than
it learned from naive.

## Experimental Thesis

Learning is treated as a movement through synaptic-weight space.  Training digs
a task-specific basin; reset attempts to flatten or move the system back toward
the naive landscape without direct weight access.

The actuator is intentionally wetware-legal: channel-level electrode stimulation
only.  Protocols vary the color of pulse-event statistics, their schedule over
time, and their spatial correlation across electrodes.  The simulator keeps
ground-truth weights for analysis, but candidate protocols are judged through
the CL1-like interface wherever possible.

## Experimental Flow

Each trial follows the same loop:

1. Build a spatial recurrent excitatory/inhibitory culture on a 64-channel MEA.
2. Record naive activity and hidden baseline weights.
3. Train an electrode-to-electrode conditioned response task to criterion.
4. Apply a reset protocol made of charge-balanced pulse events.
5. Measure post-reset task behavior, spontaneous activity, health, and weights.
6. Retrain the same task and measure savings.

In simulation, the hidden weight matrix answers whether apparent behavioral
reset corresponds to real weight-space erasure.  In CL1-style operation, the
same protocol is constrained to electrode stimulation and channel-level readout.

## Protocol Space

The sweep explores three actuator axes:

- **Temporal color:** spectral slope `beta`, from violet/blue through white to
  pink/red event timing.
- **Schedule:** static stimulation, alternating regimes, and epoch/pause
  structure.
- **Spatial pattern:** shared, independent, correlated, or phase-shifted channel
  activity.

The main candidates are not arbitrary current injection into neurons.  They are
channel-level pulse trains that approximate colored noise and remain compatible
with the multielectrode-array interface.

## Readouts

The project ranks protocols by a bundle of outcomes rather than one scalar:

- **Weight erasure:** SNN-only distance from trained weights back toward naive.
- **Residual behavior:** task response probability after reset.
- **Savings:** relearning trials-to-criterion relative to initial learning.
- **Trace detectability:** classifier AUC for distinguishing naive from
  post-reset activity using channel readouts.
- **Criticality and health:** firing rates, EI balance, avalanche statistics,
  responsiveness, and avoidance of silent or saturated regimes.
- **Cost:** stimulation duration, pulse count, and current burden.

Low residual behavior is not enough.  A strong candidate should also have low
savings, weak trace detectability, preserved trainability, and tolerable
stimulation cost.

## Interpretation

A positive result would identify stimulation statistics that make a trained
culture behave and relearn like a fresh one.  A negative result is still useful:
if savings persists despite broad protocol search and restored activity
statistics, then stimulation alone may have a reset floor and stronger
interventions may be required.

## Repository Layout

- `src/cl1_snn_reset/`: core reset simulator, including the spatial E/I culture
  model, MEA electrode interface, colored pulse protocols, train-reset-relearn
  loop, metrics, trace probe, and sweep helpers.
- `src/cl1_clsdk_bridge/`: adapters that connect experiment packages to the CL
  SDK simulator runtime.
- `src/cl/`: vendored CL SDK runtime surface.  Keep SDK compatibility work here;
  keep experiment code outside this package.
- `experiments/snn_reset/`: benchmark and sweep entrypoints.
- `tests/snn_reset/`: reset simulator tests.
- `tests/clsdk_bridge/`: SDK bridge tests.
- `docs/snn_reset/` and `docs/clsdk_bridge/`: notes for the local experiment
  packages.

Compatibility imports are kept for existing notebooks and SDK call sites:
`cl.snn_reset` re-exports `cl1_snn_reset`, and `cl.twin.ResetSNNAdapter`
re-exports the adapter from `cl1_clsdk_bridge`.

## Running

Run the benchmark:

```bash
MPLCONFIGDIR=/private/tmp/mpl-cache XDG_CACHE_HOME=/private/tmp/xdg-cache \
  .venv-uv/bin/python experiments/snn_reset/benchmarks/benchmark_snn_reset.py
```

Run focused reset and bridge tests:

```bash
.venv-uv/bin/python -m pytest tests/snn_reset tests/clsdk_bridge -q
```

Enable the reset SNN inside the simulator:

```bash
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_DYNAMICS=snn_reset
```

## Relevant Prior Art

The experiment is motivated by several neighboring lines of work: attractor
unlearning, coordinated reset stimulation, depotentiation, spontaneous activity
and criticality recovery, dissociated-culture learning, and machine unlearning.
Useful starting points include Hopfield-style unlearning, Crick-Mitchison sleep
unlearning, Tass coordinated reset, DishBrain, Wagenaar/Pine/Potter culture
stimulation results, depotentiation studies, SNN unlearning, and recent work on
noise-driven maintenance of critical dynamics.
