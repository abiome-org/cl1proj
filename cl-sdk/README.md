# CL SDK

This package provides an implementation of the CL API to assist with local development of applications that can run on a Cortical Labs CL1 system.

Please refer to [docs.corticallabs.com](https://docs.corticallabs.com) for the latest documentation.

## Prerequisites

This SDK requires Python 3.12 or later.

## Installation

Use of a venv is recommended:
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip3 install cl-sdk
```

## Cortical Labs Developer Guide

This SDK is capable of running most of the Jupyter notebooks in our developer guide. Install cl-sdk as above, then:

```bash
$ git clone https://github.com/Cortical-Labs/cl-api-doc.git
```

From here you can open and run the `*.ipynb` notebooks directly in Visual Studio Code, or by installing and running Jupyter Lab:

```bash
$ pip3 install jupyterlab
$ jupyter lab cl-api-doc
```

### Development

For working on the simulator itself:

```bash
$ pip3 install -e .
```

### Running Tests

```bash
$ pip3 install -e '.[test]'
$ pytest
```

### Building Documentation

```bash
$ pip3 install -e '.[test]'
$ python3 -m docs.make
```

Serve the built docs to view in a browser:

```bash
$ python3 -m http.server -d docs/html
```

## User Options

Several user options can be set by defining environment variables in a `.env` file of your project directory.

### Simulation from a recording

Spikes and samples are simulated by replaying recordings as set by the `CL_SDK_REPLAY_PATH` environment variable in the `.env` file. If this is omitted, a temporary recording with randomly generated samples and spikes will be used that is based on a Poisson distribution and the following optional environment variables:
- `CL_SDK_SAMPLE_MEAN`: Mean samples value (default 170). This value will be in microvolts when multiplied by the constant "uV_per_sample_unit" in the recording attributes;
- `CL_SDK_SPIKE_PERCENTILE`: Percentile threshold for sample values, above which will correspond to a spike (default 99.995);
- `CL_SDK_DURATION_SEC`: Duration of the temporary recording (default 60); and
- `CL_SDK_RANDOM_SEED`: Random seed (defaults to Unix time).

The starting position of the replay recording will be randomised every time `cl.open()` is called. This can be overriden by setting `CL_SDK_REPLAY_START_OFFSET`, where a value of `0` indicates the first frame of the recording.

### Speed of simulation

The simulator can operate in two timing modes:
- Based on wall clock time (default), or
- Accelerated time.

Accelerated time mode can be enabled by setting `CL_SDK_ACCELERATED_TIME=1` environment variable in the `.env` file. When enabled, passage of time will be decouple from the system wall clock time, enabling accelerated testing of applications.

### Biological twin mode

The default simulator remains passive replay/random generation. An experimental,
bidirectional biological twin backend can be enabled with:

```bash
CL_SDK_SIM_MODE=surrogate
```

This backend preserves the public `cl.open()` API and shared-memory producer
contract, but stimulation now perturbs future simulated readouts by creating
spatially coupled artifacts and evoked spikes. Formal twin spikes are detected
from rendered biphasic EAP waveforms with a rolling negative-threshold detector,
rather than being returned directly from hidden simulator labels. Raw baseline
frames use stateful pink-like MEA noise by default, with `CL_SDK_TWIN_NOISE_COLOR=white`
available for independent Gaussian noise. See `docs/twin.md` for the
implementation plan from the current surrogate backend to the full CL1 digital
twin north star.

Optionally load a calibrated culture profile:

```bash
CL_SDK_TWIN_PROFILE_PATH=/path/to/culture-profile.json
```

Profiles can calibrate baseline firing, raw noise, functional connectivity,
stim response probability/latency/confidence, uncertainty intervals, median
ISIs, and burst-like background spike packets. Profiles also carry a coarse
channel-density prior, informed by activity, stim responses, and functional
connectivity, that can bias SNN virtual cell placement.
Profile JSON is schema-versioned and includes bounded confidence and interval
metadata for calibrated fields.
Use `cl.twin.TwinValidator` to compare simulated spike events against a profile
with firing-rate, ISI, burst-rate, optional stim-triggered response, and
raw-frame artifact/blanking gates.

Optionally enable recurrent population dynamics:

```bash
CL_SDK_TWIN_DYNAMICS=population
```

Or the compact cell-level Izhikevich SNN:

```bash
CL_SDK_TWIN_DYNAMICS=izhikevich
```

The SNN mode places virtual cells in MEA tissue coordinates and supports
distance-decayed electrode field coupling, cell synapses, refractory gating,
and propagation-delayed recurrent transmission.
At larger configured cell counts, Izhikevich mode switches to a sparse
adjacency-list engine so thousands of virtual cells do not require dense
cell-by-cell synapse or distance matrices.
`CL_SDK_TWIN_DIV` applies a bounded maturation prior: low-DIV cultures are
sparser, less excitable, less burst-prone, and slower to propagate recurrent
activity than mature cultures.
`CL_SDK_TWIN_CULTURE_STATE` can select `normal`, `quiescent`,
`synchronized_burst`, or `auto` for culture-level activity regimes.
Inhibitory synapses can be suppressed with `CL_SDK_TWIN_GABA_BLOCK=1.0`.
Dopamine modulation can be enabled with `CL_SDK_TWIN_DOPAMINE`; in SNN mode it
gates STDP strength and slightly increases cell recruitment while staying
bounded.

Cell-level STDP can be enabled with:

```bash
CL_SDK_TWIN_PLASTICITY=stdp
```

### SNN reset platform

The SDK also includes an experimental CL1-constrained train-reset-relearn SNN
screening platform under `cl.snn_reset`. It models a spatial recurrent E/I
culture behind a 64-channel electrode interface, colored pulse reset protocols,
weight/activity/behavioral erasure metrics, trace-probe features, and parallel
protocol sweeps.

The same SNN interface can be selected inside the biological twin producer:

```bash
CL_SDK_SIM_MODE=surrogate
CL_SDK_TWIN_DYNAMICS=snn_reset
```

For offline reset sweeps, use `cl.snn_reset.run_trial()` or
`cl.snn_reset.run_sweep()`. A benchmark entrypoint is provided at
`scripts/benchmark_snn_reset.py`.

Cell-level short-term depression/facilitation can be enabled with:

```bash
CL_SDK_TWIN_PLASTICITY=stp
```

Closed-loop experiments can use `cl.twin.TwinFeedbackProtocol` to compile
correct outcomes into structured sensory-to-motor feedback and incorrect
outcomes into broad chaotic stimulation patterns, both routed through the same
twin stimulation coupling.
Use `cl.twin.TwinAcceleratedTrainer` for offline closed-loop training runs that
advance the same twin model without wall-clock sleeps, and
`cl.twin.TwinLearningEvaluator` to compare early and late task windows and
validate whether closed-loop performance improved.

### WebSocket server

An included webSocket server can be used to stream simulated data. By default, the server is disabled. It can be enabled by setting `CL_SDK_WEBSOCKET=1` environment variable in the `.env` file. The port used by the server can be set using the `CL_SDK_WEBSOCKET_PORT` environment variable (default 1025). The server will be hosted on `localhost` by default, but this can be changed by setting the `CL_SDK_WEBSOCKET_HOST` environment variable.
