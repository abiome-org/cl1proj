# `inverse_control` — learned inverse reset control

This subpackage of `cl1_snn_reset` solves the **inverse problem of reset
control** for CL1-style spiking neural cultures. The rest of the library runs
stimulation forward — apply a protocol, observe what it does to the culture.
This package runs the question backward:

> Given a desired post-reset culture state (for example: erase a learned task
> while keeping the network healthy), what stimulation protocol should we apply
> to get there?

It exists so that protocols can be designed, analyzed for feasibility, and
validated in simulation *before* spending time on real 64-channel MEA wetware
runs.

It is **library code**. Import it through the public API
(`from cl1_snn_reset.inverse_control import ...`); the runnable driver and its
YAML configs live in `experiments/regression1/learned_inverse_reset.py`, where it
also serves as a controllability regression gate.

## The pipeline

The package is organized as a sequence of composable stages. The driver wires
them together; each stage is independently importable and testable.

### 1. State representation — `state_projectors.py`

Projects a culture's spiking activity into a fixed-length feature vector
described by a `StateVectorSpec`: activity statistics, criticality, health
metrics, task-trace AUC, and so on. The spec tags each feature as either
**wetware-observable** (measurable on real hardware) or **privileged**
(available only inside the simulator), so downstream code can be honest about
what a real experiment could actually see. `build_target_state` constructs the
goal vector the optimizer aims for. `HybridStateProjector` is the default
projector.

### 2. Stimulation "language" — `blocks.py`, `stim_sampling.py`, `pulse_compiler.py`, `program_features.py`

A grammar of stimulation **blocks** — anti-STDP pairing, probe-triggered,
coordinated reset, low-frequency depotentiation, colored-noise, actuator
positive control, task-input drive, and rest — assembled into a `StimProgram`.

- `StimConstraints` encodes hardware safety limits: max amplitude and charge,
  pulse width, inter-pulse interval, per-electrode cooldown and event count,
  total duration, energy budget, allowed electrodes, and charge balancing.
- `pulse_compiler.compile_program_to_stim_events` lowers a program into
  concrete `StimEvent`s, `validate_stim_program` checks it against the
  constraints (raising `InvalidStimProgramError`), and `estimate_energy_cost`
  scores its cost.
- `program_features.stim_program_features` vectorizes a program (feature names
  in `STIM_FEATURE_NAMES`) for the forward model.

### 3. Dataset generation — `stim_sampling.py`, `rollout_dataset.py`, `training_rollout.py`

Samples many random stim programs (`sample_stim_programs`,
`StimSamplingConfig`), rolls each one through the simulator on top of a trained
baseline, and records the **causal delta** — how each program changed the state
vector relative to a no-reset control. The result is a `CausalDeltaDataset`
(built by `CausalDeltaDatasetBuilder`, one `RolloutExample` per seed/program).

### 4. Forward model — `forward_models.py`

Learns the mapping *stimulation → state change*. `RidgeDeltaModel` is the
working model; `MeanZeroDeltaModel` is the baseline to beat. Models expose
`predict_delta` and `predict_uncertainty`, and `evaluate_forward_model` scores
fit quality on held-out data.

### 5. Controllability analysis — `controllability.py`

A linear-Jacobian analysis of *which* state dimensions the available actuators
can actually reach. `analyze_controllability` returns a `ControllabilityReport`
with the controllable fraction, reachable vs. unreachable residual norms, the
top aligned and anti-aligned dimensions, per-group reachability, and a
plain-text recommendation. This is the stage used as a regression gate — it
answers "is this target even achievable?" before optimization is attempted.

### 6. Inverse optimization — `inverse_optimizer.py`

Searches the program space to minimize an `InverseResetObjective`. Available
optimizers: `RandomSearchOptimizer`, `EliteMutationStimOptimizer`, and
`CMAESStimOptimizer`. The objective is a weighted sum of competing terms,
evaluated against the predicted post-state, the no-reset state, and the target:

| Term | Default weight | Meaning |
|------|----------------|---------|
| `task_trace` | 3.0 | drive task/readout features to target (erase the task) |
| `input_target_path` | 3.0 | erase the learned input→target path |
| `privileged_weight_projection` | 2.0 | match the privileged weight-state target |
| `health` | 2.0 | keep health/criticality features near target |
| `off_target_drift` | 1.5 | penalize disturbing dimensions that should be left alone |
| `savings_proxy` | 2.0 | penalize residual task that would relearn quickly |
| `energy` | 0.2 | penalize energy cost (normalized by budget) |
| `uncertainty` | 1.0 | penalize regions where the forward model is unsure |

The output is a ranked list of `CandidateProtocol`s, each carrying its predicted
delta, predicted post-state, and the per-term loss breakdown.

### 7. Validation — `validation.py`

Re-runs the top candidates through the *full* simulator (not the surrogate
forward model) and compares each against a no-reset control with bootstrap
effect sizes (`validate_candidate_against_no_reset`,
`validate_candidates_against_no_reset`, `bootstrap_candidate_effects`). This
guards against the optimizer winning by exploiting flaws in the learned model
rather than producing a protocol that genuinely works.

### 8. Reporting — `reporting.py`

Writes the run outputs: `write_candidate_csv` (ranked candidates),
`write_controllability_artifacts` (Jacobian/reachability arrays), and
`write_inverse_reset_report` / `write_markdown_report` (the human-readable
report).

## Data flow at a glance

```
target state (build_target_state)
        │
sample programs ──► roll out in sim ──► CausalDeltaDataset
        │                                      │
        │                              fit forward model (RidgeDeltaModel)
        │                                      │
        ├────────────► controllability analysis (is the target reachable?)
        │                                      │
        └──► inverse optimizer ──► ranked CandidateProtocols
                                               │
                                  full-sim validation vs. no-reset control
                                               │
                                     CSV + controllability + report
```

## Public API

The package `__init__` re-exports the stable surface. Prefer these over reaching
into internal modules:

`StimProgram`, `StimConstraints`, `StimSamplingConfig`, `sample_stim_programs`,
`stim_program_features`, `STIM_FEATURE_NAMES`, `compile_program_to_stim_events`,
`validate_stim_program`, `estimate_energy_cost`, `InvalidStimProgramError`,
`StateProjector`, `HybridStateProjector`, `StateVectorSpec`, `build_target_state`,
`CausalDeltaDataset`, `CausalDeltaDatasetBuilder`, `RolloutExample`,
`RidgeDeltaModel`, `MeanZeroDeltaModel`, `evaluate_forward_model`,
`analyze_controllability`, `ControllabilityReport`, `InverseResetObjective`,
`CandidateProtocol`, `RandomSearchOptimizer`, `EliteMutationStimOptimizer`,
`CMAESStimOptimizer`, `validate_candidate_against_no_reset`,
`bootstrap_candidate_effects`.

## Running it

```bash
python experiments/regression1/learned_inverse_reset.py \
    --config experiments/regression1/configs/inverse_reset_smoke.yaml
```

Configs of increasing scope live in `experiments/regression1/configs/`
(`inverse_reset_smoke.yaml`, `inverse_reset_mvp.yaml`,
`inverse_reset_controllability.yaml`, `inverse_reset_reachability.yaml`,
`inverse_reset_actuator_causality.yaml`, `inverse_reset_closed_loop.yaml`).
Outputs are written under `experiments/regression1/results/`. Tests live in
`tests/snn_reset/inverse_control/`.
