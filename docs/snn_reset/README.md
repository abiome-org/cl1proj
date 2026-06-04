# SNN Reset Notes

The reset simulator is implemented in `src/cl1_snn_reset`.

Its public API is centered on:

- `CultureConfig` and `build_network` for constructing scaled MEA cultures.
- `StimEvent`, `ResetProtocol`, and `protocol_events` for channel-level reset
  stimulation.
- `run_trial` and `run_sweep` for train-reset-relearn screening.
- `rank_protocols` and `pareto_front` for protocol selection.

Reference docs in this folder:

- [`simulator.md`](simulator.md) — the forward simulation engine (config,
  network, electrodes, protocols, colored noise, the train-reset-relearn trial).
- [`metrics.md`](metrics.md) — reset metrics, scoring, and multi-objective
  protocol selection.
- [`control_checks.md`](control_checks.md) — the reset-vs-no-reset and
  noise/actuator diagnostics study.
- [`full_grid_10k_calibrated_20260602.md`](full_grid_10k_calibrated_20260602.md)
  — the calibrated full-grid screening report.

Learned inverse reset control is documented separately in
[`src/cl1_snn_reset/inverse_control/README.md`](../../src/cl1_snn_reset/inverse_control/README.md).

The calibrated 10k-neuron full-grid report is in
`docs/snn_reset/full_grid_10k_calibrated_20260602.md`.  The headline result is
negative for true reset: the best protocols were low-burden red/epoch-pause
variants, but every protocol moved hidden weights farther from the naive state
rather than back toward it.
