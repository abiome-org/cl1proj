# SNN Reset Notes

The reset simulator is implemented in `src/cl1_snn_reset`.

Its public API is centered on:

- `CultureConfig` and `build_network` for constructing scaled MEA cultures.
- `StimEvent`, `ResetProtocol`, and `protocol_events` for channel-level reset
  stimulation.
- `run_trial` and `run_sweep` for train-reset-relearn screening.
- `rank_protocols` and `pareto_front` for protocol selection.
