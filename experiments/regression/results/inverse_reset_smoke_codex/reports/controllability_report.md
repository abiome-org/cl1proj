# Learned Inverse Reset Controllability

## Run Metadata

- Run ID: `inverse_reset_smoke_codex`
- State vector version: `hybrid_v1`
- State vector hash: `a56a7923ff5bf812`
- Dataset size: `24`
- Model RMSE: `0`
- Mean cosine similarity: `0`

## Reachable Subspace

- Target anti-trace norm: `0.0500615`
- Reachable norm: `0`
- Unreachable residual norm: `0.0500615`
- Controllable fraction: `0`
- Best linear predicted reset loss: `5.96846e-07`
- Recommendation: Outcome B risk: actuator grammar appears unable to reach the anti-trace direction.
- Scientific criterion status: Actuator causality criterion fails: no sampled stimulation family produced causal change beyond no-reset.

## Controllability By Seed

| seed | example_index | controllable_fraction | target_norm | reachable_norm | unreachable_residual_norm |
| ---- | ------------- | --------------------- | ----------- | -------------- | ------------------------- |
| 1    | 0             | 0                     | 0.050062    | 0              | 0.050062                  |
| 3    | 12            | 0                     | 0.058922    | 0              | 0.058922                  |

## Actuator Causality By Family

| program_family               | examples | mean_stimulus_effect_norm | max_stimulus_effect_norm | mean_spike_delta | mean_weight_delta_norm | mean_energy_cost |
| ---------------------------- | -------- | ------------------------- | ------------------------ | ---------------- | ---------------------- | ---------------- |
| anti_stdp_pairing            | 7        | 0                         | 0                        | 0                | 0                      | 0.0012434        |
| colored_noise                | 4        | 0                         | 0                        | 0                | 0                      | 0.002656         |
| low_frequency_depotentiation | 8        | 0                         | 0                        | 0                | 0                      | 0.001872         |
| rest                         | 5        | 0                         | 0                        | 0                | 0                      | 0                |

## Top Aligned Stimulation Dimensions

| dimension                           | score |
| ----------------------------------- | ----- |
| training_response_probability       | 0     |
| rough_energy_uC                     | 0     |
| family:probe_triggered              | 0     |
| family:coordinated_reset            | 0     |
| family:low_frequency_depotentiation | 0     |
| family:colored_noise                | 0     |
| family:actuator_positive_control    | 0     |
| family:task_input_drive             | 0     |
| family:rest                         | 0     |
| total_duration_s                    | 0     |

## Top Anti-Aligned Stimulation Dimensions

| dimension                | score |
| ------------------------ | ----- |
| family:anti_stdp_pairing | 0     |
| duration_s               | 0     |
| rest_fraction            | 0     |
| electrode_entropy        | 0     |
| electrode_coverage       | 0     |
| colored_event_rate_hz    | 0     |
| colored_beta             | 0     |
| low_frequency_hz         | 0     |
| anti_pair_delay_ms       | 0     |
| pulse_width_us           | 0     |

## Feature Group Reachability

| feature_group                | target_norm | reachable_norm | unreachable_residual_norm | controllable_fraction |
| ---------------------------- | ----------- | -------------- | ------------------------- | --------------------- |
| channel_path                 | 0           | 0              | 0                         | 0                     |
| task_path                    | 0.021754    | 0              | 0.021754                  | 0                     |
| evoked_activity              | 0           | 0              | 0                         | 0                     |
| readout                      | 0           | 0              | 0                         | 0                     |
| weight_histogram             | 0           | 0              | 0                         | 0                     |
| privileged_weight_projection | 0.045088    | 0              | 0.045088                  | 0                     |
| health                       | 0           | 0              | 0                         | 0                     |
| criticality                  | 0           | 0              | 0                         | 0                     |
| cost_context                 | 0           | 0              | 0                         | 0                     |
