# Learned Inverse Reset Controllability

## Run Metadata

- Run ID: `inverse_reset_reachability_10k`
- State vector version: `hybrid_v1`
- State vector hash: `2ae55702c4bd8628`
- Dataset size: `71`
- Model RMSE: `0.46977`
- Mean cosine similarity: `0.17881`

## Reachable Subspace

- Target anti-trace norm: `1.5577`
- Reachable norm: `1.51948`
- Unreachable residual norm: `0.342904`
- Controllable fraction: `0.975469`
- Best linear predicted reset loss: `2.77909e-05`
- Recommendation: Strong reachable-subspace signal; prioritize validation and generalization.
- Scientific criterion status: Actuator causality and reachability criteria pass: proceed to candidate validation before claiming reset.

## Controllability By Seed

| seed | example_index | controllable_fraction | target_norm | reachable_norm | unreachable_residual_norm |
| ---- | ------------- | --------------------- | ----------- | -------------- | ------------------------- |
| 1    | 0             | 0.97547               | 1.5577      | 1.5195         | 0.3429                    |
| 3    | 27            | 0.98011               | 1.5509      | 1.5201         | 0.3078                    |
| 5    | 48            | 0.97667               | 1.9241      | 1.8792         | 0.41318                   |

## Actuator Causality By Family

| program_family   | examples | mean_stimulus_effect_norm | max_stimulus_effect_norm | mean_spike_delta | mean_weight_delta_norm | mean_energy_cost |
| ---------------- | -------- | ------------------------- | ------------------------ | ---------------- | ---------------------- | ---------------- |
| task_input_drive | 30       | 58.225                    | 335.35                   | 1160.4           | 0.90457                | 0.4416           |
| rest             | 41       | 0                         | 0                        | 0                | 0                      | 0                |

## Top Aligned Stimulation Dimensions

| dimension                     | score    |
| ----------------------------- | -------- |
| family:rest                   | 4.1044   |
| mean_amplitude_uA             | 0.62     |
| max_amplitude_uA              | 0.62     |
| rest_fraction                 | 0.25804  |
| rough_event_count             | 0.11138  |
| low_frequency_hz              | 0.044622 |
| training_response_probability | 0.021239 |
| colored_beta                  | 0        |
| colored_event_rate_hz         | 0        |
| family:probe_triggered        | 0        |

## Top Anti-Aligned Stimulation Dimensions

| dimension                | score     |
| ------------------------ | --------- |
| rough_energy_uC          | -37.248   |
| family:task_input_drive  | -4.2115   |
| energy_cost              | -1.9005   |
| electrode_coverage       | -1.8689   |
| electrode_entropy        | -0.052659 |
| total_duration_s         | -0.026685 |
| pulse_width_us           | -0.021058 |
| block_count              | -0.013342 |
| anti_pair_delay_ms       | 0         |
| family:anti_stdp_pairing | 0         |

## Feature Group Reachability

| feature_group                | target_norm | reachable_norm | unreachable_residual_norm | controllable_fraction |
| ---------------------------- | ----------- | -------------- | ------------------------- | --------------------- |
| readout                      | 1.5432      | 1.5088         | 0.20133                   | 0.97775               |
| privileged_weight_projection | 0.19768     | 0.0048262      | 0.19694                   | 0.024414              |
| task_path                    | 0.077378    | 0.00025604     | 0.077483                  | 0.0033089             |
| channel_path                 | 0           | 0.00043262     | 0.00043262                | 0                     |
| evoked_activity              | 0           | 0.17747        | 0.17747                   | 0                     |
| weight_histogram             | 0           | 1.7752e-05     | 1.7752e-05                | 0                     |
| health                       | 0           | 0.026254       | 0.026254                  | 0                     |
| criticality                  | 0           | 0.0086789      | 0.0086789                 | 0                     |
| cost_context                 | 0           | 0              | 0                         | 0                     |
