# Learned Inverse Reset Controllability

## Run Metadata

- Run ID: `inverse_reset_actuator_causality_10k`
- State vector version: `hybrid_v1`
- State vector hash: `2ae55702c4bd8628`
- Dataset size: `18`
- Model RMSE: `0.031249`
- Mean cosine similarity: `0.55556`

## Reachable Subspace

- Target anti-trace norm: `1.5577`
- Reachable norm: `0.04356`
- Unreachable residual norm: `1.55709`
- Controllable fraction: `0.0279643`
- Best linear predicted reset loss: `0.000573037`
- Recommendation: Outcome B risk: actuator grammar appears unable to reach the anti-trace direction.
- Scientific criterion status: Actuator causality criterion passes; reachability criterion fails because anti-trace reachability is weak.

## Controllability By Seed

| seed | example_index | controllable_fraction | target_norm | reachable_norm | unreachable_residual_norm |
| ---- | ------------- | --------------------- | ----------- | -------------- | ------------------------- |
| 1    | 0             | 0.027964              | 1.5577      | 0.04356        | 1.5571                    |
| 3    | 6             | 0.036798              | 1.5509      | 0.05707        | 1.5499                    |
| 5    | 12            | 0.011436              | 1.9241      | 0.022005       | 1.924                     |

## Actuator Causality By Family

| program_family            | examples | mean_stimulus_effect_norm | max_stimulus_effect_norm | mean_spike_delta | mean_weight_delta_norm | mean_energy_cost |
| ------------------------- | -------- | ------------------------- | ------------------------ | ---------------- | ---------------------- | ---------------- |
| actuator_positive_control | 10       | 9870.7                    | 12227                    | 2.4426e+05       | 54.204                 | 5.12             |
| rest                      | 8        | 0                         | 0                        | 0                | 0                      | 0                |

## Top Aligned Stimulation Dimensions

| dimension                           | score  |
| ----------------------------------- | ------ |
| family:rest                         | 12.493 |
| rest_fraction                       | 3.1939 |
| family:anti_stdp_pairing            | 0      |
| duration_s                          | 0      |
| family:probe_triggered              | 0      |
| family:coordinated_reset            | 0      |
| colored_event_rate_hz               | 0      |
| colored_beta                        | 0      |
| anti_pair_delay_ms                  | 0      |
| family:low_frequency_depotentiation | 0      |

## Top Anti-Aligned Stimulation Dimensions

| dimension                        | score    |
| -------------------------------- | -------- |
| family:actuator_positive_control | -11.083  |
| mean_amplitude_uA                | -3.9924  |
| max_amplitude_uA                 | -3.9924  |
| electrode_entropy                | -3.1939  |
| electrode_coverage               | -3.1939  |
| rough_energy_uC                  | -2.1647  |
| total_duration_s                 | -0.81623 |
| energy_cost                      | -0.62381 |
| training_response_probability    | -0.42531 |
| block_count                      | -0.40812 |

## Feature Group Reachability

| feature_group                | target_norm | reachable_norm | unreachable_residual_norm | controllable_fraction |
| ---------------------------- | ----------- | -------------- | ------------------------- | --------------------- |
| readout                      | 1.5432      | 0.004528       | 1.5419                    | 0.0029343             |
| privileged_weight_projection | 0.19768     | 8.896e-05      | 0.19768                   | 0.00045001            |
| task_path                    | 0.077378    | 7.2801e-06     | 0.077382                  | 9.4085e-05            |
| channel_path                 | 0           | 4.8711e-05     | 4.8711e-05                | 0                     |
| evoked_activity              | 0           | 0.04331        | 0.04331                   | 0                     |
| weight_histogram             | 0           | 1.7809e-05     | 1.7809e-05                | 0                     |
| health                       | 0           | 0.0001594      | 0.0001594                 | 0                     |
| criticality                  | 0           | 0.0010862      | 0.0010862                 | 0                     |
| cost_context                 | 0           | 0              | 0                         | 0                     |
