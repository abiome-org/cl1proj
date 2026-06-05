# SNN Reset Control Checks

## Setup

- Neurons: `10000`
- Seeds: `[1, 3, 4]`
- Protocols: `['low_burden_0.75s', 'mid_burden_1.5s', 'high_burden_3s']`

## Noise Spectra

| beta | target_power_slope | estimated_power_slope_mean | estimated_power_slope_std | event_count_mean | total_pulses_mean |
| ---- | ------------------ | -------------------------- | ------------------------- | ---------------- | ----------------- |
| -2   | 2                  | 1.9988                     | 0.023425                  | 259.5            | 937.5             |
| -1   | 1                  | 0.99982                    | 0.021234                  | 205              | 706.75            |
| 0    | -0                 | -0.0010012                 | 0.022824                  | 175.5            | 620               |
| 1    | -1                 | -1.0008                    | 0.029208                  | 144.75           | 530.75            |
| 2    | -2                 | -1.9924                    | 0.024348                  | 120.25           | 430.5             |

## Trained Reset Versus No Reset

| control_mode | source_protocol_id | weight_erasure | residual_performance | savings  | trace_auc_proxy | post_delta_norm | reset_window_neuron_spikes | total_pulses |
| ------------ | ------------------ | -------------- | -------------------- | -------- | --------------- | --------------- | -------------------------- | ------------ |
| no_reset     | high_burden_3s     | -1.8658        | 0.45833              | -0.44444 | 0.53251         | 143.65          | 3579                       | 0            |
| reset        | high_burden_3s     | -1.8658        | 0.45833              | -0.44444 | 0.53251         | 143.65          | 3579                       | 1513.7       |
| no_reset     | low_burden_0.75s   | -0.43727       | 0.5                  | -1.3889  | 0.53065         | 78.382          | 888                        | 0            |
| reset        | low_burden_0.75s   | -0.43727       | 0.5                  | -1.3889  | 0.53065         | 78.382          | 888                        | 18.667       |
| no_reset     | mid_burden_1.5s    | -0.85366       | 0.66667              | -0.61111 | 0.53418         | 97.407          | 1788.3                     | 0            |
| reset        | mid_burden_1.5s    | -0.85366       | 0.66667              | -0.61111 | 0.53418         | 97.407          | 1788.3                     | 465.33       |

## Reset Minus No Reset

| source_protocol_id | weight_erasure_delta | residual_performance_delta | savings_delta | trace_auc_proxy_delta | post_delta_norm_delta | reset_window_neuron_spikes_delta |
| ------------------ | -------------------- | -------------------------- | ------------- | --------------------- | --------------------- | -------------------------------- |
| high_burden_3s     | 0                    | 0                          | 0             | 0                     | 0                     | 0                                |
| low_burden_0.75s   | 0                    | 0                          | 0             | 0                     | 0                     | 0                                |
| mid_burden_1.5s    | 0                    | 0                          | 0             | 0                     | 0                     | 0                                |

## Untrained Reset Controls

| control_mode | source_protocol_id | weight_drift_norm | weight_drift_rel_to_baseline | pre_behavior | post_behavior | trace_auc_proxy | reset_window_neuron_spikes | total_pulses |
| ------------ | ------------------ | ----------------- | ---------------------------- | ------------ | ------------- | --------------- | -------------------------- | ------------ |
| no_reset     | high_burden_3s     | 61.586            | 0.31066                      | 0.41667      | 0.54167       | 0.52965         | 3581.7                     | 0            |
| reset        | high_burden_3s     | 61.586            | 0.31066                      | 0.41667      | 0.54167       | 0.52965         | 3581.7                     | 1513.7       |
| no_reset     | low_burden_0.75s   | 14.448            | 0.072879                     | 0.41667      | 0.5           | 0.53826         | 892                        | 0            |
| reset        | low_burden_0.75s   | 14.448            | 0.072879                     | 0.41667      | 0.5           | 0.53826         | 892                        | 18.667       |
| no_reset     | mid_burden_1.5s    | 28.198            | 0.14224                      | 0.41667      | 0.58333       | 0.54268         | 1792.7                     | 0            |
| reset        | mid_burden_1.5s    | 28.198            | 0.14224                      | 0.41667      | 0.58333       | 0.54268         | 1792.7                     | 465.33       |
