# Full SNN Reset Grid Search

## Run Metadata

- Run ID: `full_grid_10k_calibrated_20260602T042050Z`
- Git commit: `0b0778699e801eff68fd1a6ba0408268e335a7dc`
- Neurons: `10000`
- Mean out-degree: `64`
- Protocols: `540`
- Seeds: `[1, 3, 4]`
- Jobs completed: `1620 / 1620`
- Workers: `8` via `process` executor
- Elapsed seconds: `3725.12`
- Jobs per second: `0.4349`

## Data Quality

- Initial-at-criterion rows: `0`
- Relearn-at-criterion rows: `0`
- Initial max-trial rows: `0`

## Objective

This run screens the full coarse reset protocol grid on a 10000-neuron
CL1-style SNN.  Protocols are compared on true simulation-only weight
erasure and CL1-like readouts: behavior after reset, relearning savings,
trace detectability, health, criticality distance, and stimulation cost.

## Top Ranked Protocols

| protocol_id                                      | reset_score | weight_erasure | path_erasure | residual_performance | savings | trace_auc_proxy | health  | energy_cost | replicates |
| ------------------------------------------------ | ----------- | -------------- | ------------ | -------------------- | ------- | --------------- | ------- | ----------- | ---------- |
| b2_epoch_pause_independent_0.75s_0.8uA           | -0.4648     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.004779    | 3          |
| b1_epoch_pause_independent_0.75s_0.8uA           | -0.4649     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.006741    | 3          |
| b2_static_independent_0.75s_0.8uA                | -0.4649     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.006997    | 3          |
| b0_epoch_pause_independent_0.75s_0.8uA           | -0.465      | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.008021    | 3          |
| b2_epoch_pause_independent_0.75s_1.6uA           | -0.465      | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.009557    | 3          |
| b1_static_independent_0.75s_0.8uA                | -0.465      | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.009557    | 3          |
| b-1_epoch_pause_independent_0.75s_0.8uA          | -0.4651     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01015     | 3          |
| b0_alternating_blue_red_independent_0.75s_0.8uA  | -0.4651     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01058     | 3          |
| b1_alternating_blue_red_independent_0.75s_0.8uA  | -0.4651     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01058     | 3          |
| b2_alternating_blue_red_independent_0.75s_0.8uA  | -0.4651     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01058     | 3          |
| b-1_alternating_blue_red_independent_0.75s_0.8uA | -0.4651     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01058     | 3          |
| b-2_alternating_blue_red_independent_0.75s_0.8uA | -0.4651     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01058     | 3          |
| b-2_epoch_pause_independent_0.75s_0.8uA          | -0.4651     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01178     | 3          |
| b0_static_independent_0.75s_0.8uA                | -0.4652     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01203     | 3          |
| b1_epoch_pause_independent_0.75s_1.6uA           | -0.4652     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01348     | 3          |
| b2_static_independent_0.75s_1.6uA                | -0.4653     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01399     | 3          |
| b-1_static_independent_0.75s_0.8uA               | -0.4653     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01425     | 3          |
| b2_epoch_pause_independent_0.75s_2.6uA           | -0.4653     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01553     | 3          |
| b0_epoch_pause_independent_0.75s_1.6uA           | -0.4654     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01604     | 3          |
| b-2_static_independent_0.75s_0.8uA               | -0.4654     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.01681     | 3          |

## Pareto Front

| protocol_id                            | reset_score | weight_erasure | path_erasure | residual_performance | savings | trace_auc_proxy | health  | energy_cost | replicates |
| -------------------------------------- | ----------- | -------------- | ------------ | -------------------- | ------- | --------------- | ------- | ----------- | ---------- |
| b2_epoch_pause_independent_0.75s_0.8uA | -0.4648     | -0.4373        | -0.4372      | 0.5                  | -1.389  | 0.5307          | 0.09958 | 0.004779    | 3          |
| b2_epoch_pause_independent_1.5s_0.8uA  | -2.696      | -0.8537        | -0.8537      | 0.6667               | -0.6111 | 0.5342          | 0.1003  | 0.009643    | 3          |
| b2_epoch_pause_independent_3s_0.8uA    | -5.648      | -1.866         | -1.866       | 0.4583               | -0.4444 | 0.5325          | 0.09959 | 0.01877     | 3          |

## Best Settings By Factor

### By Beta

| beta | protocols | reset_score | weight_erasure | path_erasure | residual_performance | savings | trace_auc_proxy | health  | energy_cost |
| ---- | --------- | ----------- | -------------- | ------------ | -------------------- | ------- | --------------- | ------- | ----------- |
| 2    | 108       | -2.942      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1287      |
| 1    | 108       | -2.943      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1493      |
| 0    | 108       | -2.944      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1679      |
| -1   | 108       | -2.945      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1951      |
| -2   | 108       | -2.947      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.2232      |

### By Schedule

| schedule             | protocols | reset_score | weight_erasure | path_erasure | residual_performance | savings | trace_auc_proxy | health  | energy_cost |
| -------------------- | --------- | ----------- | -------------- | ------------ | -------------------- | ------- | --------------- | ------- | ----------- |
| epoch_pause          | 180       | -2.943      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1422      |
| alternating_blue_red | 180       | -2.945      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1795      |
| static               | 180       | -2.946      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.197       |

### By Spatial Mode

| spatial_mode  | protocols | reset_score | weight_erasure | path_erasure | residual_performance | savings | trace_auc_proxy | health  | energy_cost |
| ------------- | --------- | ----------- | -------------- | ------------ | -------------------- | ------- | --------------- | ------- | ----------- |
| independent   | 135       | -2.938      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.04776     |
| correlated    | 135       | -2.944      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1668      |
| phase_shifted | 135       | -2.945      | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.1904      |
| shared        | 135       | -2.95       | -1.052         | -1.052       | 0.5417               | -0.8148 | 0.5324          | 0.09981 | 0.2866      |

## Protocol Parameter View

| beta | schedule             | spatial_mode  | duration_s | current_uA | reset_score | weight_erasure | residual_performance | savings | trace_auc_proxy | health  |
| ---- | -------------------- | ------------- | ---------- | ---------- | ----------- | -------------- | -------------------- | ------- | --------------- | ------- |
| 2    | epoch_pause          | independent   | 0.75       | 0.8        | -0.4648     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 1    | epoch_pause          | independent   | 0.75       | 0.8        | -0.4649     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | static               | independent   | 0.75       | 0.8        | -0.4649     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 0    | epoch_pause          | independent   | 0.75       | 0.8        | -0.465      | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | epoch_pause          | independent   | 0.75       | 1.6        | -0.465      | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 1    | static               | independent   | 0.75       | 0.8        | -0.465      | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -1   | epoch_pause          | independent   | 0.75       | 0.8        | -0.4651     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 0    | alternating_blue_red | independent   | 0.75       | 0.8        | -0.4651     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 1    | alternating_blue_red | independent   | 0.75       | 0.8        | -0.4651     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | alternating_blue_red | independent   | 0.75       | 0.8        | -0.4651     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -1   | alternating_blue_red | independent   | 0.75       | 0.8        | -0.4651     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -2   | alternating_blue_red | independent   | 0.75       | 0.8        | -0.4651     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -2   | epoch_pause          | independent   | 0.75       | 0.8        | -0.4651     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 0    | static               | independent   | 0.75       | 0.8        | -0.4652     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 1    | epoch_pause          | independent   | 0.75       | 1.6        | -0.4652     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | static               | independent   | 0.75       | 1.6        | -0.4653     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -1   | static               | independent   | 0.75       | 0.8        | -0.4653     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | epoch_pause          | independent   | 0.75       | 2.6        | -0.4653     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 0    | epoch_pause          | independent   | 0.75       | 1.6        | -0.4654     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -2   | static               | independent   | 0.75       | 0.8        | -0.4654     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | epoch_pause          | correlated    | 0.75       | 0.8        | -0.4655     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 1    | static               | independent   | 0.75       | 1.6        | -0.4655     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | epoch_pause          | phase_shifted | 0.75       | 0.8        | -0.4655     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -1   | epoch_pause          | independent   | 0.75       | 1.6        | -0.4656     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 0    | alternating_blue_red | independent   | 0.75       | 1.6        | -0.4656     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -2   | alternating_blue_red | independent   | 0.75       | 1.6        | -0.4656     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 1    | alternating_blue_red | independent   | 0.75       | 1.6        | -0.4656     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 2    | alternating_blue_red | independent   | 0.75       | 1.6        | -0.4656     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| -1   | alternating_blue_red | independent   | 0.75       | 1.6        | -0.4656     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |
| 1    | epoch_pause          | independent   | 0.75       | 2.6        | -0.4657     | -0.4373        | 0.5                  | -1.389  | 0.5307          | 0.09958 |

## Artifacts

- `raw_trials.csv`: every protocol x seed trial.
- `summary.csv`: replicate means by protocol.
- `ranked.csv`: summary with scalar screen score.
- `pareto.csv`: nondominated protocol rows.
- `metadata.json`: run configuration and progress metadata.
