# Full Grid 10k Calibrated Run

This report summarizes the calibrated 10,000-neuron reset sweep.

## Run Setup

- Network: 10,000 neurons, mean out-degree 64, 64 electrodes
- Task: input channel 8 to target channel 9
- Criterion: response probability 0.875 over 8 evaluation trials
- Seeds: 1, 3, 4
- Protocols: 540
- Jobs: 1,620

## Data Quality

| Check | Count |
| --- | ---: |
| Initial-at-criterion rows | 0 |
| Initial max-trial rows | 0 |
| Relearn-at-criterion rows | 0 |
| Completed jobs | 1,620 / 1,620 |

The calibrated input-target pair avoided the earlier failure mode where trials
started at criterion or never reached the initial learning criterion.

## Top Ranked Protocols

| Protocol | Reset score | Weight erasure | Residual performance | Savings | Trace AUC | Health | Energy cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b2_epoch_pause_independent_0.75s_0.8uA` | -0.4648 | -0.4373 | 0.5000 | -1.3889 | 0.5307 | 0.0996 | 0.0048 |
| `b1_epoch_pause_independent_0.75s_0.8uA` | -0.4649 | -0.4373 | 0.5000 | -1.3889 | 0.5307 | 0.0996 | 0.0067 |
| `b2_static_independent_0.75s_0.8uA` | -0.4649 | -0.4373 | 0.5000 | -1.3889 | 0.5307 | 0.0996 | 0.0070 |
| `b0_epoch_pause_independent_0.75s_0.8uA` | -0.4650 | -0.4373 | 0.5000 | -1.3889 | 0.5307 | 0.0996 | 0.0080 |
| `b2_epoch_pause_independent_0.75s_1.6uA` | -0.4650 | -0.4373 | 0.5000 | -1.3889 | 0.5307 | 0.0996 | 0.0096 |

The top rows are nearly tied on scientific metrics.  Their ordering is mostly
set by stimulation cost.

## Pareto Front

| Protocol | Duration | Current | Weight erasure | Residual performance | Savings | Trace AUC | Health | Energy cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b2_epoch_pause_independent_0.75s_0.8uA` | 0.75s | 0.8uA | -0.4373 | 0.5000 | -1.3889 | 0.5307 | 0.0996 | 0.0048 |
| `b2_epoch_pause_independent_1.5s_0.8uA` | 1.50s | 0.8uA | -0.8537 | 0.6667 | -0.6111 | 0.5342 | 0.1003 | 0.0096 |
| `b2_epoch_pause_independent_3s_0.8uA` | 3.00s | 0.8uA | -1.8658 | 0.4583 | -0.4444 | 0.5325 | 0.1408 | 0.0188 |

All Pareto candidates are red, independent, epoch/pause protocols at the lowest
current.  Longer duration improved some health and savings terms but made
weight-space drift away from naive substantially worse.

## Factor-Level Means

By temporal color, all scientific metrics were effectively identical; beta only
changed pulse burden:

| Beta | Mean reset score | Mean weight erasure | Mean residual performance | Mean savings | Mean energy cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | -2.9284 | -1.0522 | 0.5417 | -0.8148 | 0.1287 |
| 1 | -2.9294 | -1.0522 | 0.5417 | -0.8148 | 0.1493 |
| 0 | -2.9304 | -1.0522 | 0.5417 | -0.8148 | 0.1679 |
| -1 | -2.9317 | -1.0522 | 0.5417 | -0.8148 | 0.1951 |
| -2 | -2.9331 | -1.0522 | 0.5417 | -0.8148 | 0.2232 |

By schedule, epoch/pause had the lowest average cost:

| Schedule | Mean reset score | Mean weight erasure | Mean residual performance | Mean savings | Mean energy cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `epoch_pause` | -2.9291 | -1.0522 | 0.5417 | -0.8148 | 0.1422 |
| `alternating_blue_red` | -2.9310 | -1.0522 | 0.5417 | -0.8148 | 0.1795 |
| `static` | -2.9318 | -1.0522 | 0.5417 | -0.8148 | 0.1970 |

By spatial mode, independent stimulation had the lowest average cost:

| Spatial mode | Mean reset score | Mean weight erasure | Mean residual performance | Mean savings | Mean energy cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `independent` | -2.9244 | -1.0522 | 0.5417 | -0.8148 | 0.0478 |
| `correlated` | -2.9303 | -1.0522 | 0.5417 | -0.8148 | 0.1668 |
| `phase_shifted` | -2.9315 | -1.0522 | 0.5417 | -0.8148 | 0.1904 |
| `shared` | -2.9363 | -1.0522 | 0.5417 | -0.8148 | 0.2866 |

## Interpretation

The full grid did not find a true reset protocol for this simulator setting.
Negative weight-erasure values mean post-reset weights moved farther from the
naive state instead of back toward it.  Behavioral chance performance and a
near-random trace probe were therefore not enough to establish erasure.

The most useful outcome is the falsification boundary: with this fixed-sign
STDP model, this task, and this electrode-only actuator space, colored
stimulation did not restore the culture to a naive weight landscape.  Future
screens should vary training depth, consolidation delay, homeostatic parameters,
and stronger protocol families before carrying candidates into wetware.
