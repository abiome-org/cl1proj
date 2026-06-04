# Learned Inverse Reset Report

## Run Metadata

- Run ID: `inverse_reset_actuator_causality_10k`
- Mode: `full`
- Dataset examples: `18`
- Candidates: `2`

## Candidate Protocols

| protocol_id                               | program_family            | duration_s | energy_cost | predicted_loss | predicted_task_erasure | predicted_health_penalty | model_uncertainty |
| ----------------------------------------- | ------------------------- | ---------- | ----------- | -------------- | ---------------------- | ------------------------ | ----------------- |
| optimized_00001_rest                      | rest                      | 0.5        | 0           | 6.5707         | -0.10826               | 0.37803                  | 0.031249          |
| optimized_00000_actuator_positive_control | actuator_positive_control | 0.5        | 5.12        | 1.4191e+05     | -206.62                | 178.27                   | 0.031249          |

## Paired Validation

| protocol_id                               | program_family            | validated_causal_task_erasure | validated_weight_erasure | validated_path_erasure | validated_residual_performance | validated_savings | validated_trace_auc_proxy | validated_health | validated_orthogonal_damage | stimulus_effect_norm | beats_no_reset | passes_health_criterion | passes_generalization_criterion |
| ----------------------------------------- | ------------------------- | ----------------------------- | ------------------------ | ---------------------- | ------------------------------ | ----------------- | ------------------------- | ---------------- | --------------------------- | -------------------- | -------------- | ----------------------- | ------------------------------- |
| optimized_00001_rest                      | rest                      | 0                             | -0.81565                 | -0.81555               | 0.58333                        | 0.41667           | 0.5462                    | 0.20201          | 0.2255                      | 0                    | 0              | 0.66667                 | 0                               |
| optimized_00000_actuator_positive_control | actuator_positive_control | -209.21                       | -2.2839                  | 0.048041               | 0.58333                        | 0.41667           | 0.97473                   | 0.005009         | 9888.9                      | 9891.2               | 0              | 0                       | 0                               |

## Failure Cases

| protocol_id                               | program_family            | validated_causal_task_erasure | validated_health | stimulus_effect_norm | beats_no_reset |
| ----------------------------------------- | ------------------------- | ----------------------------- | ---------------- | -------------------- | -------------- |
| optimized_00001_rest                      | rest                      | 0                             | 0.20201          | 0                    | 0              |
| optimized_00000_actuator_positive_control | actuator_positive_control | -209.21                       | 0.005009         | 9891.2               | 0              |

## Next Grammar Changes

Outcome B: stimulation is causal but not yet anti-trace aligned. Expand anti-causal timing, probe-triggered, and stronger positive-control families.
