# Learned Inverse Reset Report

## Run Metadata

- Run ID: `inverse_reset_reachability_10k`
- Mode: `full`
- Dataset examples: `71`
- Candidates: `8`

## Candidate Protocols

| protocol_id                      | program_family   | duration_s | energy_cost | predicted_loss | predicted_task_erasure | predicted_health_penalty | model_uncertainty |
| -------------------------------- | ---------------- | ---------- | ----------- | -------------- | ---------------------- | ------------------------ | ----------------- |
| optimized_00036_task_input_drive | task_input_drive | 0.5        | 0.176       | 1.5328         | 0.79922                | 0.050662                 | 0.46977           |
| optimized_00001_rest             | rest             | 0.5        | 0           | 5.8012         | -0.00042436            | 0.0068084                | 0.46977           |
| optimized_00029_task_input_drive | task_input_drive | 0.5        | 0.16        | 6.2874         | -0.065544              | 0.059272                 | 0.46977           |
| optimized_00007_task_input_drive | task_input_drive | 0.5        | 0.154       | 10.834         | -0.58614               | 0.047131                 | 0.46977           |
| optimized_00074_task_input_drive | task_input_drive | 0.5        | 0.16        | 19.353         | -1.3139                | 0.19233                  | 0.46977           |
| optimized_00010_task_input_drive | task_input_drive | 0.5        | 0.7         | 21.135         | -1.4401                | 0.11498                  | 0.46977           |
| optimized_00019_task_input_drive | task_input_drive | 0.5        | 0.77        | 23.039         | -1.5692                | 0.22593                  | 0.46977           |
| optimized_00048_task_input_drive | task_input_drive | 0.5        | 0.14        | 23.924         | -1.6412                | 0.048647                 | 0.46977           |

## Paired Validation

| protocol_id                      | program_family   | validated_causal_task_erasure | validated_weight_erasure | validated_path_erasure | validated_residual_performance | validated_savings | validated_trace_auc_proxy | validated_health | validated_orthogonal_damage | stimulus_effect_norm | beats_no_reset | passes_health_criterion | passes_generalization_criterion |
| -------------------------------- | ---------------- | ----------------------------- | ------------------------ | ---------------------- | ------------------------------ | ----------------- | ------------------------- | ---------------- | --------------------------- | -------------------- | -------------- | ----------------------- | ------------------------------- |
| optimized_00001_rest             | rest             | 0                             | -0.81565                 | -0.81555               | 0.58333                        | 0.41667           | 0.5462                    | 0.20201          | 0.2255                      | 0                    | 0              | 0.66667                 | 0                               |
| optimized_00007_task_input_drive | task_input_drive | 0                             | -0.81565                 | -0.81555               | 0.58333                        | 0.41667           | 0.5462                    | 0.20201          | 0.2255                      | 0                    | 0              | 0.66667                 | 0                               |
| optimized_00029_task_input_drive | task_input_drive | 0                             | -0.81565                 | -0.81555               | 0.58333                        | 0.41667           | 0.5462                    | 0.20201          | 0.2255                      | 0                    | 0              | 0.66667                 | 0                               |
| optimized_00036_task_input_drive | task_input_drive | 0                             | -0.81565                 | -0.81555               | 0.58333                        | 0.41667           | 0.5462                    | 0.20201          | 0.2255                      | 0                    | 0              | 0.66667                 | 0                               |

## Failure Cases

| protocol_id                      | program_family   | validated_causal_task_erasure | validated_health | stimulus_effect_norm | beats_no_reset |
| -------------------------------- | ---------------- | ----------------------------- | ---------------- | -------------------- | -------------- |
| optimized_00001_rest             | rest             | 0                             | 0.20201          | 0                    | 0              |
| optimized_00007_task_input_drive | task_input_drive | 0                             | 0.20201          | 0                    | 0              |
| optimized_00029_task_input_drive | task_input_drive | 0                             | 0.20201          | 0                    | 0              |
| optimized_00036_task_input_drive | task_input_drive | 0                             | 0.20201          | 0                    | 0              |

## Next Grammar Changes

Outcome A: actuator appears inert under this grammar. Verify pulse-driven spikes enter STDP and increase positive-control dose before trusting the model.
