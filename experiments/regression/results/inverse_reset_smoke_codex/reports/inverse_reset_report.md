# Learned Inverse Reset Report

## Run Metadata

- Run ID: `inverse_reset_smoke_codex`
- Mode: `full`
- Dataset examples: `24`
- Candidates: `4`

## Candidate Protocols

| protocol_id             | program_family | duration_s | energy_cost | predicted_loss | predicted_task_erasure | predicted_health_penalty | model_uncertainty |
| ----------------------- | -------------- | ---------- | ----------- | -------------- | ---------------------- | ------------------------ | ----------------- |
| optimized_r0_00003_rest | rest           | 0.2        | 0           | 0.0019674      | 0                      | 0                        | 0                 |
| optimized_r0_00009_rest | rest           | 0.12       | 0           | 0.0019674      | 0                      | 0                        | 0                 |
| optimized_r1_00000_rest | rest           | 0.19837    | 0           | 0.0019674      | 0                      | 0                        | 0                 |
| optimized_r1_00001_rest | rest           | 0.13483    | 0           | 0.0019674      | 0                      | 0                        | 0                 |

## Paired Validation

| protocol_id             | program_family | validated_causal_task_erasure | validated_weight_erasure | validated_path_erasure | validated_residual_performance | validated_savings | validated_trace_auc_proxy | validated_health | validated_orthogonal_damage | stimulus_effect_norm | beats_no_reset | passes_health_criterion | passes_generalization_criterion |
| ----------------------- | -------------- | ----------------------------- | ------------------------ | ---------------------- | ------------------------------ | ----------------- | ------------------------- | ---------------- | --------------------------- | -------------------- | -------------- | ----------------------- | ------------------------------- |
| optimized_r0_00003_rest | rest           | 0                             | -0.51818                 | -0.51818               | 0                              | 1.25e-10          | 0.5                       | 0                | 0.037061                    | 0                    | 0              | 0                       | 0                               |
| optimized_r0_00009_rest | rest           | 0                             | -0.51818                 | -0.51818               | 0                              | 1.25e-10          | 0.5                       | 0                | 0.037061                    | 0                    | 0              | 0                       | 0                               |

## Failure Cases

| protocol_id             | program_family | validated_causal_task_erasure | validated_health | stimulus_effect_norm | beats_no_reset |
| ----------------------- | -------------- | ----------------------------- | ---------------- | -------------------- | -------------- |
| optimized_r0_00003_rest | rest           | 0                             | 0                | 0                    | 0              |
| optimized_r0_00009_rest | rest           | 0                             | 0                | 0                    | 0              |

## Next Grammar Changes

Outcome A: actuator appears inert under this grammar. Verify pulse-driven spikes enter STDP and increase positive-control dose before trusting the model.
