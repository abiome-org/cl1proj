# SNN reset results — provenance and validity

Result directories are kept, not regenerated, so this index records which ones
are citable and which are forensic-only. Status reflects the 2026-06-06 task-assay
work that invalidated the original behavioral endpoint and replaced it with a
weight-sensitive, control-passing readout.

| Directory | Date | Status | What it is |
|-----------|------|--------|------------|
| `snn_reset_relearning_exhaustive_20260606T070543Z` | 2026-06-06 | **Citable (latest)** | Validated 60-protocol grid with the relearning/savings round on the two learned tasks |
| `snn_reset_validated_grid_20260606T043336Z` | 2026-06-06 | **Citable** | Validated grid: baseline 0 → trained 1, naive-weight control 0 on `conditioned_electrode_association` and `pattern_discrimination` |
| `task_assay_diagnostics_20260606T010902Z` | 2026-06-06 | **Citable (forensic)** | Endpoint diagnostics: input and sham trials match and the old score is invariant to total weight destruction — the evidence that invalidated the original endpoint |
| `grid_diagnostics_20260605T011018Z` | 2026-06-05 | Forensic | Orthogonality and paired-comparison diagnostics run against the first calibrated grid |
| `figures_20260604` | 2026-06-04 | Superseded | Figures derived from the pre-validation grid; kept for history |
| `control_checks_10k_20260604` | 2026-06-04 | Superseded | Control checks on the pre-validation regime |
| `full_grid_10k_calibrated_20260602T042050Z` | 2026-06-02 | **Invalidated** | First calibrated 10k grid. Do not cite: the protocol/actuator error and an endpoint that could not separate input from sham make its rankings meaningless |
| `control_checks_10k_20260602` | 2026-06-02 | Superseded | Control checks on the invalidated first grid |

Each directory carries its own `metadata.json` with the exact `argv`, git commit,
and seeds used to produce it. Regenerate figures for a citable suite with
`figures.py --suite-dir <dir>` and the relearning analysis with
`relearning_analysis.py --suite-dir <dir>`.
