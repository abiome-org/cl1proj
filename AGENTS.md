## Learned User Preferences

- Run project Python, pytest, and experiment scripts with `.venv-uv/bin/python` unless another interpreter is specified.
- Use Seaborn (with matplotlib) for scientific plots when generating experiment figures or Pareto visualizations.
- Save local plot and CSV deliverables to the macOS Desktop (`~/Desktop`) when the user asks for files on "desktop" or does not name another output path.
- Run matplotlib-heavy benchmarks with isolated cache dirs: `MPLCONFIGDIR=/private/tmp/mpl-cache` and `XDG_CACHE_HOME=/private/tmp/xdg-cache`.
- Apply strict structural code-quality refactors (centralize duplicated logic, remove dead API fields, avoid spaghetti branching) and deslop on touched files; avoid changelog-style comments.
- Create git commits only when explicitly requested; use Conventional Commits message format.
- Do not add new markdown documentation files unless the user asks; report outcomes in chat instead.
- Never gitignore experiment results, generated reports, or other documentation artifacts; track them or explicitly rewrite them when validity is in question.

## Learned Workspace Facts

- `cl1proj` (`https://github.com/abiome-org/cl1proj.git`) simulates Cortical Labs CL1 train–reset–relearn experiments on 64-channel MEA cultures before wetware runs.
- Packages: `src/cl1_snn_reset/` (reset library), `src/cl1_clsdk_bridge/` (SDK twin wiring), `src/cl/` (vendored CL SDK); shims include `cl.snn_reset` and `cl.twin.ResetSNNAdapter`.
- Python ≥3.12; local virtualenv at `.venv-uv`.
- Large sweep artifacts live under `experiments/snn_reset/results/` (e.g. `full_grid_10k_calibrated_20260602T042050Z` with `summary.csv`, `pareto.csv`, `ranked.csv`).
- The first calibrated SNN reset grid (`full_grid_10k_calibrated_20260602T042050Z`) is invalidated by the discovered protocol/actuator error. Do not cite it as a valid experimental result in papers, docs, or external writeups; mention it only as internal debugging/forensics with explicit invalidated status.
- The 2026-06-06 task assay diagnostics invalidated the old baseline task endpoint: input and sham trials matched, and destructive direct-weight controls did not change that measured score. The replacement `experiments/snn_reset` regime validates `conditioned_electrode_association` and `pattern_discrimination` with baseline 0, trained 1, and naive-weight control 0 across seeds 1/3/4; `evoked_channel_response` is a sensory control, while delayed/order tasks are task-viability checks unless redesigned.
- Reset temporal color is set by `beta` in `noise.py` (−2 violet-like through 2 red/brown); in the invalidated first grid, apparent Pareto candidates were all `beta=2` (red), `epoch_pause`, `independent`, 0.8 µA, differing mainly in duration.
- Surrogate reset runs use `CL_SDK_SIM_MODE=surrogate` with `CL_SDK_TWIN_DYNAMICS` in `snn_reset` / `reset_snn` / `brian2_reset` (canonical helpers in `cl1_clsdk_bridge`).
- Protocol ranking uses multi-objective `pareto_front()` in `cl1_snn_reset.analysis` (weight erasure, health, path erasure, residual performance, savings, trace AUC, criticality distance, energy cost).

## Source vs experiments

- **`src/`** is the installable library only: `cl1_snn_reset`, `cl1_clsdk_bridge`, vendored `cl`. No experiment entrypoints, no result CSVs, no grid CLIs.
- **`experiments/`** holds runnable studies. Each subdirectory has a single `README.md` (scripts + results layout). Do not edit `src/` when adding or running experiments.
- **Imports:** experiments use `import cl1_snn_reset` / `cl1_clsdk_bridge` public APIs (`__init__` exports and documented subpackages such as `cl1_snn_reset.inverse_control`). Do not import private `_` symbols or reach into internal modules when a public export exists.
- **`experiments/regression1/`** — `smoke.py`, `benchmark.py`, `learned_inverse_reset.py`, inverse-reset YAML configs; outputs under `experiments/regression1/results/`; suitable for quick checks and controllability regressions.
- **`experiments/snn_reset/`** — modular SNN reset grid regime: `run_grid.py` launches one script per task (`task_evoked_channel_response.py`, `task_conditioned_electrode_association.py`, `task_delayed_conditioned_response.py`, `task_pattern_discrimination.py`, `task_temporal_order_discrimination.py`); outputs under `experiments/snn_reset/results/`.
- **`tests/`** — fast pytest for library and bridge; may use internal modules for white-box tests.

## Library API (cl1_snn_reset)

Prefer root imports: `run_trial`, `capture_phase`, `build_trial_artifacts`, `ResetProtocol`, `coarse_protocol_grid`, `run_sweep`, `pareto_front`, `rank_protocols`. Plotting: `cl1_snn_reset.analysis.plot_*`.
