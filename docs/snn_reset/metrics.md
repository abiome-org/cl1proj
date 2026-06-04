# Reset metrics and protocol selection

This doc covers the metrics, scoring, and protocol-selection layer of
`cl1_snn_reset`: how a train-reset-relearn trial is reduced to numbers, and how
those numbers are compared and ranked across a sweep. The relevant modules are
`metrics.py`, `trace_probe.py`, `analysis.py`, and `sweep.py`. The simulator
modules that produce the trials (`config`, `network`, `experiment`, `protocols`,
`noise`, `task`) are documented separately.

> Note: `docs/analysis.md` documents the vendored `cl` SDK's `RecordingView`
> analysis toolkit. That is a different thing. This doc is about
> `cl1_snn_reset.analysis` and the reset metric functions.

## From trial to metrics

A trial is run by `run_trial` (in `experiment.py`) and packaged into a
`TrialArtifacts` (`artifacts.py`), which carries the raw inputs every metric
needs:

- `W0`, `Wtrained`, `Wpost`: hidden weight matrices at naive, post-training, and
  post-reset.
- `A0`, `Apost`: `ChannelActivity` at naive and post-reset.
- `initial`, `relearn`: `TrainingResult` objects exposing `trials_to_criterion`.
- `post_behavior`: residual task performance after reset.
- `path0`, `path_trained`, `path_post`: a scalar readout-path strength at the
  three phases.
- `protocol`, `seed`, `total_pulses`, `trace_auc_proxy`.

`compute_trial_metrics(artifacts: TrialArtifacts) -> TrialMetrics` is the single
entry point that calls each scoring function and assembles a flat, frozen
`TrialMetrics` record. `TrialMetrics.to_row()` returns a plain `dict` suitable
for a DataFrame row; this is what `run_sweep` collects.

## Metric vocabulary

All scoring functions are pure and operate on the artifact fields. Signs below
state what the scalar means, not whether it is good for reset (a high
`residual_performance` is a faithful measurement but a *bad* reset outcome; see
the ranking section for how direction is applied).

| Metric | Function / source | Range | Higher means |
| --- | --- | --- | --- |
| Weight erasure | `weight_erasure_score(W0, Wtrained, Wpost)` | up to 1.0, can go negative | more of the trained weight change was undone |
| Residual trace correlation | `residual_trace_correlation(W0, Wtrained, Wpost)` | `[-1, 1]` | post-reset weight delta still aligned with the trained delta |
| Path erasure | `path_erasure_score(path0, path_trained, path_post)` | up to 1.0, can go negative | the trained readout-path change was undone |
| Savings | `savings_score(initial_trials, relearn_trials)` | up to 1.0, can go to -1.0 | relearning was *faster* than initial learning (more retained skill) |
| Health | `health_metrics(...).score` | `[0, 1]` | culture is in a trainable firing regime |
| Criticality distance | `criticality(activity, naive).distance_from_naive` | `>= 0` | post-reset activity statistics drifted farther from naive |
| Trace AUC | `trace_auc_proxy` / `trace_probe_auc` | `[0.5, 1.0]` | residual trace is more detectable (worse erasure) |
| Energy cost | `protocol.total_charge_uC(total_pulses)` | `>= 0` | more stimulation charge delivered |
| Activity features | `activity_features(activity)` | vector | (feature substrate, not a score) |

### Weight erasure

```python
def weight_erasure_score(W0, Wtrained, Wpost) -> float
```

Computes `trained_delta = Wtrained - W0` and `residual_delta = Wpost - W0`, then
returns `1.0 - ||residual_delta|| / (||trained_delta|| + 1e-9)`. A value of
`1.0` means the post-reset weights returned exactly to naive (`Wpost == W0`).
`0.0` means the reset left as much deviation as training created. **Negative**
values mean the reset pushed the weights *farther* from naive than training did,
i.e. it added drift rather than erasing it. This is the headline metric for true
reset.

### Residual trace correlation

```python
def residual_trace_correlation(W0, Wtrained, Wpost) -> float
```

Pearson correlation between `trained_delta` and `residual_delta` (flattened).
Returns `0.0` when either delta is essentially constant (std below `1e-12`).
High positive correlation means whatever weight change survives the reset still
points in the trained direction, even if its magnitude shrank.

### Path erasure

```python
def path_erasure_score(path0, path_trained, path_post) -> float
```

Scalar analog of weight erasure on the readout-path strength:
`1.0 - |path_post - path0| / (|path_trained - path0| + 1e-9)`. `1.0` is a full
return to the naive path; negative means overshoot past naive.

### Savings

```python
def savings_score(initial_trials, relearn_trials) -> float
```

Returns `1.0 - relearn_trials / (initial_trials + 1e-9)`. This is a behavioral
retention measure: high savings means the culture relearned the task quickly,
which implies the original skill was *not* erased. For reset you want savings
low (ideally near 0, where relearning is as slow as learning from scratch).
Edge cases: `initial_trials <= 0` returns `0.0`, or `-1.0` if there were also
relearn trials.

### Health

```python
def health_metrics(activity, *, duration_s, ei_balance=1.0) -> HealthMetrics
```

`HealthMetrics` fields: `firing_rate_hz`, `active_channel_fraction`,
`ei_balance`, `saturated`, `trainable`, `score`. Firing rate is total spikes
over `duration_s * 64` (per-channel mean across 64 electrodes). The culture is
flagged `saturated` if rate `> 80 Hz` or active fraction `> 0.98`, and
`trainable` if `0.01 <= rate <= 80`, active fraction `>= 0.03`, and not
saturated. The composite `score` in `[0, 1]` is the geometric mean of a
log-Gaussian rate term centered at 3 Hz (`exp(-|log((rate+1e-6)/3)|)`) and an
activity term `min(1, active_fraction / 0.35)`, multiplied by `0.25` if
saturated. High score means a usable, non-pathological firing regime. Only the
`score` flows into `TrialMetrics.health`; `firing_rate_hz` and
`active_channel_fraction` are also copied through.

### Criticality

```python
def criticality(activity, naive=None) -> CriticalityMetrics
```

`CriticalityMetrics` fields: `avalanche_alpha`, `branching_ratio`,
`mean_avalanche_size`, `distance_from_naive`. Activity is binned at 10 ms;
per-bin totals are treated as avalanche sizes. `branching_ratio` is the mean
ratio of consecutive avalanche sizes (near 1.0 is the critical regime).
`avalanche_alpha` is a crude log-log slope of the empirical CCDF of avalanche
sizes, intended only for screening, and falls back to `0.0` when there are too
few distinct sizes or the fit fails. `distance_from_naive` is set only when a
`naive` activity is passed: it is the relative L2 distance between the two
`activity_features` vectors, `||a - b|| / (||b|| + 1e-9)`. `compute_trial_metrics`
records `distance_from_naive` as `criticality_distance` and `branching_ratio`
separately. Higher criticality distance means the post-reset network statistics
moved away from the naive baseline.

### Activity features

```python
def activity_features(activity, *, channel_count=64) -> np.ndarray
```

Shared feature substrate used by criticality distance and the trace probes. It
concatenates per-channel firing rates (length `channel_count`) with four summary
scalars: active channel fraction, total rate, mean inter-spike interval, and ISI
coefficient of variation. Not a score on its own.

### Trace AUC

`trace_probe.py` provides two detectability estimates for "can a classifier tell
naive from post-reset activity," both ranging in `[0.5, 1.0]` where `0.5` is
chance (good erasure) and higher means the residual trace is detectable (bad
erasure).

```python
def trace_auc_proxy(naive, post) -> float
```

Single-trial proxy. Computes the standardized feature distance between the two
`activity_features` vectors and squashes it through
`0.5 + 0.5 * (1 - exp(-3 * distance))`. Used per-row so a single trial still
exposes a trace signal.

```python
def trace_probe_auc(naive_activities, post_activities, *, random_state=1) -> float
```

Multi-seed version. With at least two activities per class it trains a
`StandardScaler` + `LogisticRegression` pipeline and returns cross-validated
`roc_auc_score` (stratified k-fold, up to 5 splits). It degrades gracefully: a
single pair per class falls back to `trace_auc_proxy`; missing scikit-learn
falls back to the mean proxy; insufficient class data returns `0.5`. The
per-trial `TrialMetrics.trace_auc_proxy` comes from the proxy form;
`trace_probe_auc` is the cross-seed aggregate intended for multi-replicate
analysis.

### Energy cost

`energy_cost` is `protocol.total_charge_uC(total_pulses)`, total delivered charge
in microcoulombs. It is a burden term, not a scientific outcome; it breaks ties
between protocols that are otherwise equivalent.

## Multi-objective ranking

The selection layer treats the metric vocabulary as competing objectives. The
Pareto front is authoritative; the scalar score is a quick screen.

### Pareto front

```python
def pareto_front(
    df,
    *,
    maximize=("weight_erasure", "health", "path_erasure"),
    minimize=("residual_performance", "savings", "trace_auc_proxy",
              "criticality_distance", "energy_cost"),
) -> pd.DataFrame
```

Returns the nondominated rows. Each objective is oriented by a sign (`+1` for
maximize, `-1` for minimize) and a row is dropped if some other row is
better-or-equal on every oriented objective and strictly better on at least one.
The trade-off, read in reset terms: maximize erasure (weight, path) and health;
minimize residual performance, savings, trace detectability, criticality drift,
and energy. A helper `_trace_metric(df)` selects `trace_auc_proxy` when present,
otherwise `trace_auc`, so the front works on both per-trial frames and
summarized frames. Empty input returns an empty copy.

### Scalar score

```python
def rank_protocols(df) -> pd.DataFrame
```

Adds a `reset_score` column and sorts descending. The weighting:

```text
reset_score = 1.8 * weight_erasure
            + 1.2 * path_erasure
            + 1.0 * health
            - 1.2 * residual_performance
            - 1.0 * savings
            - 0.8 * (trace_auc - 0.5)
            - 0.4 * criticality_distance
            - 0.05 * energy_cost
```

Trace AUC enters centered at chance (`trace_auc - 0.5`), so a perfectly
undetectable trace contributes zero. Energy has the smallest weight, acting as a
tiebreaker. The docstring is explicit that this is a screen and the Pareto front
remains authoritative.

### Plot helpers

- `plot_protocol_scatter(df, *, x="weight_erasure", y="residual_performance", hue="beta")`
  — quick seaborn scatter of any two metrics, styled by `schedule`. Returns
  `(fig, ax)`.
- `plot_pareto_summary(df)` — overlays the `pareto_front(df)` rows on the full
  screened set in weight-erasure vs trace-detectability space, with health as
  hue and energy cost as marker size, and a chance line at `0.5`. Returns
  `(fig, ax)`.

Both import matplotlib/seaborn lazily so the metric layer has no hard plotting
dependency.

## Feeding the selection layer from sweeps

```python
def run_sweep(cfg, protocols, *, seeds, workers=1) -> pd.DataFrame
```

Runs the full protocol x seed grid. Each job replaces the config seed, calls
`run_trial(...).to_row()`, and the rows become a DataFrame. With `workers > 1`
it fans out over a `ThreadPoolExecutor`. Provenance is stashed in
`df.attrs`: `elapsed_s`, `workers`, and `jobs`.

```python
def summarize_sweep(df) -> pd.DataFrame
```

Aggregates replicate rows by `protocol_id`: protocol descriptor fields
(`beta`, `schedule`, `spatial_mode`, `duration_s`, `current_uA`,
`pulse_width_us`, `total_pulses`) are taken with `first`; the eight metric
columns (`weight_erasure`, `residual_performance`, `savings`,
`trace_auc_proxy`, `criticality_distance`, `health`, `energy_cost`,
`path_erasure`) are averaged; and the `seed` count becomes a `replicates`
column.

Typical pipeline:

```python
raw = run_sweep(cfg, protocols, seeds=[1, 3, 4], workers=4)
summary = summarize_sweep(raw)
front = pareto_front(summary)
ranked = rank_protocols(summary)
```

`pareto_front` and `rank_protocols` accept either the raw per-trial frame or the
per-protocol summary; running them on the summary keeps one row per protocol.

## Headline calibrated-grid finding

The calibrated 10k-neuron full grid
(`docs/snn_reset/full_grid_10k_calibrated_20260602.md`) is a negative result for
true reset, and it is the canonical example of why the sign of each metric
matters. The top-ranked and all Pareto-front protocols were low-burden,
`beta=2` (red), `epoch_pause`, `independent`, 0.8 µA variants differing mainly in
duration; ordering among the top rows was set mostly by energy cost. But every
protocol had **negative weight erasure** (top rows near `-0.44`), meaning
post-reset hidden weights moved farther from the naive state instead of back
toward it. Chance-level behavior and a near-random trace probe were not enough to
call this erasure. The useful outcome is the falsification boundary: with this
fixed-sign STDP model, task, and electrode-only actuator space, colored
stimulation did not restore a naive weight landscape.

## Public exports

From `cl1_snn_reset.__init__`: `TrialMetrics`, `compute_trial_metrics`,
`savings_score`, `weight_erasure_score`, `pareto_front`, `rank_protocols`,
`run_sweep`, `summarize_sweep`, `trace_auc_proxy`, `trace_probe_auc`. The
remaining scoring functions (`path_erasure_score`, `residual_trace_correlation`,
`health_metrics`, `criticality`, `activity_features`) and the plot helpers are
reachable through their modules (`cl1_snn_reset.metrics`,
`cl1_snn_reset.analysis`).
