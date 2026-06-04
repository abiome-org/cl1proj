# CL SDK reset bridge

`cl1_clsdk_bridge` is the thin adapter layer between two packages that are
otherwise kept apart: the standalone reset simulator `cl1_snn_reset` and the
vendored CL SDK surrogate twin under `src/cl/`. It holds all of the SDK-facing
wiring so `cl1_snn_reset` never imports `cl`, and `src/cl/` only needs a small,
stable hook to reach the reset backend.

For the twin backend itself (selection, capability ladder, profiles,
plasticity), see [`docs/twin.md`](../twin.md). This doc covers only the bridge.

## What the bridge connects

The SDK twin (`cl.twin.surrogate.SurrogateTwinModel`) already supports several
internal dynamics engines (surrogate, `population`, dense/sparse `izhikevich`).
The bridge adds one more: the reset SNN. When the SDK is asked for a reset
dynamics mode, `SurrogateTwinModel.__init__` constructs a `ResetSNNAdapter`
instead of an `IzhikevichNetwork`/`SparseIzhikevichNetwork`, and drives it with
the same `apply_timed_stim(...)` / `render(...)` calls it uses for the other SNN
engines.

```
neurons.stim(...)
  -> SurrogateTwinModel.apply_stim   -> ResetSNNAdapter.apply_timed_stim
  -> SurrogateTwinModel.render       -> ResetSNNAdapter.render -> [SNNSpike, ...]
                                              |
                                              v
                                    cl1_snn_reset.build_network(...).advance(...)
```

## Public surface

Exported from `cl1_clsdk_bridge/__init__.py`:

| Name | Source module | Role |
| --- | --- | --- |
| `ResetSNNAdapter` | `reset_adapter.py` | CL producer adapter wrapping the reset network. |
| `culture_config_from_twin` | `twin_mapping.py` | Builds a `cl1_snn_reset.CultureConfig` from a `TwinConfig`. |
| `RESET_DYNAMICS_MODES` | `dynamics.py` | Frozenset of accepted dynamics-mode strings. |
| `is_reset_dynamics` | `dynamics.py` | Predicate: is a mode string a reset mode? |
| `reset_backend_for_mode` | `dynamics.py` | Maps a reset mode string to a backend name. |

## Env-var selection contract

The bridge is reached entirely through the SDK's existing environment-variable
configuration. Two variables matter:

```bash
# Use the twin/surrogate simulator instead of replay.
CL_SDK_SIM_MODE=surrogate

# Route the twin's dynamics layer to the reset SNN backend.
CL_SDK_TWIN_DYNAMICS=snn_reset
```

`CL_SDK_SIM_MODE` is consumed by the SDK to pick the surrogate twin (aliases
`surrogate`, `biological`, `twin`; see `docs/twin.md`). `CL_SDK_TWIN_DYNAMICS`
populates `TwinConfig.dynamics_mode`, which the bridge inspects.

### Accepted dynamics-mode values

`dynamics.py` defines the exact accepted set (matching is case-insensitive,
via `.lower()`):

```python
RESET_DYNAMICS_MODES = frozenset({"snn_reset", "reset_snn", "brian2_reset"})
```

| `CL_SDK_TWIN_DYNAMICS` | `is_reset_dynamics` | `reset_backend_for_mode` |
| --- | --- | --- |
| `snn_reset` | `True` | `numpy` |
| `reset_snn` | `True` | `numpy` |
| `brian2_reset` | `True` | `brian2` |
| anything else (e.g. `izhikevich`, `off`) | `False` | n/a |

`is_reset_dynamics(mode)` is what `SurrogateTwinModel` calls to decide whether
to instantiate `ResetSNNAdapter`. `reset_backend_for_mode(mode)` returns
`"brian2"` only for `brian2_reset`, otherwise `"numpy"`.

## `ResetSNNAdapter`

`ResetSNNAdapter` (in `reset_adapter.py`) is a CL producer adapter that accepts
the same stimulation/render calls as the Izhikevich twin engines but routes
stimulation through the reset simulator's channel-level electrode interface.

Constructed with keyword-only args:

```python
ResetSNNAdapter(
    *,
    channel_count: int,
    neuron_count: int,
    frames_per_second: int,
    rng: np.random.Generator,
    config: TwinConfig,
)
```

Construction details:

- It requires `channel_count == 64` and raises `ValueError` otherwise (it
  targets the CL1-style 64-channel MEA).
- It calls `culture_config_from_twin(config, channel_count=..., neuron_count=...)`
  to derive a `CultureConfig`, then `build_network(culture, seed=int(config.seed))`
  to create the reset network stored on `self.network`.
- It keeps a `self._pending_stims` list of `(timestamp, channel, current)`
  tuples queued between renders.

Public members:

| Member | Signature | Behaviour |
| --- | --- | --- |
| `synapse_count` | property `-> int` | Proxies `self.network.synapse_count`. |
| `apply_timed_stim` | `(timestamp: int, channel: int, drive: np.ndarray) -> None` | Queues a pulse. The drive array is reduced to a scalar current `clip(max(abs(drive)), 0.05, 8.0)` and appended to `_pending_stims`. |
| `apply_stim` | `(channel: int, drive: np.ndarray) -> None` | Compatibility shim for older producers; calls `apply_timed_stim(0, channel, drive)`. |
| `render` | `(from_timestamp, frame_count, *, excitability: np.ndarray, response_gain: np.ndarray) -> list[SNNSpike]` | Advances the reset network and returns SDK spikes. |

`render(...)` is where the frame/stim translation happens:

- It computes `duration_ms = frame_count * 1000 / frames_per_second` and a
  window `[from_timestamp, from_timestamp + frame_count)`.
- Pending stims whose timestamp falls before the window end are converted into
  `cl1_snn_reset.StimEvent` objects (`time_us` from the local frame offset,
  `channels=(channel,)`, `current_uA=current`, `pulse_width_us=160`); later
  stims stay queued.
- It calls `self.network.advance(duration_ms, due, plasticity=True, record=True)`
  and reads back `activity.spike_times_ms` and `activity.channels`.
- Each spike is mapped to a CL frame offset and emitted as a
  `cl.twin.izhikevich.SNNSpike` with `strength` set to
  `response_gain[channel] * excitability[channel]`, clamped to `[0.1, 2.0]`.
  `SNNSpike` is imported lazily inside the method to avoid a hard import of
  `cl` at module load.

## `culture_config_from_twin` (twin_mapping.py)

`twin_mapping.py` maps SDK twin configuration onto the reset simulator's
`CultureConfig`. `culture_config_from_twin(config, *, channel_count, neuron_count)`:

- Lowercases `config.dynamics_mode`. If the mode is a reset mode and equals
  `brian2_reset`, the backend comes from `reset_backend_for_mode(mode)`
  (`"brian2"`); otherwise it uses `config.snn_reset_backend`
  (env `CL_SDK_TWIN_SNN_RESET_BACKEND`, default `"numpy"`).
- Translates `TwinConfig` fields into `CultureConfig` fields:

| `CultureConfig` field | Source |
| --- | --- |
| `n_neurons` | `max(channel_count, neuron_count)` |
| `excitatory_fraction` | `config.snn_excitatory_fraction` |
| `field_size_mm` | constant `3.0` |
| `n_electrodes` | `channel_count` |
| `connection_length_mm` | `max(0.03, config.snn_length_constant_um / 1000)` |
| `long_range_prob` | constant `0.02` |
| `mean_out_degree` | `min(config.snn_max_targets_per_source, 64)` |
| `max_out_degree` | `max(8, config.snn_max_targets_per_source)` |
| `background_noise_mv` | constant `1.0` |
| `spontaneous_rate_hz` | `max(0.0, config.baseline_rate_hz)` |
| `stim_gain_mv_per_uA` | `4.8 * config.snn_coupling` |
| `backend` | reset backend (see above) |

This keeps the SDK env-var vocabulary (`CL_SDK_TWIN_SNN_*`, `CL_SDK_TWIN_BASELINE_RATE_HZ`,
etc.) on the SDK side and converts it once, at the boundary, into reset-library terms.

## dynamics.py

`dynamics.py` is the dynamics-env helper. It is dependency-free (no `cl`, no
`cl1_snn_reset` imports) and holds the canonical mode vocabulary so both the
bridge and `src/cl/twin/surrogate.py` agree on the accepted strings:

- `RESET_DYNAMICS_MODES` — the frozenset above.
- `is_reset_dynamics(mode)` — `mode.lower() in RESET_DYNAMICS_MODES`.
- `reset_backend_for_mode(mode)` — `"brian2"` for `brian2_reset`, else `"numpy"`.

`surrogate.py` imports `RESET_DYNAMICS_MODES`, `is_reset_dynamics`, and
`ResetSNNAdapter` directly from `cl1_clsdk_bridge`, so the SDK never hard-codes
the reset mode names.

## Compatibility re-exports

Older code and the SDK's own twin package reach the adapter through `cl.twin`:

- `cl.twin.ResetSNNAdapter` is re-exported in `src/cl/twin/__init__.py`.
- `src/cl/twin/reset_adapter.py` is a shim whose only job is
  `from cl1_clsdk_bridge import ResetSNNAdapter`; its docstring directs new
  code to import from `cl1_clsdk_bridge` instead.

So `from cl.twin import ResetSNNAdapter` and
`from cl1_clsdk_bridge import ResetSNNAdapter` resolve to the same class.

## Design boundary

The split is deliberate:

- `cl1_snn_reset` stays a standalone reset-simulation library with no knowledge
  of the CL SDK. The bridge depends on it, not the reverse.
- `src/cl/` carries only a minimal hook: `surrogate.py` imports three names from
  `cl1_clsdk_bridge` and the `cl.twin` re-export shims. All translation logic
  (config mapping, frame/stim conversion, mode vocabulary) lives in the bridge.
- Runnable studies do not live here; they sit under `experiments/`. The bridge
  is wiring only.

This is the same boundary the SDK twin uses for its other engines: the producer
asks an adapter for frames and spikes through a fixed contract, and the reset
backend is just another implementation behind it.
