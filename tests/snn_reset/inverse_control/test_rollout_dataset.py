import numpy as np

from cl1_snn_reset import CultureConfig, ExperimentConfig, TaskConfig
from cl1_snn_reset.inverse_control import (
    CausalDeltaDatasetBuilder,
    HybridStateProjector,
    StimConstraints,
)


def test_dataset_builder_uses_no_reset_relative_delta():
    cfg = ExperimentConfig(
        culture=CultureConfig(
            n_neurons=96,
            mean_out_degree=8,
            max_out_degree=12,
            local_candidate_multiplier=2,
            build_workers=1,
            spontaneous_rate_hz=0.0,
            backend="numpy",
        ),
        task=TaskConfig(
            input_channels=(8,),
            target_channels=(9,),
            criterion_response_probability=0.5,
            max_trials=4,
            eval_interval_trials=2,
            eval_trials=2,
            inter_trial_ms=30.0,
        ),
        readout_window_s=0.05,
        warmup_s=0.0,
    )
    projector = HybridStateProjector(cfg.task, weight_projection_dim=4, weight_hist_bins=4)
    builder = CausalDeltaDatasetBuilder(
        projector=projector,
        experiment_config=cfg,
        constraints=StimConstraints(max_energy_cost=1.0, max_total_duration_s=0.2),
        seeds=(1,),
        programs_per_trained_state=4,
        stim_sampling={
            "include_blocks": ["anti_stdp_pairing", "rest_control"],
            "duration_s": [0.08, 0.12],
            "amplitude_uA": [0.8],
            "delays_ms": [2, 5],
        },
        random_seed=9,
    )

    dataset = builder.build()
    example = dataset.examples[0]

    assert dataset.trained_states.shape[0] >= 1
    assert np.allclose(example.causal_delta, example.stimmed_state - example.no_reset_state)
    assert example.metadata["protocol_id"]
