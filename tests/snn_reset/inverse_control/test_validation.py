import numpy as np

from cl1_snn_reset.inverse_control.validation import bootstrap_candidate_effects


def test_bootstrap_candidate_effects_marks_nonzero_interval():
    import pandas as pd

    df = pd.DataFrame(
        {
            "protocol_id": ["a", "a", "a"],
            "validated_causal_task_erasure": [1.0, 1.2, 0.9],
        }
    )

    result = bootstrap_candidate_effects(df, samples=50, random_seed=1)

    assert result.loc[0, "protocol_id"] == "a"
    assert bool(result.loc[0, "ci_excludes_zero"])
