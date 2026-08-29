import os
import tempfile

import EpiDive_vp_ho_cpu as pipeline
import EpiDive_vp_ho_cpu_cli as cli


hybrid = pipeline.classify_subgroup(
    30, original_samples=1550, ess_mode="hybrid"
)
assert hybrid["formal_min_ess"] == 78
assert hybrid["exploratory_min_ess"] == 31
assert hybrid["analysis_level"] == "skip"

absolute = pipeline.classify_subgroup(30, ess_mode="absolute")
assert absolute["formal_min_ess"] == 40
assert absolute["exploratory_min_ess"] == 25
assert absolute["analysis_level"] == "exploratory"

with tempfile.TemporaryDirectory() as directory:
    final_path = os.path.join(directory, "final.parquet")
    intermediate_path = os.path.join(directory, "intermediate.parquet")
    open(final_path, "wb").close()
    open(intermediate_path, "wb").close()
    kept, removed = cli._cleanup_background_intermediates(
        {"paths": {"pattern_map": intermediate_path}},
        {
            "paths": {
                "filtered": final_path,
                "representative_filtered": intermediate_path,
            }
        },
    )
    assert kept == final_path
    assert os.path.isfile(final_path)
    assert intermediate_path in removed
    assert not os.path.exists(intermediate_path)

print("Hybrid ESS and final-only test: PASS")
