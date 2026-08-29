import importlib
from pathlib import Path


MODULES = [
    "epidis_cpu",
    "epidis_cpu_test",
    "epidis_cpu_fast",
    "epidis_reweight",
    "epidis_gwes",
    "epidis_gwes_high",
    "epidis_gwes_vp",
    "epidis_filter",
    "epidis_gpu",
    "EpiDive_vp_ho_gpu",
]


for module_name in MODULES:
    importlib.import_module(module_name)

from epidis_cpu_fast import estimate_adaptive_epidis_threshold
from epidis_filter import collapse_duplicate_patterns, restore_pattern_pairs
from EpiDive_vp_ho_gpu import main, run_highorder_comparison_pipeline, run_vp_pipeline
from epidive_data import reference_paths

assert callable(estimate_adaptive_epidis_threshold)
assert callable(collapse_duplicate_patterns)
assert callable(restore_pattern_pairs)
assert callable(run_vp_pipeline)
assert callable(run_highorder_comparison_pipeline)
assert callable(main)
for reference_path in reference_paths().values():
    assert Path(reference_path).is_file()
print("EpiDive GPU package smoke test: PASS")
