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
    "EpiDive_vp_ho_cpu",
    "EpiDive_vp_ho_cpu_cli",
]


for module_name in MODULES:
    importlib.import_module(module_name)

from epidis_cpu_fast import estimate_adaptive_epidis_threshold
from epidis_filter import collapse_duplicate_patterns, restore_pattern_pairs
from EpiDive_vp_ho_cpu import run_highorder_comparison_pipeline, run_vp_pipeline
from EpiDive_vp_ho_cpu_cli import main
from epidive_data import reference_paths

assert callable(estimate_adaptive_epidis_threshold)
assert callable(collapse_duplicate_patterns)
assert callable(restore_pattern_pairs)
assert callable(run_vp_pipeline)
assert callable(run_highorder_comparison_pipeline)
assert callable(main)
for reference_path in reference_paths().values():
    assert Path(reference_path).is_file()
print("EpiDive CPU package smoke test: PASS")
