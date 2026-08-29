import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import EpiDive_vp_ho_cpu_cli as pipeline
import epidis_gwes_high as high


required = [
    "--input", "input.parquet",
    "--output-dir", "/tmp/epidive-plot-test",
    "--background", "10",
]
assert pipeline._build_cli_parser().parse_args(required).make_plots is True
assert (
    pipeline._build_cli_parser().parse_args(required + ["--no-plot"]).make_plots
    is False
)
assert (
    pipeline._build_cli_parser().parse_args(required + ["--plot"]).make_plots
    is True
)

rng = np.random.default_rng(2026)
n = 3_000
diff = rng.normal(0, 0.18, n)
frame = pd.DataFrame({
    "locus1": np.r_[np.arange(1, 1501), np.arange(3_300_000, 3_301_500)],
    "locus2": np.r_[np.arange(101, 1601), np.arange(3_300_100, 3_301_600)],
    "v_f": np.full(n, 0.5),
    "v_h": 0.5 + diff,
    "geno_dis": np.full(n, 1000),
})

with mock.patch.object(
    high,
    "gwes_high_from_df",
    side_effect=AssertionError("plot function must not be called"),
):
    iqr = high.run_highorder_threshold_analysis_bootstrap(
        frame,
        "/tmp/epidive-plot-test/iqr",
        n_bootstrap=20,
        bootstrap_sample_size=1000,
        separate_chromosomes=True,
        save_filtered=False,
        make_plots=False,
        verbose=False,
    )
    quantile = high.run_highorder_threshold_analysis_bootstrap_quantile(
        frame,
        "/tmp/epidive-plot-test/quantile",
        n_bootstrap=20,
        bootstrap_sample_size=1000,
        min_tail_pairs=20,
        save_filtered=False,
        make_plots=False,
        verbose=False,
    )
    auto = high.run_highorder_threshold_analysis_auto(
        frame,
        "/tmp/epidive-plot-test/auto",
        n_bootstrap=20,
        bootstrap_sample_size=1000,
        min_tail_pairs=20,
        save_filtered=False,
        make_plots=False,
        verbose=False,
    )

for result in (iqr, quantile, auto):
    assert result["plot_chr1"] is None
    assert result["plot_chr2"] is None

print("Plot toggle test: PASS")
