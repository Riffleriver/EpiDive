import inspect
import os
import sys
import tempfile

import numpy as np
import pandas as pd


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "src"))

import EpiDive_vp_ho_gpu as pipeline


def main():
    assert inspect.signature(pipeline.run_vp_pipeline).parameters["min_mac"].default == 5
    args = pipeline._build_cli_parser().parse_args(
        ["--input", "matrix.parquet", "--output-dir", "out", "--background", "1"]
    )
    assert args.min_mac == 5

    matrix = pd.DataFrame(
        [
            np.ones(10, dtype=np.uint8),
            np.array([0] + [1] * 9, dtype=np.uint8),
            np.array([0] * 4 + [1] * 6, dtype=np.uint8),
            np.array([0] * 5 + [1] * 5, dtype=np.uint8),
        ],
        index=["mac0", "mac1", "mac4", "mac5"],
    )

    original = pipeline.er.run_sample_reweight

    def capture_filtered(data_df, **kwargs):
        assert data_df.index.tolist() == ["mac5"]
        raise RuntimeError("FILTER_CAPTURED")

    pipeline.er.run_sample_reweight = capture_filtered
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            try:
                pipeline.run_vp_pipeline(
                    matrix,
                    output_dir,
                    maf=0.10,
                    min_mac=5,
                    verbose=False,
                )
            except RuntimeError as exc:
                assert str(exc) == "FILTER_CAPTURED"
            else:
                raise AssertionError("The filtering capture hook was not reached")
    finally:
        pipeline.er.run_sample_reweight = original

    print("min_mac_filter_test: OK")


if __name__ == "__main__":
    main()
