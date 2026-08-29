import tempfile
from pathlib import Path
import sys
from unittest import mock

import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import EpiDive_vp_ho_gpu as pipeline


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    background_file = root / "backgrounds.tsv"
    pd.DataFrame({
        "group": range(1, 8),
        "representative_snp": [10, 11, 12, 13, 14, 15, 15],
    }).to_csv(background_file, sep="\t", index=False)

    parser = pipeline._build_cli_parser()
    args = parser.parse_args([
        "--input", "input.parquet",
        "--output-dir", str(root / "output"),
        "--background-file", str(background_file),
        "--shard-count", "2",
        "--shard-index", "1",
        "--no-plot",
    ])
    assert pipeline._load_background_ids(args) == [11, 13, 15]

    signature = pipeline._analysis_signature(args)
    marker = root / "_SUCCESS.json"
    pipeline._write_success_marker(
        str(marker), {"analysis_signature": signature, "status": "complete"}
    )
    assert pipeline._matching_success_marker(str(marker), signature)
    assert not pipeline._matching_success_marker(str(marker), "different")

    full = {
        "final_pairs": pd.DataFrame({
            "locus1": [1], "locus2": [2], "v": [0.5]
        }),
        "snp_bin": pd.DataFrame(
            {"sample1": [1, 0], "sample2": [0, 1]}, index=[1, 2]
        ),
        "weight_ser": pd.Series(
            [0.5, 0.5], index=["sample1", "sample2"], name="weight"
        ),
        "pattern_map": pd.DataFrame({
            "locus": [1, 2], "representative": [1, 2]
        }),
    }
    full_signature = pipeline._full_analysis_signature(args)
    cache_dir = root / "full_cache"
    pipeline._save_full_cache(full, str(cache_dir), full_signature)
    restored = pipeline._load_full_cache(str(cache_dir), full_signature)
    assert restored is not None
    pd.testing.assert_frame_equal(restored["final_pairs"], full["final_pairs"])
    pd.testing.assert_frame_equal(restored["snp_bin"], full["snp_bin"])
    pd.testing.assert_series_equal(
        restored["weight_ser"], full["weight_ser"], check_names=True
    )
    assert pipeline._load_full_cache(str(cache_dir), "different") is None

    matrix = pd.DataFrame(
        {
            "s1": ["A", "A", "C", "G"],
            "s2": ["A", "C", "C", "G"],
            "s3": ["C", "C", "T", "A"],
            "s4": ["C", "A", "T", "A"],
        },
        index=[10, 11, 12, 100],
    )
    run_file = root / "run_backgrounds.tsv"
    pd.DataFrame({"representative_snp": [10, 11, 12]}).to_csv(
        run_file, sep="\t", index=False
    )
    run_output = root / "batch_output"
    calls = {"full": 0, "subgroup": 0, "comparison": 0}

    def fake_vp(snp_m, dir_save, **kwargs):
        is_subgroup = kwargs.get("highorder") is not None
        calls["subgroup" if is_subgroup else "full"] += 1
        loci = snp_m.index
        binary = pd.DataFrame(
            1, index=loci, columns=snp_m.columns, dtype="int8"
        )
        return {
            "skipped": False,
            "final_pairs": pd.DataFrame({
                "locus1": [int(loci[0])],
                "locus2": [int(loci[-1])],
                "v": [0.5],
            }),
            "snp_bin": binary,
            "weight_ser": pd.Series(
                1 / len(snp_m.columns), index=snp_m.columns, name="weight"
            ),
            "pattern_map": pd.DataFrame({
                "locus": loci.astype(int),
                "representative": loci.astype(int),
            }),
        }

    def fake_comparison(**kwargs):
        calls["comparison"] += 1
        return {}

    argv = [
        "--input", "matrix.parquet",
        "--output-dir", str(run_output),
        "--background-file", str(run_file),
        "--no-plot",
    ]
    with (
        mock.patch.object(pipeline.er, "load_snp_matrix", return_value=matrix),
        mock.patch.object(pipeline, "run_vp_pipeline", side_effect=fake_vp),
        mock.patch.object(
            pipeline,
            "run_highorder_comparison_pipeline",
            side_effect=fake_comparison,
        ),
    ):
        assert pipeline.main(argv) == 0
        assert calls == {"full": 1, "subgroup": 6, "comparison": 6}
        (run_output / "background_12" / "_SUCCESS.json").unlink()
        assert pipeline.main(argv) == 0
        assert calls == {"full": 1, "subgroup": 6, "comparison": 6}
        assert pipeline.main(argv) == 0
        assert calls == {"full": 1, "subgroup": 6, "comparison": 6}

    for background in (10, 11, 12):
        directory = run_output / f"background_{background}"
        assert (directory / "_SUCCESS.json").is_file()
        assert (directory / "_ALLELE_1_SUCCESS.json").is_file()
        assert (directory / "_ALLELE_0_SUCCESS.json").is_file()

print("Batch background test: PASS")
