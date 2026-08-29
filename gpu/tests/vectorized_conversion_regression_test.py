import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import epidis_cpu as optimized
import epidis_reweight as reweight
from EpiDive_vp_ho_gpu import transform_row


def legacy_convert(frame):
    rows = []
    for _, row in frame.iterrows():
        major = row.value_counts().idxmax()
        rows.append((row == major).astype(np.float16).to_numpy())
    return pd.DataFrame(rows, index=frame.index, columns=frame.columns)


matrix = pd.DataFrame(
    [
        ["T", "C", "T", "C"],
        ["C", "T", "T", "C"],
        ["-", "A", "-", "A"],
        ["G", "G", "A", "A"],
        ["A", "A", "A", "T"],
        ["C", "G", "T", "-"],
    ],
    index=[1, 2, 3, 4, 5, 6],
    columns=["s1", "s2", "s3", "s4"],
)

expected = legacy_convert(matrix)
direct = optimized.convert_snp_to_binary(matrix, n_jobs=1)
encoded = reweight.encode_snp_alleles_uint8(matrix)
cached = optimized.convert_snp_to_binary(encoded, n_jobs=1)

np.testing.assert_array_equal(expected.to_numpy(), direct.to_numpy())
np.testing.assert_array_equal(expected.to_numpy(), cached.to_numpy())
np.testing.assert_array_equal(
    reweight._convert_to_numeric(matrix),
    reweight._convert_to_numeric(encoded),
)
np.testing.assert_array_equal(
    transform_row(matrix.iloc[0]).to_numpy(),
    transform_row(encoded.iloc[0]).to_numpy(),
)

raw_weights = reweight.run_sample_reweight(
    matrix, model="SCC", scc_threshold=0.5, n_jobs=1,
)["snp_wet"]
encoded_weights = reweight.run_sample_reweight(
    encoded, model="SCC", scc_threshold=0.5, n_jobs=1,
)["snp_wet"]
np.testing.assert_allclose(
    raw_weights.to_numpy(), encoded_weights.to_numpy(), rtol=0, atol=0,
)

assert encoded.dtypes.nunique() == 1
assert encoded.dtypes.iloc[0] == np.dtype(np.uint8)
assert encoded.attrs[reweight.ALLELE_ENCODING_ATTR]

print("Vectorized conversion regression test: PASS")
