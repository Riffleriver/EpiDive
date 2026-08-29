import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import epidis_cpu as optimized
import epidis_reweight as reweight
from EpiDive_vp_ho_cpu import transform_row


DATA_PATH = Path(
    "/Volumes/Rivers_SSD/MAP/EpiDive1550/10k.NR150S.SNP.GENE.snp"
)
REFERENCE_PATH = Path(
    "/Volumes/Rivers_SSD/IPS_phd/EpiDive/EpiDive_normal/EpiDive/epidis_cpu.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_module("epidis_cpu_conversion_reference", REFERENCE_PATH)

# Explicitly exercise ties and all standard symbols.
edge_cases = pd.DataFrame(
    [
        ["T", "C", "T", "C"],
        ["C", "T", "T", "C"],
        ["-", "A", "-", "A"],
        ["G", "G", "N", "N"],
    ],
    index=[1, 2, 3, 4],
)
np.testing.assert_array_equal(
    reference.convert_snp_to_binary(edge_cases, n_jobs=1).to_numpy(),
    optimized.convert_snp_to_binary(edge_cases, n_jobs=1).to_numpy(),
)

if DATA_PATH.is_file():
    real_data = pd.read_table(DATA_PATH, index_col=0, nrows=5000)
    start = time.perf_counter()
    expected = reference.convert_snp_to_binary(real_data, n_jobs=1)
    reference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    observed = optimized.convert_snp_to_binary(real_data, n_jobs=1)
    optimized_seconds = time.perf_counter() - start
    np.testing.assert_array_equal(expected.to_numpy(), observed.to_numpy())
    assert expected.index.equals(observed.index)
    assert expected.columns.equals(observed.columns)
    print(
        f"Real conversion: reference={reference_seconds:.3f}s, "
        f"optimized={optimized_seconds:.3f}s"
    )

    encoded = reweight.encode_snp_alleles_uint8(real_data)
    assert encoded.dtypes.nunique() == 1
    assert encoded.dtypes.iloc[0] == np.dtype(np.uint8)
    assert encoded.attrs[reweight.ALLELE_ENCODING_ATTR]
    cached_binary = optimized.convert_snp_to_binary(encoded)
    np.testing.assert_array_equal(expected.to_numpy(), cached_binary.to_numpy())
    np.testing.assert_array_equal(
        reweight._convert_to_numeric(real_data),
        reweight._convert_to_numeric(encoded),
    )
    np.testing.assert_array_equal(
        transform_row(real_data.iloc[0]).to_numpy(),
        transform_row(encoded.iloc[0]).to_numpy(),
    )
    raw_weights = reweight.run_sample_reweight(
        real_data, model="SCC", scc_threshold=0.98, n_jobs=2,
    )["snp_wet"]
    encoded_weights = reweight.run_sample_reweight(
        encoded, model="SCC", scc_threshold=0.98, n_jobs=2,
    )["snp_wet"]
    np.testing.assert_allclose(
        raw_weights.to_numpy(), encoded_weights.to_numpy(), rtol=0, atol=0,
    )

print("Vectorized conversion regression test: PASS")
