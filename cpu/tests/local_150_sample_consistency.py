"""Optional regression test using the user's local 150-sample SNP matrix."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import epidis_filter as ef
import epidis_reweight as er
from EpiDive_vp_ho_cpu import transform_row


DATA_PATH = Path(
    "/Volumes/Rivers_SSD/MAP/EpiDive1550/10k.NR150S.SNP.GENE.snp"
)
OLD_PATH = Path(
    "/Volumes/Rivers_SSD/IPS_phd/EpiDive/EpiDive_normal/EpiDive/"
    "epidis_cpu_fast.py"
)
BACKGROUND = 4938638


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_fast = load_module("epidis_cpu_fast_reference", OLD_PATH)
new_fast = load_module("epidis_cpu_fast_optimized", SRC / "epidis_cpu_fast.py")

raw = pd.read_table(DATA_PATH, index_col=0, nrows=5000)
background_row = None
with DATA_PATH.open("rt") as handle:
    columns = handle.readline().rstrip("\n").split("\t")[1:]
    for line in handle:
        fields = line.rstrip("\n").split("\t")
        if int(fields[0]) == BACKGROUND:
            background_row = pd.Series(fields[1:], index=columns)
            break
if background_row is None:
    raise RuntimeError(f"Background {BACKGROUND} was not found")

background_binary = transform_row(background_row)
datasets = {
    "full": raw,
    "high_1": raw.loc[:, background_binary.to_numpy() == 1],
    "high_0": raw.loc[:, background_binary.to_numpy() == 0],
}

for label, matrix in datasets.items():
    binary = new_fast.convert_snp_to_binary(matrix)
    frequency = binary.mean(axis=1)
    maf_mask = np.minimum(frequency, 1.0 - frequency) >= 0.02
    matrix = matrix.loc[maf_mask]
    binary = binary.loc[maf_mask]
    reweight = er.run_sample_reweight(
        matrix,
        model="SCC",
        scc_threshold=0.98,
        n_jobs=8,
    )
    weights = reweight["snp_wet"]
    representatives, _ = ef.collapse_duplicate_patterns(
        binary,
        weight_ser=weights,
        verbose=False,
    )
    kwargs = dict(
        data_df=representatives,
        weight_ser=weights,
        threshold=0.3,
        n_jobs=8,
        block_size=512,
        accumulation_dtype="float32",
        show_progress=False,
    )
    expected = old_fast.run_epidis_normal_cpu(**kwargs)
    observed = new_fast.run_epidis_normal_cpu(**kwargs)
    keys = ["locus1", "locus2"]
    expected = expected.sort_values(keys).reset_index(drop=True)
    observed = observed.sort_values(keys).reset_index(drop=True)
    np.testing.assert_array_equal(expected[keys], observed[keys])
    np.testing.assert_allclose(expected["v"], observed["v"], rtol=0, atol=1e-7)
    max_error = (
        float(np.max(np.abs(expected["v"] - observed["v"])))
        if len(expected) else 0.0
    )
    print(
        f"{label}: samples={matrix.shape[1]}, representatives={len(representatives)}, "
        f"pairs={len(observed)}, max_abs_error={max_error:.3g}"
    )

print("Local 150-sample consistency test: PASS")
