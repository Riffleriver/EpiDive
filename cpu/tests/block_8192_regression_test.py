"""Optional local comparison of 4096 and 8192 EpiDis block sizes."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from epidis_cpu_fast import run_epidis_normal_cpu


rng = np.random.default_rng(8192)
loci = 18_000
samples = 150
matrix = rng.integers(0, 2, size=(loci, samples), dtype=np.uint8)
for row in range(1, loci, 50):
    matrix[row] = matrix[row - 1]
data = pd.DataFrame(
    matrix,
    index=np.arange(1, loci + 1, dtype=np.int64),
)
weights = pd.Series(rng.uniform(0.5, 1.5, samples), index=data.columns)

results = {}
elapsed = {}
for block_size in (4096, 8192):
    start = time.perf_counter()
    results[block_size] = run_epidis_normal_cpu(
        data_df=data,
        weight_ser=weights,
        threshold=0.3,
        n_jobs=8,
        block_size=block_size,
        accumulation_dtype="float32",
        show_progress=False,
    ).sort_values(["locus1", "locus2"]).reset_index(drop=True)
    elapsed[block_size] = time.perf_counter() - start

np.testing.assert_array_equal(
    results[4096][["locus1", "locus2"]].to_numpy(),
    results[8192][["locus1", "locus2"]].to_numpy(),
)
np.testing.assert_allclose(
    results[4096]["v"], results[8192]["v"], rtol=0, atol=0,
)
print(
    f"Block regression: 4096={elapsed[4096]:.3f}s, "
    f"8192={elapsed[8192]:.3f}s, pairs={len(results[4096])}"
)
print("Block 8192 regression test: PASS")
