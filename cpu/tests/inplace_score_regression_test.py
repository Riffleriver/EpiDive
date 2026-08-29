import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from epidis_cpu_fast import run_epidis_normal_cpu


def reference_scores(data, weights, threshold):
    x = np.ascontiguousarray(data.to_numpy(dtype=np.float32))
    w = np.ascontiguousarray(weights.to_numpy(dtype=np.float32))
    weighted_x = np.ascontiguousarray(x * w[None, :])
    dp_all = weighted_x.sum(axis=1, dtype=np.float64)
    dq_all = ((1.0 - x) * w[None, :]).sum(axis=1, dtype=np.float64)
    rows = []
    for i in range(len(data) - 1):
        dp = dp_all[i]
        dq = dq_all[i]
        size_p = dp / (dp + dq)
        size_q = 1.0 - size_p
        for j in range(i + 1, len(data)):
            data_p = float(x[i] @ weighted_x[j])
            data_q = dp_all[j] - data_p
            p1 = max(min(data_p / dp, 1.0 - 1e-9), 1e-9)
            p0 = 1.0 - p1
            q1 = max(min(data_q / dq, 1.0 - 1e-9), 1e-9)
            q0 = 1.0 - q1
            st1_0, st1_1 = p1 + 1e-27, p0 + 1e-27
            st2_0, st2_1 = q1 + 1e-27, q0 + 1e-27
            st3_0 = st1_0 * size_p + st2_0 * size_q
            st3_1 = st1_1 * size_p + st2_1 * size_q
            js = (
                (st1_0 * np.log2(st1_0 / st3_0)
                 + st1_1 * np.log2(st1_1 / st3_1)) * size_p
                + (st2_0 * np.log2(st2_0 / st3_0)
                   + st2_1 * np.log2(st2_1 / st3_1)) * size_q
            )
            score = np.sqrt(js) if js > 0.0 else 0.0
            if score >= threshold:
                rows.append((int(data.index[i]), int(data.index[j]), score))
    return pd.DataFrame(rows, columns=["locus1", "locus2", "v"])


rng = np.random.default_rng(20260819)
x = rng.integers(0, 2, size=(97, 31), dtype=np.uint8)
data = pd.DataFrame(x, index=np.arange(1000, 1097))
weights = pd.Series(rng.uniform(0.2, 2.0, 31), index=data.columns)

for threshold in (0.0, 0.3, 0.7):
    expected = reference_scores(data, weights, threshold)
    observed = run_epidis_normal_cpu(
        data,
        weights,
        threshold=threshold,
        n_jobs=2,
        block_size=32,
        accumulation_dtype="float32",
        show_progress=False,
    )
    expected = expected.sort_values(["locus1", "locus2"]).reset_index(drop=True)
    observed = observed.sort_values(["locus1", "locus2"]).reset_index(drop=True)
    np.testing.assert_array_equal(
        expected[["locus1", "locus2"]].to_numpy(),
        observed[["locus1", "locus2"]].to_numpy(),
    )
    np.testing.assert_allclose(expected["v"], observed["v"], rtol=2e-5, atol=2e-6)

print("In-place score regression test: PASS")
