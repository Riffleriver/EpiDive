import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from epidis_filter import restore_pattern_pairs
from epidis_gwes_vp import merge_pairs_with_fill


pattern_map = pd.DataFrame({
    "locus": [10, 11, 20, 21, 22, 30],
    "row_position": [0, 1, 2, 3, 4, 5],
    "pattern_group": [0, 0, 1, 1, 1, 2],
    "representative": [10, 10, 20, 20, 20, 30],
    "pattern_size": [2, 2, 3, 3, 3, 1],
    "within_pattern_v": [0.5, 0.5, 0.6, 0.6, 0.6, 0.0],
})
pairs = pd.DataFrame({
    "locus1": [10, 20],
    "locus2": [20, 30],
    "v": np.array([0.4, 0.7], dtype=np.float32),
    "v_f": np.array([0.4, 0.7], dtype=np.float32),
    "v_h": np.array([0.6, 0.8], dtype=np.float32),
})

unchunked = restore_pattern_pairs(
    pairs,
    pattern_map,
    include_within_pattern=False,
    input_stage="post_gwes",
    chunk_output_rows=1_000_000,
    verbose=False,
)
chunked = restore_pattern_pairs(
    pairs,
    pattern_map,
    include_within_pattern=False,
    input_stage="post_gwes",
    chunk_output_rows=2,
    verbose=False,
)
sort_columns = ["locus1", "locus2"]
pd.testing.assert_frame_equal(
    unchunked.sort_values(sort_columns).reset_index(drop=True),
    chunked.sort_values(sort_columns).reset_index(drop=True),
)
assert len(chunked) == 9

mixed_left = pd.DataFrame({
    "locus1": ["20"], "locus2": ["10"], "v": [0.4]
})
mixed_right = pd.DataFrame({
    "locus1": [10], "locus2": [20], "v": [0.6]
})
mixed_merged = merge_pairs_with_fill(
    mixed_left, mixed_right, fill_value=-1, return_pandas=True
)
assert len(mixed_merged) == 1
assert int(mixed_merged.loc[0, "locus1"]) == 10
assert int(mixed_merged.loc[0, "locus2"]) == 20
assert float(mixed_merged.loc[0, "v1"]) == 0.4
assert float(mixed_merged.loc[0, "v2"]) == 0.6


def old_formula(joint, marginal, safe_dp, safe_dq, size_p, size_q):
    eps = np.float32(1e-27)
    prob_eps = np.float32(1e-7)
    p1 = np.clip(joint / safe_dp, prob_eps, 1 - prob_eps)
    q1 = np.clip((marginal - joint) / safe_dq, prob_eps, 1 - prob_eps)
    p0 = 1 - p1
    q0 = 1 - q1
    st10, st11 = p1 + eps, p0 + eps
    st20, st21 = q1 + eps, q0 + eps
    mix0 = st10 * size_p + st20 * size_q
    mix1 = st11 * size_p + st21 * size_q
    js = (
        (st10 * np.log2(np.maximum(st10 / mix0, eps))
         + st11 * np.log2(np.maximum(st11 / mix1, eps))) * size_p
        + (st20 * np.log2(np.maximum(st20 / mix0, eps))
           + st21 * np.log2(np.maximum(st21 / mix1, eps))) * size_q
    )
    return np.sqrt(np.maximum(js, 0))


rng = np.random.default_rng(2026)
joint = rng.uniform(0.01, 0.7, size=(31, 17)).astype(np.float32)
marginal = rng.uniform(0.75, 1.0, size=(31, 1)).astype(np.float32)
safe_dp = rng.uniform(0.75, 1.0, size=(1, 17)).astype(np.float32)
safe_dq = rng.uniform(0.75, 1.0, size=(1, 17)).astype(np.float32)
size_p = rng.uniform(0.1, 0.9, size=(1, 17)).astype(np.float32)
size_q = 1 - size_p
expected = old_formula(joint, marginal, safe_dp, safe_dq, size_p, size_q)
fused_model = np.empty_like(expected)
for j in range(joint.shape[0]):
    for i in range(joint.shape[1]):
        p1 = min(max(float(joint[j, i] / safe_dp[0, i]), 1e-7), 1 - 1e-7)
        q1 = min(
            max(float((marginal[j, 0] - joint[j, i]) / safe_dq[0, i]), 1e-7),
            1 - 1e-7,
        )
        p0, q0 = 1 - p1, 1 - q1
        sp, sq = float(size_p[0, i]), float(size_q[0, i])
        mix0 = p1 * sp + q1 * sq
        mix1 = p0 * sp + q0 * sq
        js = (
            (p1 * math.log2(p1 / mix0) + p0 * math.log2(p0 / mix1)) * sp
            + (q1 * math.log2(q1 / mix0) + q0 * math.log2(q0 / mix1)) * sq
        )
        fused_model[j, i] = math.sqrt(max(js, 0.0))
np.testing.assert_allclose(expected, fused_model, rtol=2e-5, atol=1e-5)

print("Optimization regression test: PASS")
