import logging
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import epidis_reweight as reweight


logger = logging.getLogger("reweight-backend-test")
logger.addHandler(logging.NullHandler())
logger.propagate = False

rng = np.random.default_rng(20260820)
snp_numeric = rng.integers(0, 5, size=(257, 17), dtype=np.uint8)
expected = np.empty((17, 17), dtype=np.float64)
for left in range(17):
    for right in range(17):
        expected[left, right] = np.count_nonzero(
            snp_numeric[:, left] == snp_numeric[:, right]
        )

cpu = reweight._compute_distance_matrix(
    snp_numeric,
    n_jobs=1,
    block_size=31,
    logger=logger,
    show_progress=False,
    backend="cpu",
)
np.testing.assert_array_equal(cpu, expected)

gpu_available = False
try:
    import cupy as cp

    gpu_available = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    pass

if gpu_available:
    gpu = reweight._compute_distance_matrix(
        snp_numeric,
        logger=logger,
        show_progress=False,
        backend="gpu",
        gpu_id=0,
        gpu_block_size=29,
    )
    np.testing.assert_array_equal(gpu, expected)
    np.testing.assert_array_equal(gpu, cpu)
    print("Reweight backend CPU/GPU equivalence test: PASS")
else:
    automatic = reweight._compute_distance_matrix(
        snp_numeric,
        n_jobs=1,
        logger=logger,
        show_progress=False,
        backend="auto",
    )
    np.testing.assert_array_equal(automatic, expected)
    print("Reweight backend CPU test: PASS; GPU test SKIP (no visible CUDA GPU)")
