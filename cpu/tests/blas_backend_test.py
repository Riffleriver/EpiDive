import json
import os

import numpy as np
from threadpoolctl import threadpool_info


# Load the BLAS runtime before querying it.
np.ones((2, 2), dtype=np.float32) @ np.ones((2, 2), dtype=np.float32)
backends = threadpool_info()
names = " ".join(
    str(item.get("internal_api", "")) + " " + str(item.get("prefix", ""))
    for item in backends
).lower()
print("BLAS backends:", backends)
expected = os.environ.get("EPIDIVE_EXPECTED_BLAS", "mkl").lower()
if expected not in names:
    raise RuntimeError(
        f"Expected BLAS backend {expected!r} was not detected. "
        "Recreate the environment with install.sh."
    )

print(f"BLAS backend test ({expected}): PASS")
