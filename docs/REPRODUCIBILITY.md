# Reproducibility checklist

Record the following for every production analysis:

- EpiDive CPU or GPU component version and Git commit.
- Complete command line, including default overrides.
- `conda env export --from-history` and `python -m pip freeze` output.
- Operating system, CPU model, BLAS backend, and thread counts.
- For GPU runs: GPU model, NVIDIA driver, CUDA runtime, and CuPy version.
- Input matrix checksum and dimensions.
- Background-list checksum, selected column, shard count, and shard index.
- Randomness-related parameters such as bootstrap count and sample size.
- Output-directory checksum or immutable archive identifier.

Suggested checksum command:

```bash
sha256sum INPUT_FILE BACKGROUND_FILE
```

On macOS, use `shasum -a 256`.

Before archiving, check that every intended background has a compatible `_SUCCESS.json`, and review all shard-level error files. Do not treat a scheduler exit code alone as proof that every background completed.
