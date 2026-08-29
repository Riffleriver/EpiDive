# Installation

## Requirements

- Linux x86_64 is recommended for server deployment.
- Conda, Miniforge, or Miniconda.
- Python 3.10 (the packages declare support for Python `>=3.10,<3.12`).
- Access to conda-forge and PyPI during environment creation.
- GPU mode additionally requires an NVIDIA driver compatible with CUDA 11.8 or 12. The installer supplies CuPy and the CUDA runtime; it does not install the system driver.

## CPU from source

Choose exactly one BLAS environment.

```bash
cd cpu

# Intel/AMD server where MKL is desired
conda env create -f environment-cpu-mkl.yml
conda activate epidive-cpu

# Or use OpenBLAS
# conda env create -f environment-cpu-openblas.yml
# conda activate epidive-cpu

python -m pip install --no-deps .
python tests/smoke_test.py
python tests/min_mac_filter_test.py
EPIDIVE_EXPECTED_BLAS=mkl python tests/blas_backend_test.py
epidive-vp-ho-cpu --help
```

For an OpenBLAS environment, set `EPIDIVE_EXPECTED_BLAS=openblas` for the backend test. The CPU MKL and OpenBLAS builds share the distribution name `epidive-vp-ho-cpu`; keep them in separate Conda environments if both are needed.

## GPU from source

Check the driver before installation:

```bash
nvidia-smi
```

Then run:

```bash
cd gpu
bash install.sh --cuda 12 --blas mkl
conda activate epidive-gpu
```

For an older compatible driver, choose CUDA 11:

```bash
bash install.sh --cuda 11 --blas mkl
```

For OpenBLAS:

```bash
bash install.sh --cuda 12 --blas openblas
```

The installer runs import, MAC-filter, BLAS, vectorized-conversion, and reweight-backend checks. Use `--name ENV_NAME` for a custom environment name. Use `--reinstall` only when you intentionally want the installer to remove and recreate an environment with that name.

## Release archives

GitHub Releases provide self-contained server archives containing a wheel, source, environment definition, tests, and a Slurm template. After downloading an archive:

```bash
tar -xzf ARCHIVE.tar.gz
cd EXTRACTED_DIRECTORY
bash install.sh
```

Verify downloaded files against `SHA256SUMS` published with the release.

## Updating an existing environment

Do not update an environment while a scheduler job is using it. After all jobs finish, install the new wheel in place:

```bash
conda run -n ENV_NAME python -m pip install \
  --upgrade --force-reinstall --no-deps ./dist/PACKAGE.whl
```

Then run the included smoke and regression checks. An in-place package update does not remove existing result directories or compatible success markers.
