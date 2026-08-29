#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="epidive-gpu"
CUDA_MAJOR="12"
BLAS_BACKEND="mkl"
REINSTALL=0

usage() {
  echo "Usage: bash install.sh [--name ENV_NAME] [--cuda 11|12] [--blas mkl|openblas] [--reinstall]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="${2:?missing environment name}"
      shift 2
      ;;
    --cuda)
      CUDA_MAJOR="${2:?missing CUDA major version}"
      shift 2
      ;;
    --blas)
      BLAS_BACKEND="${2:?missing BLAS backend}"
      shift 2
      ;;
    --reinstall)
      REINSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$CUDA_MAJOR" != "11" && "$CUDA_MAJOR" != "12" ]]; then
  echo "--cuda must be 11 or 12" >&2
  exit 2
fi

if [[ "$BLAS_BACKEND" != "mkl" && "$BLAS_BACKEND" != "openblas" ]]; then
  echo "--blas must be mkl or openblas" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Miniforge first." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL="$SCRIPT_DIR/dist/epidive_vp_ho_gpu-1.0.14-py3-none-any.whl"
INSTALL_TARGET="$SCRIPT_DIR"
[[ -f "$WHEEL" ]] && INSTALL_TARGET="$WHEEL"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$REINSTALL" -eq 1 ]]; then
    echo "Removing existing Conda environment: $ENV_NAME"
    conda env remove -y -n "$ENV_NAME"
  else
    echo "Conda environment '$ENV_NAME' already exists." >&2
    echo "Use --reinstall to remove and recreate it." >&2
    exit 1
  fi
fi

if [[ "$BLAS_BACKEND" == "mkl" ]]; then
  BLAS_PACKAGES=("libblas=*=*mkl" "mkl")
else
  BLAS_PACKAGES=("libblas=*=*openblas" "openblas")
fi

# Install compiled scientific libraries from conda-forge. The explicit BLAS
# selection prevents a GPU environment from silently using a slow generic
# backend for SCC sample reweighting.
conda create -y -n "$ENV_NAME" -c conda-forge \
  python=3.10 \
  pip \
  numpy=1.26 \
  "${BLAS_PACKAGES[@]}" \
  pandas=2.2 \
  "scipy>=1.11,<2" \
  "numba>=0.59,<0.62" \
  joblib \
  threadpoolctl \
  tqdm \
  "pyarrow>=14" \
  "polars>=0.20" \
  "matplotlib>=3.8" \
  "seaborn>=0.13"

if [[ "$CUDA_MAJOR" == "12" ]]; then
  CUDA_VERSION="12"
else
  CUDA_VERSION="11.8"
fi

# Conda supplies a pre-built CuPy and matching CUDA runtime; no compiler or
# system-wide CUDA Toolkit is required, but a compatible NVIDIA driver is.
conda install -y -n "$ENV_NAME" -c conda-forge \
  "cupy>=13,<14" "cuda-version=$CUDA_VERSION"

# All third-party requirements are already present, so install only our wheel.
conda run -n "$ENV_NAME" python -m pip install --no-deps "$INSTALL_TARGET"
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/tests/smoke_test.py"
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/tests/min_mac_filter_test.py"
EPIDIVE_EXPECTED_BLAS="$BLAS_BACKEND" \
  conda run -n "$ENV_NAME" python "$SCRIPT_DIR/tests/blas_backend_test.py"
conda run -n "$ENV_NAME" python \
  "$SCRIPT_DIR/tests/vectorized_conversion_regression_test.py"
conda run -n "$ENV_NAME" python \
  "$SCRIPT_DIR/tests/reweight_backend_test.py"

echo
echo "Installation completed."
echo "Activate with: conda activate $ENV_NAME"
echo "Check options with: epidive-vp-ho-gpu --help"
