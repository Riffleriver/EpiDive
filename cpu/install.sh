#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="epidive-cpu"
BLAS_BACKEND="mkl"
REINSTALL=0

usage() { echo "Usage: bash install.sh [--name ENV_NAME] [--blas mkl|openblas] [--reinstall]"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) ENV_NAME="${2:?missing environment name}"; shift 2 ;;
    --blas) BLAS_BACKEND="${2:?missing BLAS backend}"; shift 2 ;;
    --reinstall) REINSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$BLAS_BACKEND" != "mkl" && "$BLAS_BACKEND" != "openblas" ]]; then
  echo "--blas must be mkl or openblas" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Miniforge first." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL="$SCRIPT_DIR/dist/epidive_vp_ho_cpu-1.0.5-py3-none-any.whl"
INSTALL_TARGET="$SCRIPT_DIR"
[[ -f "$WHEEL" ]] && INSTALL_TARGET="$WHEEL"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$REINSTALL" -eq 1 ]]; then
    conda env remove -y -n "$ENV_NAME"
  else
    echo "Conda environment '$ENV_NAME' already exists; use --reinstall." >&2
    exit 1
  fi
fi

if [[ "$BLAS_BACKEND" == "mkl" ]]; then
  BLAS_PACKAGES=("libblas=*=*mkl" "mkl")
else
  BLAS_PACKAGES=("libblas=*=*openblas" "openblas")
fi

conda create -y -n "$ENV_NAME" -c conda-forge \
  python=3.10 pip numpy=1.26 "${BLAS_PACKAGES[@]}" pandas=2.2 \
  "scipy>=1.11,<2" "numba>=0.59,<0.62" joblib threadpoolctl tqdm \
  "pyarrow>=14" "polars>=0.20" "matplotlib>=3.8" "seaborn>=0.13"

conda run -n "$ENV_NAME" python -m pip install --no-deps "$INSTALL_TARGET"
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/tests/smoke_test.py"
conda run -n "$ENV_NAME" python "$SCRIPT_DIR/tests/min_mac_filter_test.py"
EPIDIVE_EXPECTED_BLAS="$BLAS_BACKEND" \
  conda run -n "$ENV_NAME" python "$SCRIPT_DIR/tests/blas_backend_test.py"

echo "Installation completed."
echo "Activate with: conda activate $ENV_NAME"
echo "Check options with: epidive-vp-ho-cpu --help"
