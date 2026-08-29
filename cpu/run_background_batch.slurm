#!/usr/bin/env bash
#SBATCH --job-name=epidive_cpu_bg
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=24:00:00
#SBATCH --output=epidive_cpu_bg_%j.out
#SBATCH --error=epidive_cpu_bg_%j.err

set -euo pipefail

INPUT_MATRIX="/path/to/snp_matrix.parquet"
BACKGROUND_FILE="/path/to/backgrounds.tsv"
OUTPUT_DIR="/path/to/results/background_batch"
CONDA_ENV="epidive-cpu"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun --cpu-bind=cores epidive-vp-ho-cpu \
  --input "${INPUT_MATRIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --background-file "${BACKGROUND_FILE}" \
  --background-column representative_snp \
  --min-mac 5 \
  --pair-jobs "${SLURM_CPUS_PER_TASK}" \
  --no-plot \
  --prefix vp10k_batch
