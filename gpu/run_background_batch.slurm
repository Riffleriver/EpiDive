#!/usr/bin/env bash
#SBATCH --job-name=epidive_bg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-99%4
#SBATCH --output=epidive_bg_%A_%a.out
#SBATCH --error=epidive_bg_%A_%a.err

set -euo pipefail

INPUT_MATRIX="/cluster/home/yangchao/xuwx/EpiDive/data/10k.NR1550S.full.aln.core0.99.maf002.SNP.Gene.parquet"
BACKGROUND_FILE="/cluster/home/yangchao/xuwx/EpiDive/data/10k.NR1550S.nonpaired.last10000.tsv"
OUTPUT_DIR="/cluster/home/yangchao/xuwx/EpiDive/results/highorder_nw/background_batch"
CONDA_ENV="epidive-gpu"

# Initialize Conda in this non-interactive Slurm shell.
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH" >&2
    echo "Load your Conda module or replace this block with its absolute path." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Start: $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --cpu-bind=cores epidive-vp-ho-gpu \
    --input "${INPUT_MATRIX}" \
    --output-dir "${OUTPUT_DIR}" \
    --background-file "${BACKGROUND_FILE}" \
    --background-column representative_snp \
    --min-mac 5 \
    --shard-count 100 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --apply-gwes-filter \
    --ess-mode absolute \
    --final-only \
    --reweight-jobs 20 \
    --gpu-convert-jobs 12 \
    --no-plot \
    --prefix vp10k_batch

echo "Finish: $(date '+%F %T')"
