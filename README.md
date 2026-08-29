<p align="center">
  <img src="docs/assets/epidive-logo.png" alt="EpiDive logo" width="240">
</p>

# EpiDive

EpiDive is a scalable CPU/GPU workflow for VP and high-order epistasis analysis from SNP matrices. It supports one-background and batch-background analyses, sample reweighting, GWES-based filtering, deterministic sharding across servers, resumable execution, and memory-aware pairwise computation.

> **Release status:** CPU package `1.0.5`; GPU package `1.0.14`. The CPU and GPU components use separate version numbers.

[中文说明](README.zh-CN.md) · [Installation](docs/INSTALLATION.md) · [Usage](docs/USAGE.md) · [Input and output](docs/INPUT_OUTPUT.md) · [HPC/Slurm](docs/HPC.md)

## Highlights

- CPU execution with Intel MKL or OpenBLAS.
- GPU acceleration with CuPy and CUDA 11/12, with a validated CPU fallback for selected stages.
- Independent MAF and minor-allele-count filters (`--maf`, `--min-mac`).
- SCC, HCC, and NW sample-reweighting modes.
- Streamed Parquet output and compact `uint8` allele caches.
- Batch backgrounds with deterministic multi-server sharding.
- Resume markers, cached full-data VP results, error logs, and per-shard manifests.
- Optional hybrid effective-sample-size screening and final-only output.

## Repository layout

```text
cpu/                  CPU package and tests
gpu/                  GPU package and tests
docs/                 Installation, usage, I/O, HPC, and reproducibility guides
release-assets/       Local-only release archives (not committed)
```

## Quick start

Clone the repository and choose one implementation.

### CPU (MKL)

```bash
cd cpu
conda env create -f environment-cpu-mkl.yml
conda activate epidive-cpu
python -m pip install --no-deps .
python tests/smoke_test.py
epidive-vp-ho-cpu --help
```

For OpenBLAS, use `environment-cpu-openblas.yml`. Do not install the MKL and OpenBLAS CPU variants into the same Conda environment.

### GPU (CUDA 12)

```bash
cd gpu
bash install.sh --cuda 12 --blas mkl
conda activate epidive-gpu
epidive-vp-ho-gpu --help
```

### Run one background

```bash
epidive-vp-ho-cpu \
  --input /data/snp_matrix.parquet \
  --output-dir /data/results/background_4938638 \
  --background 4938638 \
  --pair-jobs 32 \
  --prefix vp10k
```

Use `epidive-vp-ho-gpu` with the same core options for GPU execution.

## Input at a glance

The input is a **loci × samples** matrix. The first column is the locus identifier and becomes the row index; remaining columns are samples. Supported input is Parquet or tab-delimited text. Alleles may be `A`, `T`, `C`, `G`, or `-`; numeric matrices are also accepted. The background locus must exist in the row index.

See [Input and output](docs/INPUT_OUTPUT.md) before running production data.

## Reproducible production runs

- Record the component version, full command, Conda environment export, input checksum, and shard parameters.
- Start with a small batch using `--background-limit 2`.
- Keep resume enabled for production; `--no-resume` intentionally bypasses compatible cached results.
- Use a separate output directory per server, then merge results after all shards finish.
- Keep BLAS thread counts aligned with scheduler CPU allocations.

See [Reproducibility](docs/REPRODUCIBILITY.md) for a checklist.

## Documentation

- [Installation and verification](docs/INSTALLATION.md)
- [Command-line usage and examples](docs/USAGE.md)
- [Input and output contract](docs/INPUT_OUTPUT.md)
- [HPC and Slurm deployment](docs/HPC.md)
- [Reproducibility checklist](docs/REPRODUCIBILITY.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Citation and license

EpiDive is released under the [MIT License](LICENSE). Formal citation metadata has not yet been supplied; until it is added, cite the repository URL, release tag, and CPU/GPU component version used in the analysis.
