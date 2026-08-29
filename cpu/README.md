# EpiDive VP/High-order CPU server package 1.0.5

Version 1.0.5 adds an independent per-locus minor allele count filter. The
default is `--min-mac 5`; a locus must pass both MAF and MAC filters. The MAC
value is included in cache/resume signatures, so incompatible old results are
not reused. Set `--min-mac 0` only when legacy MAF-only behavior is required.

Version 1.0.4 adds absolute/hybrid ESS screening and a final-only batch mode.
It retains the 1.0.3 in-place score buffer, threshold-aware square-root
elimination, uint8 allele cache, and vectorized conversion optimizations.

Version 1.0.3 reuses the joint-count block as the score buffer, avoids square
roots for pairs that clearly fail the threshold, and installs an Intel MKL
BLAS backend for the target x86_64 server environment. Subgroup matrices are
still independently converted, MAF-filtered, reweighted, and pattern-collapsed.
Standard one-character allele matrices use a vectorized converter that keeps
the original per-row major-allele and tie-breaking semantics without copying
large DataFrame chunks into separate worker processes.
The CLI also caches the original categorical allele identities once as a
compact uint8 matrix. Full and subgroup runs still independently determine
their major alleles, MAF filters, sample weights, and duplicate patterns.

This package provides the result-compatible multi-CPU implementation of the
VP/high-order EpiDive workflow. It includes all local Python modules and the
three GWES reference resources, so it does not depend on workstation paths.

## Main improvements

- Multi-CPU BLAS and Numba representative-pair calculation.
- In-place block scoring with threshold-aware square-root elimination.
- Intel MKL BLAS selected by the supplied server environment.
- Streamed Parquet output to reduce peak pairwise memory.
- Polars-backed full pair unions with integer-normalized locus keys.
- Chunked duplicate-pattern restoration.
- Optional plot suppression with unchanged filtering results.
- One-time full-data VP cache reused by every background.
- TSV/CSV background batches, deterministic server sharding, and resume.
- Atomic success markers, error traces, and per-shard manifests.

The statistical workflow, default thresholds, MAF filtering, SCC/HCC/NW
weights, GWES filtering, high-order thresholds, and final restoration are the
same as the updated CPU workstation pipeline.

## Installation

```bash
tar -xzf EpiDive_vp_ho_cpu_server_package-1.0.5.tar.gz
cd EpiDive_vp_ho_cpu_server_package-1.0.5
bash install.sh
conda activate epidive-cpu
epidive-vp-ho-cpu --help
```

## One background

```bash
epidive-vp-ho-cpu \
  --input /data/snp_matrix.parquet \
  --output-dir /data/results/background_4938638 \
  --background 4938638 \
  --pair-jobs 32 \
  --prefix vp10k
```

## Batch backgrounds

```bash
epidive-vp-ho-cpu \
  --input /data/snp_matrix.parquet \
  --output-dir /data/results/background_batch \
  --background-file /data/backgrounds.tsv \
  --background-column representative_snp \
  --pair-jobs 32 \
  --no-plot \
  --prefix vp10k_batch
```

The full-data VP result is calculated once and cached under `_full/`. Reusing
the same command resumes only incomplete backgrounds. Use `--no-resume` to
force recomputation.

For multiple servers, use the same input list and `--shard-count`, with a
different zero-based `--shard-index` on each server.

```bash
epidive-vp-ho-cpu ... --shard-count 8 --shard-index 0
epidive-vp-ho-cpu ... --shard-count 8 --shard-index 1
```

## Important defaults

- Full-data initial threshold: `0.3`.
- Locus filtering requires `MAF >= 0.02` and `MAC >= 5` by default.
- Subgroup initial threshold: `auto`.
- Full-data `k=1`, allele-1 subgroup `k=1`, allele-0 subgroup `k=2`.
- GWES filtering of VP output is disabled unless `--apply-gwes-filter` is set.
- Final pattern restoration and plotting are enabled by default.
- Accumulation uses `float32`; use `--accumulation-dtype float64` for the
  legacy-compatible slower mode.
- CPU block size defaults to the server-validated `4096`. Block size `8192`
  is result-equivalent in regression tests and can be benchmarked on nodes
  with at least 6 GiB available for pairwise temporary matrices.

The supplied `run_background_batch.slurm` is a CPU-node template. Update its
partition and paths before submission.

For hybrid ESS screening and minimal batch storage, add:

```bash
--ess-mode hybrid --formal-min-fraction 0.05 \
  --exploratory-min-fraction 0.02 --final-only
```
