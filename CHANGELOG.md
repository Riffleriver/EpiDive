# Changelog

This repository contains components with independent version numbers.

## GPU 1.0.14 — 2026-08-21

- Added independent per-locus minor allele count filtering with default `--min-mac 5`.
- Included MAC in cache and resume signatures.
- Retained memory-safe tiled CuPy SGEMM sample-similarity calculation and CPU fallback from 1.0.13.
- Retained vectorized categorical conversion and explicit BLAS validation from 1.0.12.

## CPU 1.0.5 — 2026-08-21

- Added independent per-locus minor allele count filtering with default `--min-mac 5`.
- Included MAC in cache and resume signatures.
- Provided separate MKL and OpenBLAS environment variants.
- Retained absolute/hybrid ESS screening, final-only mode, streamed Parquet output, batch sharding, cache, and resume.

Earlier component history is documented in the corresponding release archives.
