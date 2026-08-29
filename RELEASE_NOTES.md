# Initial GitHub release

This release publishes the source and self-contained server archives for the current EpiDive VP/high-order workflow.

## Components

- **GPU 1.0.14** — CUDA 11/12 installation, memory-aware CuPy computation, CPU fallback for supported stages, independent MAF/MAC filtering, batch execution, sharding, caching, and resume.
- **CPU 1.0.5 (MKL)** — multi-CPU implementation with Intel MKL.
- **CPU 1.0.5 (OpenBLAS)** — the same CPU analysis with an OpenBLAS environment.

## Release assets

- `EpiDive_vp_ho_gpu_server_package-1.0.14.tar.gz`
- `EpiDive_vp_ho_cpu_server_package-1.0.5.tar.gz`
- `EpiDive_vp_ho_cpu_openblas_server_package-1.0.5.tar.gz`
- `SHA256SUMS`

Each archive includes source code, a wheel, an installation script, a Conda environment definition, regression tests, and a Slurm batch template.

## Important behavior

- Both components default to `--maf 0.02` and `--min-mac 5`.
- GWES filtering is disabled unless `--apply-gwes-filter` is specified.
- Plotting and resume are enabled by default.
- Use `--no-resume` only when intentional full recomputation is required.
- CPU MKL and OpenBLAS packages have the same Python distribution name and should remain in separate Conda environments.

Read `docs/INSTALLATION.md` and run the included smoke/backend tests before production use.

## License and citation

The source is released under the MIT License. A formal citation record has not yet been supplied; cite the repository URL, this release tag, and the component version used.
