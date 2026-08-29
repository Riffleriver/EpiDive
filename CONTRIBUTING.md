# Contributing

Bug reports and focused pull requests are welcome once the project owner enables the corresponding GitHub features.

When reporting a problem, include:

- CPU or GPU component version and Git commit.
- Operating system, Python, Conda, BLAS, CUDA, and CuPy versions as applicable.
- A minimal command and a small non-sensitive input that reproduces the problem.
- The full traceback and relevant EpiDive log lines.
- Expected and observed behavior.

For code changes:

1. Keep CPU and GPU statistical semantics aligned unless a difference is intentional and documented.
2. Add or update a regression test.
3. Run the relevant smoke, MAC-filter, backend, conversion, and optimization tests.
4. Do not commit biological sample data, credentials, internal hostnames, IP addresses, absolute institutional paths, caches, wheels, or build directories.
5. Update `CHANGELOG.md` for user-visible behavior.

Large performance changes should include the dataset dimensions, hardware, command, wall time, peak memory, and result-equivalence checks.
