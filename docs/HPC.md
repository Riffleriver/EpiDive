# HPC and Slurm

The `cpu/run_background_batch.slurm` and `gpu/run_background_batch.slurm` files are templates. Review every path, partition, account, resource request, array bound, and concurrency limit before submission.

## CPU guidance

- Match `--pair-jobs` and `--reweight-jobs` to `--cpus-per-task`.
- Start with the default CPU block size (`4096`). Block size `8192` is covered by regression tests but needs more temporary memory.
- Avoid oversubscription: BLAS threads, Numba workers, and scheduler CPU allocation must agree.
- MKL is generally suitable for Intel servers; OpenBLAS is available for other environments or site policy.

## GPU guidance

- Allocate one visible GPU per process unless explicitly testing multi-GPU representative-mode calculation.
- Use `CUDA_VISIBLE_DEVICES` or the scheduler GPU binding so concurrent tasks do not compete for a device.
- Start with about 20 physical CPU cores per GPU for SCC reweighting, then benchmark on the actual node.
- `--gpu-block-size` controls selected GPU work; automatic memory-aware tiling is used for sample similarity.
- The NVIDIA driver must be compatible with the chosen CUDA runtime.

## Sharded batches

For `N` workers or servers, use identical input and background files, `--shard-count N`, and one distinct `--shard-index` from `0` to `N-1` per worker.

Use separate output roots per server to avoid concurrent metadata writes. After completion, verify each shard summary and error file before merging final tables.

## Operational checklist

1. Run smoke and backend tests on the allocated compute node.
2. Run two backgrounds with `--background-limit 2`.
3. Confirm expected GPU/BLAS backend in logs.
4. Confirm output size and memory use.
5. Submit the full array with resume enabled.
6. Check success markers, shard summaries, and error logs.

Never update a Conda environment while an active job is importing from it.
