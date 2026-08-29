# Usage

## One background

```bash
epidive-vp-ho-cpu \
  --input /data/snp_matrix.parquet \
  --output-dir /data/results/background_4938638 \
  --background 4938638 \
  --pair-jobs 32 \
  --prefix vp10k
```

Use `epidive-vp-ho-gpu` for the GPU implementation. The common analysis options have the same meaning.

## Batch backgrounds

Prepare a TSV or CSV with a numeric background column, for example:

```text
representative_snp
4938638
7685000
```

Run the batch:

```bash
epidive-vp-ho-gpu \
  --input /data/snp_matrix.parquet \
  --output-dir /data/results/background_batch \
  --background-file /data/backgrounds.tsv \
  --background-column representative_snp \
  --no-plot \
  --prefix vp10k_batch
```

The input matrix is loaded once and the full-data VP result is cached under `_full/`. Repeating a compatible command resumes incomplete backgrounds. Use `--no-resume` only when all selected backgrounds must be recomputed.

## Multi-server sharding

Every server uses the same input file and background list. Set the same shard count and a unique zero-based shard index:

```bash
epidive-vp-ho-gpu ... --shard-count 8 --shard-index 0
epidive-vp-ho-gpu ... --shard-count 8 --shard-index 1
# continue through --shard-index 7
```

## Important defaults

| Option | Default | Meaning |
|---|---:|---|
| `--maf` | `0.02` | Minimum minor allele frequency |
| `--min-mac` | `5` | Minimum minor allele count; `0` disables MAC filtering |
| `--model` | `SCC` | Sample reweighting mode (`SCC`, `HCC`, `NW`) |
| `--threshold` | `0.3` | Full-data initial threshold |
| `--subgroup-threshold` | `auto` | Subgroup initial threshold |
| `--full-k` | `1.0` | Full-data `k` |
| `--subgroup-1-k` | `1.0` | Allele-1 subgroup `k` |
| `--subgroup-0-k` | `2.0` | Allele-0 subgroup `k` |
| `--ess-mode` | `absolute` | Effective-sample-size screening |
| plotting | enabled | Disable with `--no-plot` |
| GWES filter | disabled | Enable with `--apply-gwes-filter` |
| resume | enabled | Disable with `--no-resume` |

CPU-specific controls include `--pair-jobs`, `--reweight-jobs`, `--cpu-block-size`, and `--accumulation-dtype`. GPU-specific controls include `--selected-pair-gpu`, `--gpu-block-size`, `--gpu-convert-jobs`, and `--reweight-jobs`.

For hybrid ESS screening and minimal batch storage:

```bash
--ess-mode hybrid \
--formal-min-fraction 0.05 \
--exploratory-min-fraction 0.02 \
--final-only
```

Always inspect the version-specific command reference:

```bash
epidive-vp-ho-cpu --help
epidive-vp-ho-gpu --help
```
