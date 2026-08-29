# Input and output

## SNP matrix

EpiDive expects a two-dimensional **loci × samples** matrix.

- The first column contains locus identifiers and is read as the row index.
- Each remaining column represents one sample.
- The background identifier supplied on the command line must match a row-index value.
- Supported text input is tab-delimited (`.tsv`, `.tab`, or `.txt`). Parquet input is also supported.
- Character alleles may be `A`, `T`, `C`, `G`, or `-`.
- Numeric matrices are accepted when all entries are finite and convertible to the internal unsigned-integer representation.

Example:

```text
locus\tsample_1\tsample_2\tsample_3
1001\tA\tA\tG
1002\tC\tT\tC
1003\t-\tG\tG
```

When a non-Parquet matrix is loaded, the workflow attempts to save a same-name Parquet copy for later reuse. Ensure the input directory is writable if you want this optimization.

## Background list

Batch mode accepts TSV or CSV. The selected column must contain numeric locus identifiers. Duplicate identifiers are removed while preserving first occurrence order.

```text
representative_snp
1001
1002
```

## Output behavior

The workflow writes analysis tables primarily as Parquet, together with optional figures and JSON/text status files. Exact filenames depend on the command and component version. Important operational artifacts include:

- `_full/`: reusable full-data VP cache.
- `_SUCCESS.json`: completed-background marker used by resume logic.
- `_ALLELE_0_SUCCESS.json` and `_ALLELE_1_SUCCESS.json`: subgroup completion markers.
- `_ERROR.txt` and shard-level batch error TSV files: failure diagnostics.
- Per-shard manifests and summaries for batch execution.

Do not edit success markers by hand. Resume compatibility is based on an analysis signature containing relevant inputs and parameters, including MAF and MAC settings.

With `--final-only`, known intermediate files are removed after each successful background, while final filtered results and status markers are retained.

## Data safety

Use a new output directory when changing biological input, component version, or interpretation-critical parameters. Preserve the command line and environment export next to archived results.
