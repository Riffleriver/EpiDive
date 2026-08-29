#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
epidis_gwes_vp: EpiDis 配对计算、GWES 绘图与异常值过滤（DataFrame 输入，可导入）
"""

from __future__ import annotations
import gc
import logging
import multiprocessing as mp
from typing import Callable, Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from epidive_data import DIR_IDS, GENE_REF_PATH, GENE_REP65_PATH

__all__ = [
    "run_gwes_vp",
    "run_gwes_vp_full",
    "bootstrap_pair_stability",
    "run_gwes_vp_bootstrap",
    "calculate_pair_distances",
    "calculate_pair_distance",
    "filter_gwes_pairs_with_threshold",
    "plot_gwes_with_threshold",
    "plot_bootstrap_stable_pairs",
    "genome_dis",
    "genome_detect",
    "genome_snp_dis",
    "genome_gene_pos",
    "genome_gene_dis",
    "process_chunk",
    "process_gene_pos_chunk",
    "detect_out",
    "find_outliers_k",
    "find_outliers",
    "plot_GWES_PU",
    "plot_GWES_high",
    "merge_pairs_with_fill",
]

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
pd.options.mode.chained_assignment = None

# ---- 染色体长度常量（可被 run_gwes_vp 动态覆盖）----
CHR1_LEN = 3288558
GENOME_TOTAL = 5165770
CHR2_LEN = GENOME_TOTAL - CHR1_LEN  # 1877212

# ----------------------------- 全局参考数据（懒加载） -----------------------------
_gene_ids: Optional[pd.DataFrame] = None
_gene_ref: Optional[pd.DataFrame] = None
_gene_rep65: Optional[pd.DataFrame] = None
_gene_pos_cache: dict[object, object] = {}

def _load_refs(
    dir_ids: str,
    gene_ref_path: str,
    gene_rep65_path: str
):
    global _gene_ids, _gene_ref, _gene_rep65
    if _gene_ids is None:
        # logging.info("Step 1: Loading gene reference data ...")
        gene_ids = pd.read_table(dir_ids, sep=' ', header=None)
        gene_ids = gene_ids.set_index(0)
        gene_ids.columns = ['g']
        _gene_ids = gene_ids
        _gene_ref = pd.read_csv(gene_ref_path, index_col='qacc')
        _gene_rep65 = pd.read_csv(gene_rep65_path, index_col='qacc')
        # logging.info("Gene reference data loaded.")

def _init_worker(dir_ids, gene_ref_path, gene_rep65_path, chr1_len, chr2_len):
    global CHR1_LEN, CHR2_LEN, GENOME_TOTAL
    CHR1_LEN = int(chr1_len)
    CHR2_LEN = int(chr2_len)
    GENOME_TOTAL = CHR1_LEN + CHR2_LEN
    _load_refs(dir_ids, gene_ref_path, gene_rep65_path)
    _gene_pos_cache.clear()
# --------------------------------- 距离计算 -----------------------------------
def genome_dis(pos1, pos2):
    if pos1 == pos2:
        return 0
    if pos1 > GENOME_TOTAL or pos2 > GENOME_TOTAL:
        return -1
    if pos1 <= CHR1_LEN and pos2 <= CHR1_LEN:
        ddis = abs(pos1 - pos2)
        if ddis > CHR1_LEN // 2:
            return CHR1_LEN - ddis
        return ddis
    if pos1 > CHR1_LEN and pos2 > CHR1_LEN:
        ddis = abs(pos1 - pos2)
        if ddis > CHR2_LEN // 2:
            return CHR2_LEN - ddis
        return ddis
    return -1

def genome_detect(loci1):
    gene1 = _gene_ids['g'].loc[loci1]
    if gene1 in _gene_ref.index:
        pos = int(abs(_gene_ref['sstart'].loc[gene1] - _gene_ref['send'].loc[gene1]) / 2)
        pos = pos + np.min([_gene_ref['sstart'].loc[gene1], _gene_ref['send'].loc[gene1]])
        if _gene_ref['sacc'].loc[gene1] == 'NC_004605':
            return pos + CHR1_LEN
        else:
            return pos
    if gene1 in _gene_rep65.index:
        return 0
    return -1

def genome_snp_dis(pos1, pos2):
    if pos1 == pos2:
        return 0
    if pos1 > GENOME_TOTAL or pos2 > GENOME_TOTAL:
        return -2
    if pos1 <= CHR1_LEN and pos2 <= CHR1_LEN:
        ddis = abs(pos1 - pos2)
        if ddis > CHR1_LEN // 2:
            return CHR1_LEN - ddis
        return ddis
    if pos1 > CHR1_LEN and pos2 > CHR1_LEN:
        ddis = abs(pos1 - pos2)
        if ddis > CHR2_LEN // 2:
            return CHR2_LEN - ddis
        return ddis
    return -1


def _genome_snp_dis_array(pos1, pos2):
    """Vectorized equivalent of :func:`genome_snp_dis`."""
    p1 = np.asarray(pos1)
    p2 = np.asarray(pos2)
    out = np.full(p1.shape, -1, dtype=np.int64)

    equal = p1 == p2
    out[equal] = 0
    invalid = (~equal) & ((p1 > GENOME_TOTAL) | (p2 > GENOME_TOTAL))
    out[invalid] = -2
    eligible = (~equal) & (~invalid)

    chr1 = eligible & (p1 <= CHR1_LEN) & (p2 <= CHR1_LEN)
    if np.any(chr1):
        distance = np.abs(p1[chr1] - p2[chr1])
        out[chr1] = np.where(
            distance > CHR1_LEN // 2,
            CHR1_LEN - distance,
            distance,
        ).astype(np.int64)

    chr2 = eligible & (p1 > CHR1_LEN) & (p2 > CHR1_LEN)
    if np.any(chr2):
        distance = np.abs(p1[chr2] - p2[chr2])
        out[chr2] = np.where(
            distance > CHR2_LEN // 2,
            CHR2_LEN - distance,
            distance,
        ).astype(np.int64)
    return out

def genome_gene_pos(loci1):
    if loci1 < GENOME_TOTAL:
        return None
    if loci1 in _gene_pos_cache:
        return _gene_pos_cache[loci1]
    gene1 = _gene_ids['g'].loc[loci1]
    if gene1 in _gene_ref.index:
        pos2 = int(_gene_ref['sstart'].loc[gene1])
        pos3 = int(_gene_ref['send'].loc[gene1])
        pos1 = int(np.median([pos2, pos3]))
        if _gene_ref['sacc'].loc[gene1] == 'NC_004605':
            pos1 += CHR1_LEN
            pos2 += CHR1_LEN
            pos3 += CHR1_LEN
        result = (pos1, pos2, pos3)
        _gene_pos_cache[loci1] = result
        return result
    if gene1 in _gene_rep65.index:
        contig1 = _gene_rep65['sacc'].loc[_gene_ids['g'].loc[loci1]]
        result = [contig1]
        _gene_pos_cache[loci1] = result
        return result
    _gene_pos_cache[loci1] = None
    return None

def genome_gene_dis(loci1, loci2):
    if loci1 == loci2:
        return 0
    if loci1 < GENOME_TOTAL and loci2 < GENOME_TOTAL:
        return genome_snp_dis(loci1, loci2)
    if loci1 < GENOME_TOTAL and loci2 > GENOME_TOTAL:
        pos123 = genome_gene_pos(loci2)
        if pos123 is not None and len(pos123) > 1:
            if pos123[1] <= loci1 <= pos123[2] or pos123[2] <= loci1 <= pos123[1]:
                return 0
            return np.min([genome_snp_dis(loci1, pos123[1]), genome_snp_dis(loci1, pos123[2])])
        return -1
    if loci1 > GENOME_TOTAL and loci2 < GENOME_TOTAL:
        pos123 = genome_gene_pos(loci1)
        if pos123 is not None and len(pos123) > 1:
            if pos123[1] <= loci2 <= pos123[2] or pos123[2] <= loci2 <= pos123[1]:
                return 0
            return np.min([genome_snp_dis(loci2, pos123[0]),
                           genome_snp_dis(loci2, pos123[1]),
                           genome_snp_dis(loci2, pos123[2])])
        return -1
    if loci1 > GENOME_TOTAL and loci2 > GENOME_TOTAL:
        pos123a = genome_gene_pos(loci1)
        pos123b = genome_gene_pos(loci2)
        if pos123a is not None and len(pos123a) > 1 and pos123b is not None and len(pos123b) > 1:
            if np.min(pos123a) <= CHR1_LEN and np.min(pos123b) > CHR1_LEN:
                return -1
            if np.min(pos123a) > CHR1_LEN and np.min(pos123b) <= CHR1_LEN:
                return -1
            if np.min(pos123a) <= np.max(pos123b) and np.min(pos123b) <= np.max(pos123a):
                return 0
            smaller_interval = min(pos123a, pos123b)
            larger_interval = max(pos123a, pos123b)
            a = genome_snp_dis(smaller_interval[2], larger_interval[1])
            b = genome_snp_dis(smaller_interval[1], larger_interval[2])
            return min(a, b)
        if pos123a is not None and pos123b is not None and len(pos123a) == 1 and len(pos123b) == 1:
            if pos123a[0] == pos123b[0]:
                return 0
    return -1

# --------------------------------- 分块处理 -----------------------------------
def _standardize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    if {'locus1','locus2','v'}.issubset(df.columns):
        sub = df[['locus1','locus2','v']].copy()
    else:
        if df.shape[1] < 3:
            raise ValueError("Input DataFrame must have at least 3 columns: locus1, locus2, v")
        sub = df.iloc[:, :3].copy()
        sub.columns = ['locus1','locus2','v']
    return sub

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = ['locus1', 'locus2', 'v']
    chunk['v'] = chunk['v'].astype(float)
    chunk['locus1'] = pd.to_numeric(chunk['locus1'], errors='coerce')
    chunk['locus2'] = pd.to_numeric(chunk['locus2'], errors='coerce')
    locus1 = chunk['locus1'].to_numpy(copy=False)
    locus2 = chunk['locus2'].to_numpy(copy=False)
    geno_dis = np.empty(len(chunk), dtype=np.int64)

    # SNP-SNP pairs dominate the input and can be handled in one vectorized
    # operation.  Mixed/gene pairs retain the exact legacy branch logic while
    # genome_gene_pos uses a per-process cache.
    snp_snp = (locus1 < GENOME_TOTAL) & (locus2 < GENOME_TOTAL)
    if np.any(snp_snp):
        geno_dis[snp_snp] = _genome_snp_dis_array(
            locus1[snp_snp], locus2[snp_snp]
        )
    other_positions = np.flatnonzero(~snp_snp)
    if other_positions.size:
        geno_dis[other_positions] = np.fromiter(
            (
                genome_gene_dis(locus1[i], locus2[i])
                for i in other_positions
            ),
            dtype=np.int64,
            count=len(other_positions),
        )
    chunk['geno_dis'] = geno_dis
    chunk['LD'] = 0
    chunk.loc[(chunk['geno_dis'] < 3000) & (chunk['geno_dis'] >= 0), 'LD'] = 1
    return chunk

def process_gene_pos_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    unique_loci = pd.unique(
        pd.concat([chunk['locus1'], chunk['locus2']], ignore_index=True)
    )
    position_map = {locus: genome_gene_pos(locus) for locus in unique_loci}
    chunk['gene1_pos'] = chunk['locus1'].map(position_map)
    chunk['gene2_pos'] = chunk['locus2'].map(position_map)
    return chunk


def calculate_pair_distances(
    df_input: pd.DataFrame,
    *,
    dir_ids: str = DIR_IDS,
    gene_ref_path: str = GENE_REF_PATH,
    gene_rep65_path: str = GENE_REP65_PATH,
    chunk_size: int = 100_000,
    n_jobs: int = 1,
    ld_threshold: int = 3_000,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate physical/genomic distance for locus pairs only.

    The input needs ``locus1`` and ``locus2`` columns. All original columns and
    row order are retained; existing ``geno_dis`` and ``LD`` columns are
    replaced with values calculated for the current physical locus pair.

    ``geno_dis`` follows the existing GWES conventions:

    - non-negative: distance on the same chromosome/contig;
    - ``-1``: cross-chromosome or no comparable mapped distance;
    - ``-2``: an invalid SNP-only coordinate according to the legacy rule.

    ``LD`` is 1 when ``0 <= geno_dis < ld_threshold`` and 0 otherwise.
    This function performs no IQR/KDE filtering, plotting, or file writing.
    """
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input must be a pandas DataFrame")
    missing = {"locus1", "locus2"}.difference(df_input.columns)
    if missing:
        raise ValueError(f"df_input is missing columns: {sorted(missing)}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if ld_threshold < 0:
        raise ValueError("ld_threshold cannot be negative")
    if chr1_len <= 0 or chr2_len <= 0:
        raise ValueError("chr1_len and chr2_len must be positive")

    output = df_input.copy()
    locus1 = pd.to_numeric(output["locus1"], errors="coerce")
    locus2 = pd.to_numeric(output["locus2"], errors="coerce")
    invalid = locus1.isna() | locus2.isna()
    if invalid.any():
        bad_rows = np.flatnonzero(invalid.to_numpy())[:10].tolist()
        raise ValueError(
            f"locus1/locus2 contain missing or non-numeric values in "
            f"{int(invalid.sum())} rows; first row positions: {bad_rows}"
        )

    global CHR1_LEN, CHR2_LEN, GENOME_TOTAL
    CHR1_LEN = int(chr1_len)
    CHR2_LEN = int(chr2_len)
    GENOME_TOTAL = CHR1_LEN + CHR2_LEN
    _gene_pos_cache.clear()

    # process_chunk retains the exact legacy distance branches and expects a
    # three-column frame. A temporary v column lets this public helper also
    # accept pair tables that do not contain EpiDis scores.
    work = pd.DataFrame({
        "locus1": locus1.to_numpy(copy=False),
        "locus2": locus2.to_numpy(copy=False),
        "v": np.zeros(len(output), dtype=np.float64),
    })
    n = len(work)
    if n == 0:
        output["geno_dis"] = pd.Series(index=output.index, dtype=np.int64)
        output["LD"] = pd.Series(index=output.index, dtype=np.int8)
        return output

    needs_gene_refs = bool(
        ((locus1 > GENOME_TOTAL) | (locus2 > GENOME_TOTAL)).any()
    )
    if needs_gene_refs and n_jobs == 1:
        _load_refs(dir_ids, gene_ref_path, gene_rep65_path)

    chunk_starts = range(0, n, int(chunk_size))
    chunk_count = (n + int(chunk_size) - 1) // int(chunk_size)
    chunks = (work.iloc[start:start + int(chunk_size)].copy()
              for start in chunk_starts)
    jobs = mp.cpu_count() if n_jobs <= 0 else int(n_jobs)

    if verbose:
        logging.info(
            "Calculating distances for %d pairs: chunks=%d, jobs=%d",
            n, chunk_count, jobs,
        )
    if jobs == 1:
        parts = [
            process_chunk(chunk)
            for chunk in tqdm(chunks, total=chunk_count, disable=not verbose)
        ]
    else:
        # The existing worker initializer also loads gene references. This is
        # slightly more setup for SNP-only tables but keeps spawned workers
        # compatible with mixed SNP/gene chunks.
        with mp.get_context("spawn").Pool(
            processes=jobs,
            initializer=_init_worker,
            initargs=(
                dir_ids,
                gene_ref_path,
                gene_rep65_path,
                CHR1_LEN,
                CHR2_LEN,
            ),
        ) as pool:
            parts = list(tqdm(
                pool.imap(process_chunk, chunks),
                total=chunk_count,
                disable=not verbose,
            ))

    calculated = pd.concat(parts, ignore_index=True)
    distances = calculated["geno_dis"].to_numpy(dtype=np.int64, copy=False)
    output["geno_dis"] = distances
    output["LD"] = (
        (distances >= 0) & (distances < int(ld_threshold))
    ).astype(np.int8)
    if verbose:
        logging.info("Distance calculation complete: %d pairs", len(output))
    return output


# Singular alias for callers who naturally search for this name.
calculate_pair_distance = calculate_pair_distances


def filter_gwes_pairs_with_threshold(
    df_input: pd.DataFrame,
    threshold: float,
    *,
    ld_chr1: int = 5_000,
    ld_chr2: int = 17_500,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    dir_ids: str = DIR_IDS,
    gene_ref_path: str = GENE_REF_PATH,
    gene_rep65_path: str = GENE_REP65_PATH,
    include_cross_chromosome: bool = True,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply the final fixed-threshold filtering rules of ``run_gwes_vp``.

    The input must already contain physical-pair ``geno_dis`` values, normally
    produced by :func:`calculate_pair_distances`. The filtering rules are:

    - chr1 comparable pairs: ``geno_dis > ld_chr1``;
    - chr2 comparable pairs: ``geno_dis > ld_chr2``;
    - cross-chromosome/unmapped pairs: ``geno_dis == -1`` when enabled;
    - every retained row: ``v > threshold`` (strictly greater, matching the
      original implementation).

    Rows with ``geno_dis == -2`` and same-chromosome proximal/overlapping rows
    are excluded. All input metadata columns are preserved.
    """
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input must be a pandas DataFrame")
    required = {"locus1", "locus2", "v", "geno_dis"}
    missing = required.difference(df_input.columns)
    if missing:
        raise ValueError(f"df_input is missing columns: {sorted(missing)}")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if ld_chr1 < 0 or ld_chr2 < 0:
        raise ValueError("ld_chr1 and ld_chr2 cannot be negative")
    if chr1_len <= 0 or chr2_len <= 0:
        raise ValueError("chr1_len and chr2_len must be positive")

    data = df_input.copy()
    for column in ("locus1", "locus2", "v", "geno_dis"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"column {column!r} contains missing or non-numeric values"
            )
        data[column] = numeric

    global CHR1_LEN, CHR2_LEN, GENOME_TOTAL
    CHR1_LEN = int(chr1_len)
    CHR2_LEN = int(chr2_len)
    GENOME_TOTAL = CHR1_LEN + CHR2_LEN
    _gene_pos_cache.clear()

    comparable = data.loc[data["geno_dis"] >= 0].copy()
    locus1 = comparable["locus1"]
    locus2 = comparable["locus2"]
    chr1_mask = (
        ((locus1 < CHR1_LEN) & (locus2 < CHR1_LEN)) |
        ((locus1 < CHR1_LEN) & (locus2 > GENOME_TOTAL)) |
        ((locus2 < CHR1_LEN) & (locus1 > GENOME_TOTAL))
    )
    chr2_mask = (
        (locus1 > CHR1_LEN) &
        (locus2 > CHR1_LEN) &
        ((locus1 < GENOME_TOTAL) | (locus2 < GENOME_TOTAL))
    )
    chr1_pairs = comparable.loc[chr1_mask].copy()
    chr2_pairs = comparable.loc[chr2_mask].copy()

    # Match run_gwes_vp for mapped accessory-gene pairs whose two identifiers
    # are both outside the SNP-coordinate range.
    gene_gene = comparable.loc[
        (comparable["locus1"] > GENOME_TOTAL) &
        (comparable["locus2"] > GENOME_TOTAL)
    ].copy()
    gene_gene_classified = 0
    if not gene_gene.empty:
        _load_refs(dir_ids, gene_ref_path, gene_rep65_path)
        gene_gene = process_gene_pos_chunk(gene_gene)
        valid_positions = (
            gene_gene["gene1_pos"].apply(
                lambda value: isinstance(value, (list, tuple)) and len(value) == 3
            ) &
            gene_gene["gene2_pos"].apply(
                lambda value: isinstance(value, (list, tuple)) and len(value) == 3
            )
        )
        gene_gene = gene_gene.loc[valid_positions]
        gene_chr1 = gene_gene.loc[
            gene_gene["gene1_pos"].apply(
                lambda value: all(position < CHR1_LEN for position in value)
            )
        ].drop(columns=["gene1_pos", "gene2_pos"])
        gene_chr2 = gene_gene.loc[
            gene_gene["gene1_pos"].apply(
                lambda value: all(position > CHR1_LEN for position in value)
            )
        ].drop(columns=["gene1_pos", "gene2_pos"])
        chr1_pairs = pd.concat([chr1_pairs, gene_chr1], ignore_index=True)
        chr2_pairs = pd.concat([chr2_pairs, gene_chr2], ignore_index=True)
        gene_gene_classified = len(gene_chr1) + len(gene_chr2)

    selected_chr1 = chr1_pairs.loc[
        (chr1_pairs["geno_dis"] > int(ld_chr1)) &
        (chr1_pairs["v"] > float(threshold))
    ]
    selected_chr2 = chr2_pairs.loc[
        (chr2_pairs["geno_dis"] > int(ld_chr2)) &
        (chr2_pairs["v"] > float(threshold))
    ]
    if include_cross_chromosome:
        selected_cross = data.loc[
            (data["geno_dis"] == -1) &
            (data["v"] > float(threshold))
        ]
    else:
        selected_cross = data.iloc[0:0]

    selected = pd.concat(
        [selected_chr1, selected_chr2, selected_cross], ignore_index=True
    )
    selected.attrs.update({
        "epidis_threshold": float(threshold),
        "ld_chr1": int(ld_chr1),
        "ld_chr2": int(ld_chr2),
        "include_cross_chromosome": bool(include_cross_chromosome),
        "input_rows": int(len(data)),
        "chr1_selected_rows": int(len(selected_chr1)),
        "chr2_selected_rows": int(len(selected_chr2)),
        "cross_selected_rows": int(len(selected_cross)),
    })

    if output_path is not None:
        from pathlib import Path
        destination = Path(output_path)
        if destination.suffix.lower() not in {".parquet", ".pq"}:
            raise ValueError("output_path must end with .parquet or .pq")
        destination.parent.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(destination, index=False)
        if verbose:
            logging.info("Filtered GWES pairs saved to %s", destination)
    if verbose:
        logging.info(
            "Fixed GWES filtering: input=%d, chr1=%d, chr2=%d, cross=%d, "
            "final=%d, threshold=%.6f, mapped gene-gene=%d",
            len(data), len(selected_chr1), len(selected_chr2),
            len(selected_cross), len(selected), float(threshold),
            gene_gene_classified,
        )
    return selected


def _unordered_pair_index(df: pd.DataFrame) -> pd.MultiIndex:
    """Build a normalized, unordered locus-pair index."""
    locus1 = pd.to_numeric(df["locus1"], errors="coerce")
    locus2 = pd.to_numeric(df["locus2"], errors="coerce")
    if locus1.isna().any() or locus2.isna().any():
        raise ValueError("pairs contain missing or non-numeric locus values")
    values1 = locus1.to_numpy(dtype=np.int64, copy=False)
    values2 = locus2.to_numpy(dtype=np.int64, copy=False)
    return pd.MultiIndex.from_arrays(
        [np.minimum(values1, values2), np.maximum(values1, values2)],
        names=["_pair_low", "_pair_high"],
    )


def bootstrap_pair_stability(
    data_df: pd.DataFrame,
    weight_ser: pd.Series | pd.DataFrame,
    pairs: pd.DataFrame,
    threshold: float,
    *,
    n_bootstrap: int = 100,
    stability_cutoff: float = 0.80,
    random_state: Optional[int] = 20260816,
    pair_calculator: Optional[Callable] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate candidate-pair stability by resampling samples.

    Each bootstrap iteration samples the columns of ``data_df`` with
    replacement, carries the corresponding sample weights with them, and
    calculates EpiDis only for ``pairs``. No intermediate files are written.

    Parameters
    ----------
    data_df
        Unique-pattern locus-by-sample matrix used for the representative-pair
        analysis. Passing the restored physical matrix would incorrectly give
        duplicate patterns extra statistical weight.
    pairs
        Candidate representative pairs, normally the k=2 result returned by
        :func:`run_gwes_vp`.
    threshold
        Fixed EpiDis threshold estimated from the original, non-bootstrap
        representative-pattern analysis.
    pair_calculator
        A function compatible with ``epidis_cpu_test.run_epidis_for_pairs``.
        When omitted, that function is imported automatically.

    Returns
    -------
    stable_pairs, all_candidates
        ``all_candidates`` contains Bootstrap pass counts, valid counts,
        stability, mean and standard deviation. ``stable_pairs`` is its subset
        with ``bootstrap_stability >= stability_cutoff``.
    """
    if not isinstance(data_df, pd.DataFrame) or data_df.empty:
        raise ValueError("data_df must be a non-empty pandas DataFrame")
    if not isinstance(pairs, pd.DataFrame):
        raise TypeError("pairs must be a pandas DataFrame")
    missing = {"locus1", "locus2"}.difference(pairs.columns)
    if missing:
        raise ValueError(f"pairs is missing columns: {sorted(missing)}")
    if not data_df.index.is_unique:
        raise ValueError("data_df index must contain unique locus identifiers")
    if not data_df.columns.is_unique:
        raise ValueError("data_df columns must contain unique sample identifiers")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0.0 <= stability_cutoff <= 1.0:
        raise ValueError("stability_cutoff must be between 0 and 1")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    candidates = pairs.copy().reset_index(drop=True)
    candidate_index = _unordered_pair_index(candidates)
    if candidate_index.duplicated().any():
        raise ValueError("pairs contains duplicate unordered locus pairs")
    if candidates.empty:
        for column, dtype in (
            ("bootstrap_pass_count", np.int32),
            ("bootstrap_valid_count", np.int32),
            ("bootstrap_stability", np.float64),
            ("bootstrap_v_mean", np.float64),
            ("bootstrap_v_std", np.float64),
        ):
            candidates[column] = pd.Series(dtype=dtype)
        return candidates.copy(), candidates

    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError("weight_ser DataFrame must have exactly one column")
        weight_ser = weight_ser.iloc[:, 0]
    if not isinstance(weight_ser, pd.Series):
        raise TypeError("weight_ser must be a Series or one-column DataFrame")
    aligned_weights = weight_ser.reindex(data_df.columns)
    weights = aligned_weights.to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all():
        bad = aligned_weights.index[~np.isfinite(weights)].tolist()[:10]
        raise ValueError(
            f"weight_ser has missing/non-finite weights; first samples: {bad}"
        )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample weights must be non-negative with positive sum")

    if pair_calculator is None:
        try:
            from . import epidis_cpu_test as _pair_module
        except ImportError:
            import epidis_cpu_test as _pair_module
        pair_calculator = _pair_module.run_epidis_for_pairs
    else:
        _pair_module = None

    values = data_df.to_numpy(copy=False)
    is_binary = (
        np.issubdtype(values.dtype, np.number)
        and np.isfinite(values).all()
        and np.all((values == 0) | (values == 1))
    )
    if is_binary:
        binary_df = data_df
    else:
        if _pair_module is None:
            try:
                from . import epidis_cpu_test as _pair_module
            except ImportError:
                import epidis_cpu_test as _pair_module
        if verbose:
            logging.info("Converting bootstrap input matrix to binary once ...")
        binary_df = _pair_module.convert_snp_to_binary(
            data_df, n_jobs=1, convert_chunksize=20_000
        )

    locus_values = pd.to_numeric(
        pd.Series(binary_df.index), errors="coerce"
    )
    if locus_values.isna().any():
        raise ValueError("data_df index must be numeric for run_epidis_for_pairs")
    available_loci = set(locus_values.astype(np.int64).tolist())
    pair_loci = set(candidate_index.get_level_values(0)).union(
        candidate_index.get_level_values(1)
    )
    missing_loci = pair_loci.difference(available_loci)
    if missing_loci:
        raise ValueError(
            f"{len(missing_loci)} candidate loci are missing from data_df; "
            f"first loci: {list(missing_loci)[:10]}"
        )

    pair_input = candidates[["locus1", "locus2"]]
    pair_count = len(candidates)
    pass_count = np.zeros(pair_count, dtype=np.int32)
    valid_count = np.zeros(pair_count, dtype=np.int32)
    value_sum = np.zeros(pair_count, dtype=np.float64)
    value_sumsq = np.zeros(pair_count, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    sample_count = binary_df.shape[1]
    iterator = range(int(n_bootstrap))
    iterator = tqdm(
        iterator,
        total=int(n_bootstrap),
        desc="Bootstrap EpiDis stability",
        disable=not verbose,
    )

    for bootstrap_id in iterator:
        sampled = rng.integers(0, sample_count, size=sample_count)
        bootstrap_df = binary_df.iloc[:, sampled].copy()
        bootstrap_names = [
            f"_bootstrap_{bootstrap_id}_{position}"
            for position in range(sample_count)
        ]
        bootstrap_df.columns = bootstrap_names
        bootstrap_weights = pd.Series(
            weights[sampled], index=bootstrap_names, dtype=np.float64
        )
        calculated = pair_calculator(
            data_df=bootstrap_df,
            weight_ser=bootstrap_weights,
            pairs=pair_input,
            threshold=None,
            n_jobs=n_jobs,
            prefix=None,
        )
        result_index = _unordered_pair_index(calculated)
        if result_index.duplicated().any():
            raise ValueError("pair_calculator returned duplicate locus pairs")
        score_lookup = pd.Series(
            pd.to_numeric(calculated["v"], errors="coerce").to_numpy(),
            index=result_index,
        )
        scores = score_lookup.reindex(candidate_index).to_numpy(dtype=np.float64)
        valid = np.isfinite(scores)
        valid_count += valid
        pass_count += valid & (scores >= float(threshold))
        safe_scores = np.where(valid, scores, 0.0)
        value_sum += safe_scores
        value_sumsq += safe_scores * safe_scores

    denominator = np.maximum(valid_count, 1)
    mean = value_sum / denominator
    variance = np.maximum(value_sumsq / denominator - mean * mean, 0.0)
    candidates["bootstrap_pass_count"] = pass_count
    candidates["bootstrap_valid_count"] = valid_count
    candidates["bootstrap_stability"] = pass_count / denominator
    candidates["bootstrap_v_mean"] = mean
    candidates["bootstrap_v_std"] = np.sqrt(variance)
    stable = candidates.loc[
        (candidates["bootstrap_valid_count"] == int(n_bootstrap)) &
        (candidates["bootstrap_stability"] >= float(stability_cutoff))
    ].copy().reset_index(drop=True)
    candidates.attrs.update({
        "epidis_threshold": float(threshold),
        "n_bootstrap": int(n_bootstrap),
        "stability_cutoff": float(stability_cutoff),
        "random_state": random_state,
    })
    stable.attrs.update(candidates.attrs)
    if verbose:
        logging.info(
            "Bootstrap stability complete: candidates=%d, stable=%d "
            "(cutoff=%.2f)",
            len(candidates), len(stable), float(stability_cutoff),
        )
    return stable, candidates

# --------------------------------- 异常值检测 ---------------------------------
def detect_out(df_sp: pd.Series | pd.DataFrame, k: float):
    q3 = df_sp.v.quantile(0.75)
    q1 = df_sp.v.quantile(0.25)
    outlier = q3 + k * 1.5 * (q3 - q1)
    return outlier

def find_outliers_k(df_sp1: pd.Series | pd.DataFrame, k: float):
    if k < 1:
        return k
    k = int(k)
    df_sp = df_sp1.copy()
    outlier_k = detect_out(df_sp, k)
    logging.info("Calculated outlier threshold (k=%d): %.2f", k, outlier_k)
    return outlier_k

def find_outliers(df_sp1: pd.DataFrame| pd.DataFrame, k: float):
    df_sp = df_sp1.copy()
    if k<1:
        outlier_k1 = find_outliers_k(df_sp, k)
        outlier_k2 = find_outliers_k(df_sp, k)
        outlier_k3 = find_outliers_k(df_sp, k)
    else:
        outlier_k1 = find_outliers_k(df_sp, 1)
        outlier_k2 = find_outliers_k(df_sp, 2)
        outlier_k3 = find_outliers_k(df_sp, 3)
    return [outlier_k1, outlier_k2, outlier_k3]

# ------------------------------ KDE 异常值检测（新增） ------------------------------
def _safe_kde_peak(values, bw='scott'):
    """
    安全求 KDE 峰位置：样本不足或零方差返回 None
    bw: 'scott' | 'silverman' | float
    """
    from scipy.stats import gaussian_kde

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.std(v) == 0:
        return None
    xs = np.linspace(v.min(), v.max(), 512)
    kde = gaussian_kde(v, bw_method=bw)
    ys = kde(xs)
    return float(xs[np.argmax(ys)])

def find_outliers_kde_upper(df_sp1: pd.DataFrame, margin: float = 0.1, bw='scott') -> float:
    """
    只返回上界阈值：thr = peak + margin
    目标列为 df_sp1['v']；若不可得则回退到 IQR(k=3)
    """
    s = pd.Series(df_sp1['v']).dropna().values
    peak = _safe_kde_peak(s, bw=bw)
    if peak is not None:
        return peak + float(margin)
    # 回退：IQR(k=3)
    return float(find_outliers_k(df_sp1, 3))

# ----------------------------------- 绘图 -------------------------------------
def plot_GWES_PU(df_sp2, LD_dis, flag_chr, outliers, save_prefix, outlier_mode: str = 'iqr'):
    import seaborn as sns
    from matplotlib import pyplot as plt

    plot_l = 28 if flag_chr == 'chr1' else 16

    df_sp = df_sp2.copy()
    df_sp = df_sp[df_sp['geno_dis'] > 0]
    LD_dis = int(LD_dis)
    df_sp_PLD = df_sp[df_sp['geno_dis'] <= LD_dis]
    df_sp_LD = df_sp[df_sp['geno_dis'] > LD_dis]

    outlier = outliers[0] if outliers[0] is not None else 0
    ext_outlier = outliers[1] if outliers[1] is not None else 0
    ext_plus_outlier = outliers[2] if outliers[2] is not None else 0

    df_sp_ex = df_sp_LD[df_sp_LD['v'] > outlier]
    df_sp0 = df_sp_ex[~((df_sp_ex['locus1'] > GENOME_TOTAL) | (df_sp_ex['locus2'] > GENOME_TOTAL))]
    df_sp1 = df_sp_ex[((df_sp_ex['locus1'] < GENOME_TOTAL) & (df_sp_ex['locus2'] > GENOME_TOTAL)) |
                       ((df_sp_ex['locus1'] > GENOME_TOTAL) & (df_sp_ex['locus2'] < GENOME_TOTAL))]
    df_sp2 = df_sp_ex[(df_sp_ex['locus1'] > GENOME_TOTAL) & (df_sp_ex['locus2'] > GENOME_TOTAL)]

    sns.set_style("white")
    plt.figure(figsize=(plot_l, 7))
    plt.plot(df_sp['geno_dis'].values, df_sp['v'].values, '.', alpha=0.6, markersize=0.7, color='linen', label='Common pairs')
    plt.plot(df_sp_PLD['geno_dis'].values, df_sp_PLD['v'].values, '.', alpha=0.5, markersize=0.7, color='turquoise', label='PLD pairs')
    plt.plot(df_sp0['geno_dis'].values, df_sp0['v'].values, '.', alpha=0.6, markersize=0.7, color='deepskyblue', label='Strong pairs(SNP-SNP)')
    plt.plot(df_sp1['geno_dis'].values, df_sp1['v'].values, '.', alpha=0.6, markersize=0.7, color='limegreen', label='Strong pairs(SNP-Accessory gene)')
    plt.plot(df_sp2['geno_dis'].values, df_sp2['v'].values, '.', alpha=0.6, markersize=0.7, color='greenyellow', label='Strong pairs(Accessory gene-gene)')

    # === 仅此处分支：KDE 只画一条线；IQR 画三条线 ===
    if outlier_mode == 'kde':
        plt.axhline(y=outlier, c="r", ls="--")
    else:
        plt.axhline(y=outlier, c="r", ls="--")
        plt.axhline(y=ext_outlier, c="r", ls="--")
        plt.axhline(y=ext_plus_outlier, c="r", ls="--")

    ylim = plt.ylim()
    plt.text(LD_dis, ylim[1], 'P_ld: ' + str(LD_dis), va='bottom', ha='left', color='red')
    plt.axvline(x=LD_dis, c="r", ls="--")

    xlim = plt.xlim()
    if outlier_mode == 'kde':
        plt.text(xlim[1], outlier, f'KDE threshold: {np.around(outlier, 2)}', va='bottom', ha='right', color='red')
    else:
        plt.text(xlim[1], outlier, f'Outlier(k=1): {np.around(outlier,2)}', va='bottom', ha='right', color='red')
        plt.text(xlim[1], ext_outlier, f'Outlier(k=2): {np.around(ext_outlier,2)}', va='bottom', ha='right', color='red')
        plt.text(xlim[1], ext_plus_outlier, f'Outlier(k=3): {np.around(ext_plus_outlier,2)}', va='bottom', ha='right', color='red')

    plt.xlabel('Genome Distance', fontsize=16)
    plt.ylabel('EpiDis Value', fontsize=16)
    plt.title('EpiDis_GWES ' + flag_chr, fontsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(axis='both', which='both', direction='out', length=6, width=2, colors='black', labelsize=12)
    leg = plt.legend(loc='upper right', fontsize=12, markerscale=17)
    leg.get_frame().set_alpha(1)
    save_path = save_prefix + "GWES_Epi_LD_gene.png"
    plt.savefig(save_path, dpi=220)
    plt.close()
    logging.info("Plot saved to %s", save_path)
    return outlier, ext_outlier


def _plot_gwes_fixed_threshold_panel(
    df_pairs: pd.DataFrame,
    *,
    threshold: float,
    ld_distance: int,
    chromosome: str,
    output_path: str,
    dpi: int,
    title_prefix: str,
) -> None:
    """Draw one GWES panel using a caller-supplied fixed threshold."""
    import seaborn as sns
    from matplotlib import pyplot as plt

    data = df_pairs.loc[df_pairs["geno_dis"] > 0].copy()
    proximal = data.loc[data["geno_dis"] <= ld_distance]
    distant = data.loc[data["geno_dis"] > ld_distance]
    common = distant.loc[distant["v"] <= threshold]
    strong = distant.loc[distant["v"] > threshold]

    snp_snp = strong.loc[
        ~((strong["locus1"] > GENOME_TOTAL) |
          (strong["locus2"] > GENOME_TOTAL))
    ]
    snp_gene = strong.loc[
        ((strong["locus1"] < GENOME_TOTAL) &
         (strong["locus2"] > GENOME_TOTAL)) |
        ((strong["locus1"] > GENOME_TOTAL) &
         (strong["locus2"] < GENOME_TOTAL))
    ]
    gene_gene = strong.loc[
        (strong["locus1"] > GENOME_TOTAL) &
        (strong["locus2"] > GENOME_TOTAL)
    ]

    sns.set_style("white")
    figure_width = 28 if chromosome == "chr1" else 16
    plt.figure(figsize=(figure_width, 7))
    plt.plot(
        common["geno_dis"].values, common["v"].values, ".",
        alpha=0.6, markersize=0.7, color="linen", label="Common pairs",
    )
    plt.plot(
        proximal["geno_dis"].values, proximal["v"].values, ".",
        alpha=0.5, markersize=0.7, color="turquoise", label="PLD pairs",
    )
    plt.plot(
        snp_snp["geno_dis"].values, snp_snp["v"].values, ".",
        alpha=0.6, markersize=0.7, color="deepskyblue",
        label="Strong pairs(SNP-SNP)",
    )
    plt.plot(
        snp_gene["geno_dis"].values, snp_gene["v"].values, ".",
        alpha=0.6, markersize=0.7, color="limegreen",
        label="Strong pairs(SNP-Accessory gene)",
    )
    plt.plot(
        gene_gene["geno_dis"].values, gene_gene["v"].values, ".",
        alpha=0.6, markersize=0.7, color="greenyellow",
        label="Strong pairs(Accessory gene-gene)",
    )

    plt.axhline(y=threshold, color="red", linestyle="--")
    plt.axvline(x=ld_distance, color="red", linestyle="--")
    ylim = plt.ylim()
    axis = plt.gca()
    plt.text(
        ld_distance, ylim[1], f"P_ld: {ld_distance}",
        va="bottom", ha="left", color="red",
    )
    axis.text(
        0.99, threshold, f"Fixed threshold: {threshold:.4f}",
        transform=axis.get_yaxis_transform(),
        va="bottom", ha="right", color="red",
    )
    plt.xlabel("Genome Distance", fontsize=16)
    plt.ylabel("EpiDis Value", fontsize=16)
    plt.title(f"{title_prefix} {chromosome}", fontsize=16)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().tick_params(
        axis="both", which="both", direction="out", length=6, width=2,
        colors="black", labelsize=12,
    )
    legend = plt.legend(loc="upper right", fontsize=12, markerscale=17)
    legend.get_frame().set_alpha(1)
    plt.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close()
    logging.info("Restored GWES plot saved to %s", output_path)


def plot_gwes_with_threshold(
    df_input: pd.DataFrame,
    threshold: float,
    prefix: str,
    *,
    ld_chr1: int = 5_000,
    ld_chr2: int = 17_500,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    dir_ids: str = DIR_IDS,
    gene_ref_path: str = GENE_REF_PATH,
    gene_rep65_path: str = GENE_REP65_PATH,
    dpi: int = 220,
    title_prefix: str = "Restored EpiDis_GWES",
    verbose: bool = True,
) -> dict[str, object]:
    """Plot restored GWES pairs using an already estimated EpiDis threshold.

    ``df_input`` must already contain physical-pair ``geno_dis`` values, for
    example from :func:`calculate_pair_distances`. The supplied threshold is
    drawn and applied directly; it is never re-estimated from restored rows.

    Two figures are written: ``<prefix>.chr1.GWES_Epi_restored.png`` and
    ``<prefix>.chr2.GWES_Epi_restored.png``. This function does not save the
    pair table; use ``filter_gwes_pairs_with_threshold(output_path=...)`` to
    save the final filtered result.
    """
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input must be a pandas DataFrame")
    required = {"locus1", "locus2", "v", "geno_dis"}
    missing = required.difference(df_input.columns)
    if missing:
        raise ValueError(f"df_input is missing columns: {sorted(missing)}")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if ld_chr1 < 0 or ld_chr2 < 0:
        raise ValueError("ld_chr1 and ld_chr2 cannot be negative")

    global CHR1_LEN, CHR2_LEN, GENOME_TOTAL
    CHR1_LEN = int(chr1_len)
    CHR2_LEN = int(chr2_len)
    GENOME_TOTAL = CHR1_LEN + CHR2_LEN
    _gene_pos_cache.clear()

    data = df_input.copy()
    for column in ("locus1", "locus2", "v", "geno_dis"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid = data[["locus1", "locus2", "v", "geno_dis"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} rows contain missing/non-numeric locus, v, "
            "or geno_dis values"
        )

    comparable = data.loc[data["geno_dis"] >= 0].copy()
    chr1_mask = (
        ((comparable["locus1"] < CHR1_LEN) &
         (comparable["locus2"] < CHR1_LEN)) |
        ((comparable["locus1"] < CHR1_LEN) &
         (comparable["locus2"] > GENOME_TOTAL)) |
        ((comparable["locus2"] < CHR1_LEN) &
         (comparable["locus1"] > GENOME_TOTAL))
    )
    chr2_mask = (
        (comparable["locus1"] > CHR1_LEN) &
        (comparable["locus2"] > CHR1_LEN) &
        ((comparable["locus1"] < GENOME_TOTAL) |
         (comparable["locus2"] < GENOME_TOTAL))
    )
    chr1 = comparable.loc[chr1_mask].copy()
    chr2 = comparable.loc[chr2_mask].copy()

    # Gene-gene pairs need reference positions to determine their chromosome.
    gene_gene = comparable.loc[
        (comparable["locus1"] > GENOME_TOTAL) &
        (comparable["locus2"] > GENOME_TOTAL)
    ].copy()
    if not gene_gene.empty:
        _load_refs(dir_ids, gene_ref_path, gene_rep65_path)
        unique_genes = pd.unique(pd.concat(
            [gene_gene["locus1"], gene_gene["locus2"]], ignore_index=True
        ))
        positions = {gene: genome_gene_pos(gene) for gene in unique_genes}

        def _gene_chr(value):
            pos = positions.get(value)
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                return 0
            return 1 if np.median(pos) <= CHR1_LEN else 2

        gene1_chr = gene_gene["locus1"].map(_gene_chr)
        gene2_chr = gene_gene["locus2"].map(_gene_chr)
        chr1 = pd.concat([
            chr1, gene_gene.loc[(gene1_chr == 1) & (gene2_chr == 1)]
        ], ignore_index=True)
        chr2 = pd.concat([
            chr2, gene_gene.loc[(gene1_chr == 2) & (gene2_chr == 2)]
        ], ignore_index=True)

    chr1_path = f"{prefix}.chr1.GWES_Epi_restored.png"
    chr2_path = f"{prefix}.chr2.GWES_Epi_restored.png"
    from pathlib import Path
    Path(chr1_path).parent.mkdir(parents=True, exist_ok=True)
    Path(chr2_path).parent.mkdir(parents=True, exist_ok=True)
    _plot_gwes_fixed_threshold_panel(
        chr1,
        threshold=float(threshold),
        ld_distance=int(ld_chr1),
        chromosome="chr1",
        output_path=chr1_path,
        dpi=dpi,
        title_prefix=title_prefix,
    )
    _plot_gwes_fixed_threshold_panel(
        chr2,
        threshold=float(threshold),
        ld_distance=int(ld_chr2),
        chromosome="chr2",
        output_path=chr2_path,
        dpi=dpi,
        title_prefix=title_prefix,
    )

    summary = {
        "threshold": float(threshold),
        "chr1_path": chr1_path,
        "chr2_path": chr2_path,
        "chr1_rows": int(len(chr1)),
        "chr2_rows": int(len(chr2)),
        "chr1_strong_rows": int(
            ((chr1["geno_dis"] > ld_chr1) & (chr1["v"] > threshold)).sum()
        ),
        "chr2_strong_rows": int(
            ((chr2["geno_dis"] > ld_chr2) & (chr2["v"] > threshold)).sum()
        ),
    }
    if verbose:
        logging.info(
            "Fixed-threshold restored GWES plots complete: threshold=%.6f, "
            "chr1 strong=%d, chr2 strong=%d",
            float(threshold), summary["chr1_strong_rows"],
            summary["chr2_strong_rows"],
        )
    return summary


def plot_bootstrap_stable_pairs(
    df_input: pd.DataFrame,
    threshold: float,
    prefix: str,
    *,
    stability_cutoff: float = 0.80,
    ld_chr1: int = 5_000,
    ld_chr2: int = 17_500,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    dir_ids: str = DIR_IDS,
    gene_ref_path: str = GENE_REF_PATH,
    gene_rep65_path: str = GENE_REP65_PATH,
    dpi: int = 220,
    verbose: bool = True,
) -> dict[str, object]:
    """Plot the final Bootstrap-stable physical pairs.

    If ``bootstrap_stability`` is present, rows below ``stability_cutoff`` are
    removed before plotting. Otherwise the input is assumed to have
    already been restricted to stable pairs.
    """
    if not isinstance(df_input, pd.DataFrame):
        raise TypeError("df_input must be a pandas DataFrame")
    if not 0.0 <= stability_cutoff <= 1.0:
        raise ValueError("stability_cutoff must be between 0 and 1")
    if "bootstrap_stability" in df_input.columns:
        stable_input = df_input.loc[
            pd.to_numeric(
                df_input["bootstrap_stability"], errors="coerce"
            ) >= float(stability_cutoff)
        ].copy()
    else:
        stable_input = df_input.copy()

    summary = plot_gwes_with_threshold(
        df_input=stable_input,
        threshold=threshold,
        prefix=prefix,
        ld_chr1=ld_chr1,
        ld_chr2=ld_chr2,
        chr1_len=chr1_len,
        chr2_len=chr2_len,
        dir_ids=dir_ids,
        gene_ref_path=gene_ref_path,
        gene_rep65_path=gene_rep65_path,
        dpi=dpi,
        title_prefix="Bootstrap-stable EpiDis_GWES",
        verbose=verbose,
    )
    summary["stability_cutoff"] = float(stability_cutoff)
    summary["stable_input_rows"] = int(len(stable_input))
    return summary

def plot_GWES_high(df_sp_f1, df_sp_h1, LD_dis, flag_chr, save_prefix):
    import seaborn as sns
    from matplotlib import pyplot as plt

    plot_l = 28 if flag_chr == 'chr1' else 16

    df_sp_f = df_sp_f1.copy()
    df_sp_h = df_sp_h1.copy()
    df_sp_f = df_sp_f[df_sp_f['geno_dis'] > 0]
    df_sp_h = df_sp_h[df_sp_h['geno_dis'] > 0]
    LD_dis = int(LD_dis)
    df_sp_PLD_f = df_sp_f[df_sp_f['geno_dis'] <= LD_dis]
    df_sp_LD_f = df_sp_f[df_sp_f['geno_dis'] > LD_dis]
    df_sp_PLD_h = df_sp_h[df_sp_h['geno_dis'] <= LD_dis]
    df_sp_LD_h = df_sp_h[df_sp_h['geno_dis'] > LD_dis]

    sns.set_style("white")
    plt.figure(figsize=(plot_l, 7))
    plt.plot(df_sp_LD_h['geno_dis'].values,df_sp_LD_h['v'].values, '.', alpha=0.6, markersize=0.7, color='limegreen', label='High Order pairs')
    plt.plot(df_sp_LD_f['geno_dis'].values,df_sp_LD_f['v'].values, '.', alpha=0.6, markersize=0.7, color='deepskyblue', label='First Order pairs')
    plt.plot(df_sp_PLD_f['geno_dis'].values,df_sp_PLD_f['v'].values, '.', alpha=0.5, markersize=0.7, color='turquoise', label='PLD pairs')
    plt.plot(df_sp_PLD_h['geno_dis'].values,df_sp_PLD_h['v'].values, '.', alpha=0.5, markersize=0.7, color='turquoise')

    ylim = plt.ylim()
    plt.text(LD_dis, ylim[1], 'P_ld: ' + str(LD_dis), va='bottom', ha='left', color='red')
    plt.axvline(x=LD_dis, c="r", ls="--")

    plt.xlabel('Genome Distance', fontsize=16)
    plt.ylabel('EpiDis Value', fontsize=16)
    plt.title('EpiDis_GWES ' + flag_chr, fontsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(axis='both', which='both', direction='out', length=6, width=2, colors='black', labelsize=12)
    leg = plt.legend(loc='upper right', fontsize=12, markerscale=17)
    leg.get_frame().set_alpha(1)
    save_path = save_prefix + "GWES_Epi_diff.png"
    plt.savefig(save_path, dpi=220)
    plt.close()
    logging.info("Plot saved to %s", save_path)
    return
# -------------------------------- 主函数（DataFrame 输入） ----------------------
def run_gwes_vp(
    df_input: pd.DataFrame,
    prefix: str,
    k: float = 0,
    *,
    dir_ids: str = DIR_IDS,
    gene_ref_path: str = GENE_REF_PATH,
    gene_rep65_path: str = GENE_REP65_PATH,
    chunk_size: int = 10000,
    n_jobs: int = 1,
    make_plots: bool = True,
    save_results: bool = True,
    ld_chr1: int = 5000,
    ld_chr2: int = 17500,
    chr1_len: int = 3288558,
    chr2_len: int = 1877212,
    # ---- 新增 ----
    outlier_mode: str = 'iqr',          # 'iqr' | 'kde'
    kde_margin: float = 0.1,            # KDE: 阈值 = peak + margin
    kde_bw: float | str = 'scott',  # KDE 带宽：'scott'/'silverman'/float
    verbose: bool = True,
    return_threshold: bool = False,
    return_all_pairs: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, float]:
    """
    参数
    ----
    df_input : pd.DataFrame
        至少三列（locus1, locus2, v），列名可任意，将被标准化为上述三列。
    prefix : str
        输出前缀（用于图与最终CSV）。
    k : float
        IQR异常值“级别”（与原脚本一致，传入 1/2/3 等）。
    ld_chr1 / ld_chr2 : int
        chr1 / chr2 的 LD 距离阈值（用于绘图分层与最终过滤）。
    chr1_len / chr2_len : int
        染色体长度（用于分区与距离计算）。注意：本函数将动态覆盖模块级常量。
    return_threshold : bool
        False（默认）时保持旧接口，仅返回结果 DataFrame；True 时返回
        ``(result_df, threshold)``。无论取值如何，阈值也会写入
        ``result_df.attrs['epidis_threshold']``。
    return_all_pairs : bool
        False（默认）时返回通过 GWES 筛选的结果；True 时返回全部输入
        位点对及其 ``geno_dis``、``LD``，但阈值仍按相同 GWES 逻辑计算。
    """
    # 0) 载入参考
    _load_refs(dir_ids, gene_ref_path, gene_rep65_path)
    if not verbose:
        logging.disable(logging.CRITICAL)
        tqdm_disable = True
    else:
        tqdm_disable = False
    # 0.1) 动态覆盖模块全局常量（最小改动做法）
    global CHR1_LEN, CHR2_LEN, GENOME_TOTAL
    CHR1_LEN = int(chr1_len)
    CHR2_LEN = int(chr2_len)
    GENOME_TOTAL = CHR1_LEN + CHR2_LEN
    _gene_pos_cache.clear()
    logging.info(f"Genome configuration set: CHR1_LEN={CHR1_LEN}, CHR2_LEN={CHR2_LEN}, GENOME_TOTAL={GENOME_TOTAL}")
    logging.info(f"LD thresholds: chr1={ld_chr1}, chr2={ld_chr2}, k={k}")

    # 1) 标准化输入
    df0 = _standardize_input_df(df_input)

    # 2) 分块 + 并行
    logging.info("Step 2: Processing input DataFrame in chunks...")
    CHUNK_SIZE = int(chunk_size)
    n = df0.shape[0]
    chunk_starts = range(0, n, CHUNK_SIZE)
    chunk_count = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    logging.info("Total chunks to process: %d", chunk_count)

    if n_jobs <= 0:
        jobs = mp.cpu_count()
    else:
        jobs = n_jobs
    chunks = (df0.iloc[i:i+CHUNK_SIZE].copy() for i in chunk_starts)
    if jobs == 1:
        results = [
            process_chunk(chunk)
            for chunk in tqdm(chunks, total=chunk_count, disable=tqdm_disable)
        ]
    else:
        with mp.get_context("spawn").Pool(
            processes=jobs,
            initializer=_init_worker,
            initargs=(dir_ids, gene_ref_path, gene_rep65_path, CHR1_LEN, CHR2_LEN),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap(process_chunk, chunks),
                    total=chunk_count,
                    disable=tqdm_disable,
                )
            )
    logging.info("All chunks processed.")

    df_all = pd.concat(results, ignore_index=True)
    logging.info("Combined data: %d rows", df_all.shape[0])
    df_sp_fs_gene = df_all.copy()

    # 3) GWES 基础筛选
    logging.info("Step 3: Filtering data with geno_dis >= 0 for GWES analysis...")
    df_gwes = df_sp_fs_gene[df_sp_fs_gene['geno_dis'] >= 0].copy()
    logging.info("GWES data: %d rows", df_gwes.shape[0])

    # 4) gene-gene 计算位点
    logging.info("Step 4: Processing gene-gene combinations for gene positions ...")
    df_gene_gwes = df_gwes[(df_gwes['locus1'] > GENOME_TOTAL) & (df_gwes['locus2'] > GENOME_TOTAL)].copy()
    if not df_gene_gwes.empty:
        # Gene positions have already been cached while calculating distance;
        # map the few unique gene identifiers directly instead of spawning a
        # second process pool and reloading all reference tables.
        df_gene_gwes = process_gene_pos_chunk(df_gene_gwes)
        df_gene_gwes = df_gene_gwes[
            df_gene_gwes['gene1_pos'].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 3) &
            df_gene_gwes['gene2_pos'].apply(lambda x: isinstance(x, (list, tuple)) and len(x) == 3)
        ]
        logging.info("Gene-gene data after processing: %d rows", df_gene_gwes.shape[0])
    else:
        logging.info("No gene-gene data found for gene position processing.")

    # 5) 按染色体区域拆分
    logging.info("Step 5: Splitting GWES data by chromosome region ...")
    df_gwes_chr1 = df_gwes[
        ((df_gwes['locus1'] < CHR1_LEN) & (df_gwes['locus2'] < CHR1_LEN)) |
        ((df_gwes['locus1'] < CHR1_LEN) & (df_gwes['locus2'] > GENOME_TOTAL)) |
        ((df_gwes['locus2'] < CHR1_LEN) & (df_gwes['locus1'] > GENOME_TOTAL))
    ].copy()
    if not df_gene_gwes.empty:
        df_gene_gwes_chr1 = df_gene_gwes[df_gene_gwes['gene1_pos'].apply(lambda x: all(y < CHR1_LEN for y in x))]
        df_gene_gwes_chr1 = df_gene_gwes_chr1.drop(columns=['gene1_pos', 'gene2_pos'])
        df_gwes_chr1 = pd.concat([df_gwes_chr1, df_gene_gwes_chr1], ignore_index=True)
    logging.info("GWES data for chr1: %d rows", df_gwes_chr1.shape[0])

    df_gwes_chr2 = df_gwes[(df_gwes['locus1'] > CHR1_LEN) & (df_gwes['locus2'] > CHR1_LEN)].copy()
    df_gwes_chr2 = df_gwes_chr2[(df_gwes_chr2['locus1'] < GENOME_TOTAL) | (df_gwes_chr2['locus2'] < GENOME_TOTAL)]
    if not df_gene_gwes.empty:
        df_gene_gwes_chr2 = df_gene_gwes[df_gene_gwes['gene1_pos'].apply(lambda x: all(y > CHR1_LEN for y in x))]
        df_gene_gwes_chr2 = df_gene_gwes_chr2.drop(columns=['gene1_pos', 'gene2_pos'])
        df_gwes_chr2 = pd.concat([df_gwes_chr2, df_gene_gwes_chr2], ignore_index=True)
    logging.info("GWES data for chr2: %d rows", df_gwes_chr2.shape[0])

    # 跨染色体/其它
    df_chr12 = df_sp_fs_gene[df_sp_fs_gene['geno_dis'] == -1].copy()
    logging.info("Other (chr12) data: %d rows", df_chr12.shape[0])

    # 6) 绘图（使用可配置 LD 阈值）
    if make_plots:
        logging.info("Step 6: Drawing GWES plots ...")
        if k < -1:
            outliers_all = [0, 0, 0]
        else:
            # 取 LD 远端用于估阈
            pool_df = pd.concat([
                df_gwes_chr1[df_gwes_chr1['geno_dis'] > ld_chr1],
                df_gwes_chr2[df_gwes_chr2['geno_dis'] > ld_chr2]
            ], ignore_index=True)
            if outlier_mode == 'kde':
                thr = find_outliers_kde_upper(pool_df, margin=kde_margin, bw=kde_bw)
                outliers_all = [thr, thr, thr]  # 兼容绘图接口
                logging.info(f"Outlier mode: {outlier_mode}, k={k}, kde_margin={kde_margin}, kde_bw={kde_bw}")
                logging.info("KDE upper threshold: %.4f (margin=%.3f, bw=%s)", thr, kde_margin, str(kde_bw))
            else:
                outliers_all = find_outliers(pool_df,k)
        plot_GWES_PU(df_gwes_chr1, ld_chr1, 'chr1', outliers_all, prefix + '.chr1.', outlier_mode=outlier_mode)
        plot_GWES_PU(df_gwes_chr2, ld_chr2, 'chr2', outliers_all, prefix + '.chr2.', outlier_mode=outlier_mode)
        logging.info("GWES plots saved.")

    # 7) 异常值过滤（使用可配置 LD 阈值）
    logging.info("Step 7: Filtering data based on outlier threshold ...")
    df_epi_chr1 = df_gwes_chr1.copy()
    df_epi_chr2 = df_gwes_chr2.copy()
    df_epi_chr1 = df_epi_chr1[(df_epi_chr1['geno_dis'] > ld_chr1) | (df_epi_chr1['geno_dis'] < 0)]
    df_epi_chr2 = df_epi_chr2[(df_epi_chr2['geno_dis'] > ld_chr2) | (df_epi_chr2['geno_dis'] < 0)]
    df_epi_ots = df_chr12.copy()

    df_epi_chr11 = df_epi_chr1[df_epi_chr1['geno_dis'] > ld_chr1]
    df_epi_chr22 = df_epi_chr2[df_epi_chr2['geno_dis'] > ld_chr2]
    df_epi_LD = pd.concat([df_epi_chr11, df_epi_chr22], ignore_index=True)
    df_epi_LD = df_epi_LD[df_epi_LD['geno_dis'] > 0]
    ext_plus_outlier = (
        find_outliers_k(df_epi_LD, k)
        if outlier_mode != 'kde'
        else find_outliers_kde_upper(df_gwes, margin=kde_margin, bw=kde_bw)
    )
    if outlier_mode == 'kde':
        logging.info("Final outlier threshold: %.4f (mode=KDE, margin=%.3f, bw=%s)", float(ext_plus_outlier), kde_margin, str(kde_bw))
    else:
        logging.info("Final outlier threshold (mode=IQR, k=%d): %.2f", int(k), ext_plus_outlier)

    del df_epi_chr11, df_epi_chr22, df_epi_LD
    gc.collect()

    df_1 = df_epi_chr1[df_epi_chr1['v'] > ext_plus_outlier]
    df_2 = df_epi_chr2[df_epi_chr2['v'] > ext_plus_outlier]
    df_3 = df_epi_ots[df_epi_ots['v'] > ext_plus_outlier]
    df_epi_LD_final = pd.concat([df_1, df_2, df_3], ignore_index=True)
    result_attrs = {
        "epidis_threshold": float(ext_plus_outlier),
        "outlier_mode": outlier_mode,
        "outlier_k": float(k),
        "ld_chr1": int(ld_chr1),
        "ld_chr2": int(ld_chr2),
    }
    df_epi_LD_final.attrs.update(result_attrs)
    df_sp_fs_gene.attrs.update(result_attrs)
    logging.info("After filtering, final dataset contains %d rows.", df_epi_LD_final.shape[0])

    # 8) 保存
    if save_results:
        if outlier_mode == 'kde':
            # 直接使用实际阈值（四舍五入保留三位）
            thr_val = f"{ext_plus_outlier:.2f}".replace('.', '_')
            out_filename = f"{prefix}.pairs.LD.outlier_KDE_thr{thr_val}_Uni.csv"
        else:
            out_filename = f"{prefix}.pairs.LD.outlier_{int(k)}_Uni.csv"

        logging.info("Step 8: Saving final filtered data to: %s", out_filename)
        df_epi_LD_final.to_csv(out_filename, index=False)
    gc.collect()

    logging.info("All steps DONE.")
    if not verbose:
        logging.disable(logging.NOTSET)
    result_df = df_sp_fs_gene if return_all_pairs else df_epi_LD_final
    if return_threshold:
        return result_df, float(ext_plus_outlier)
    return result_df


def run_gwes_vp_full(
    df_input: pd.DataFrame,
    k: float = 0,
    *,
    dir_ids: str = DIR_IDS,
    gene_ref_path: str = GENE_REF_PATH,
    gene_rep65_path: str = GENE_REP65_PATH,
    chunk_size: int = 10_000,
    n_jobs: int = 1,
    ld_chr1: int = 5_000,
    ld_chr2: int = 17_500,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    outlier_mode: str = 'iqr',
    kde_margin: float = 0.1,
    kde_bw: float | str = 'scott',
    verbose: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Run the GWES calculations and return all pairs plus the threshold.

    This is the no-file-output variant of :func:`run_gwes_vp`. It performs the
    same distance annotation, chromosome/LD background construction, and
    IQR/KDE threshold calculation, but disables both plot and table writing.

    Returns
    -------
    df_all, threshold
        ``df_all`` contains every standardized input pair with calculated
        ``geno_dis`` and ``LD``. It is not filtered by the returned threshold.
    """
    result = run_gwes_vp(
        df_input=df_input,
        prefix="",
        k=k,
        dir_ids=dir_ids,
        gene_ref_path=gene_ref_path,
        gene_rep65_path=gene_rep65_path,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        make_plots=False,
        save_results=False,
        ld_chr1=ld_chr1,
        ld_chr2=ld_chr2,
        chr1_len=chr1_len,
        chr2_len=chr2_len,
        outlier_mode=outlier_mode,
        kde_margin=kde_margin,
        kde_bw=kde_bw,
        verbose=verbose,
        return_threshold=True,
        return_all_pairs=True,
    )
    return result


def run_gwes_vp_bootstrap(
    data_df: pd.DataFrame,
    weight_ser: pd.Series | pd.DataFrame,
    df_pairs: pd.DataFrame,
    *,
    k: int = 2,
    n_bootstrap: int = 100,
    stability_cutoff: float = 0.80,
    random_state: Optional[int] = 20260816,
    pair_calculator: Optional[Callable] = None,
    pair_n_jobs: int = 1,
    gwes_n_jobs: int = 1,
    dir_ids: str = DIR_IDS,
    gene_ref_path: str = GENE_REF_PATH,
    gene_rep65_path: str = GENE_REP65_PATH,
    chunk_size: int = 10_000,
    ld_chr1: int = 5_000,
    ld_chr2: int = 17_500,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    verbose: bool = True,
) -> dict[str, object]:
    """Select k=2 GWES candidates and retain Bootstrap-stable pairs.

    No CSV, Parquet, or plot files are created. ``data_df`` should be the
    unique-pattern matrix (for example ``snp_pair``), while ``df_pairs`` is its
    original representative-pair EpiDis table.
    """
    if int(k) < 1:
        raise ValueError("k must be at least 1; k=2 is recommended")
    candidate_pairs, threshold = run_gwes_vp(
        df_input=df_pairs,
        prefix="",
        k=int(k),
        dir_ids=dir_ids,
        gene_ref_path=gene_ref_path,
        gene_rep65_path=gene_rep65_path,
        chunk_size=chunk_size,
        n_jobs=gwes_n_jobs,
        make_plots=False,
        save_results=False,
        ld_chr1=ld_chr1,
        ld_chr2=ld_chr2,
        chr1_len=chr1_len,
        chr2_len=chr2_len,
        outlier_mode="iqr",
        verbose=verbose,
        return_threshold=True,
    )
    stable_pairs, all_candidates = bootstrap_pair_stability(
        data_df=data_df,
        weight_ser=weight_ser,
        pairs=candidate_pairs,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        stability_cutoff=stability_cutoff,
        random_state=random_state,
        pair_calculator=pair_calculator,
        n_jobs=pair_n_jobs,
        verbose=verbose,
    )
    return {
        "threshold": float(threshold),
        "k": int(k),
        "n_bootstrap": int(n_bootstrap),
        "stability_cutoff": float(stability_cutoff),
        "candidate_pairs": all_candidates,
        "stable_pairs": stable_pairs,
    }

def run_gwes_vp_high(
    df_f_input: pd.DataFrame,
    df_h_input: pd.DataFrame,
    prefix: str,
    ld_chr1: int = 5000,
    ld_chr2: int = 17500,
    chr1_len: int = 3288558,
    chr2_len: int = 1877212,
) -> dict[str, pd.DataFrame]:
    global CHR1_LEN, CHR2_LEN, GENOME_TOTAL
    CHR1_LEN = int(chr1_len)
    CHR2_LEN = int(chr2_len)
    GENOME_TOTAL = CHR1_LEN + CHR2_LEN
    logging.info(f"Genome configuration set: CHR1_LEN={CHR1_LEN}, CHR2_LEN={CHR2_LEN}, GENOME_TOTAL={GENOME_TOTAL}")

    # 1) 标准化输入
    logging.info("Step 1: Processing input DataFrames...")
    df_f = _standardize_input_df(df_f_input)
    df_h = _standardize_input_df(df_h_input)

    logging.info("Step 2: Detecting genome distance for GWES analysis...")
    df_f = run_gwes_vp(
        df_input=df_f,prefix='',k=0,n_jobs=-1,make_plots=False,save_results=False,
        ld_chr1=0,ld_chr2=0,chr1_len=chr1_len,chr2_len=chr2_len
    )

    df_h = run_gwes_vp(
        df_input=df_h,prefix='',k=0,n_jobs=-1,make_plots=False,save_results=False,
        ld_chr1=0,ld_chr2=0,chr1_len=chr1_len,chr2_len=chr2_len
    )
    gc.collect()


    logging.info("Step 3: Splitting GWES data by chromosome region ...")
    def split_gwes(df_gwes,chr1_len=CHR1_LEN,chr2_len=CHR2_LEN):
        df_gwes_chr1 = df_gwes[
            ((df_gwes['locus1'] < chr1_len) & (df_gwes['geno_dis'] > 0)) |
            ((df_gwes['locus2'] < chr1_len) & (df_gwes['geno_dis'] > 0))
            ].copy()
        df_gwes_chr2 = df_gwes[(df_gwes['locus1'] > chr1_len) & (df_gwes['locus2'] > chr1_len)].copy()
        df_gwes_chr2 = df_gwes_chr2[(df_gwes_chr2['locus1'] < (chr1_len + chr2_len)) | (df_gwes_chr2['locus2'] < (chr1_len + chr2_len))]
        df_gwes_chr2 = df_gwes_chr2[
            ((df_gwes_chr2['locus1'] < (chr1_len + chr2_len)) & (df_gwes_chr2['geno_dis'] > 0)) |
            ((df_gwes_chr2['locus2'] < (chr1_len + chr2_len)) & (df_gwes_chr2['geno_dis'] > 0))
            ].copy()
        return df_gwes_chr1, df_gwes_chr2

    df_f_chr1, df_f_chr2 = split_gwes(df_f,chr1_len=CHR1_LEN,chr2_len=CHR2_LEN)
    logging.info("First GWES data for chr1: %d rows", df_f_chr1.shape[0])
    logging.info("First GWES data for chr2: %d rows", df_f_chr2.shape[0])
    df_h_chr1, df_h_chr2 = split_gwes(df_h,chr1_len=CHR1_LEN,chr2_len=CHR2_LEN)
    logging.info("High GWES data for chr1: %d rows", df_h_chr1.shape[0])
    logging.info("High GWES data for chr2: %d rows", df_h_chr2.shape[0])


    logging.info("Step 4: Drawing GWES diff plots ...")
    plot_GWES_high(df_f_chr1, df_h_chr1, ld_chr1, 'chr1', prefix + '.chr1.')
    plot_GWES_high(df_f_chr2, df_h_chr2, ld_chr2, 'chr2', prefix + '.chr2.')
    logging.info("GWES diff plots saved.")
    return {
        "f_chr1": df_f_chr1,
        "f_chr2": df_f_chr2,
        "h_chr1": df_h_chr1,
        "h_chr2": df_h_chr2,
    }

def merge_pairs_with_fill(
    df1,
    df2,
    fill_value: float = 0.0,
    return_pandas: bool = True,
):
    """
    合并两张 pair 表（locus1, locus2, v[, geno_dis]），对无序对进行标准化后全外连接。
    - 输出列：locus1, locus2, v1, v2, geno_dis
    - v1/v2 缺失 -> 用 fill_value 填充
    - geno_dis（可选）：
        * 仅一侧存在 -> 保留该值，缺失填 -1
        * 两侧都存在 -> 取较大值
        * 两侧都不存在 -> 输出 -1
      注：输入中的 -1 视为“缺失”，合并时不参与 max，比完后再把缺失填回 -1。
    """
    import polars as pl

    # 转成 Polars DataFrame
    if not isinstance(df1, pl.DataFrame):
        df1 = pl.from_pandas(df1)
    if not isinstance(df2, pl.DataFrame):
        df2 = pl.from_pandas(df2)

    def normalize(df: pl.DataFrame, vcol: str) -> pl.DataFrame:
        # 最小改动：若存在 geno_dis，则一并保留
        keep_cols = ["locus1", "locus2", vcol]
        if "geno_dis" in df.columns:
            keep_cols.append("geno_dis")
        # Parquet/object inputs can represent numeric locus IDs as strings.
        # Cast both join sides before min/max and before the full join so a
        # cached full-data table and a newly calculated subgroup table always
        # use identical key dtypes.
        df = df.with_columns([
            pl.col("locus1").cast(pl.Int64, strict=True),
            pl.col("locus2").cast(pl.Int64, strict=True),
        ])
        return (
            df.with_columns([
                pl.min_horizontal("locus1", "locus2").alias("locus1"),
                pl.max_horizontal("locus1", "locus2").alias("locus2"),
                pl.col(vcol),
            ])
            .select(keep_cols)
        )

    # 标准化 + 重命名 v 列
    df1n = normalize(df1, "v").rename({"v": "v1"})
    df2n = normalize(df2, "v").rename({"v": "v2"})

    # 全外连接；右表同名列会带 _right
    merged = df1n.join(df2n, on=["locus1", "locus2"], how="full", suffix="_right")

    # 稳妥处理（极少数情况下 join 键可能出现 _right）
    cols = merged.columns
    l1_candidates = [c for c in ("locus1", "locus1_right") if c in cols]
    l2_candidates = [c for c in ("locus2", "locus2_right") if c in cols]
    merged = merged.with_columns([
        pl.coalesce([pl.col(c) for c in l1_candidates]).alias("_l1"),
        pl.coalesce([pl.col(c) for c in l2_candidates]).alias("_l2"),
    ])
    merged = merged.drop([c for c in ("locus1", "locus1_right", "locus2", "locus2_right") if c in cols]) \
                   .rename({"_l1": "locus1", "_l2": "locus2"})

    # v1/v2 -> float 并填充
    if "v1" not in merged.columns:
        merged = merged.with_columns(pl.lit(None).alias("v1"))
    if "v2" not in merged.columns:
        merged = merged.with_columns(pl.lit(None).alias("v2"))
    merged = merged.with_columns([
        pl.col("v1").cast(pl.Float64).fill_null(fill_value),
        pl.col("v2").cast(pl.Float64).fill_null(fill_value),
    ])

    # ===== 合并 geno_dis（最小改动）=====
    cols = merged.columns  # 刷新列名
    has_left = "geno_dis" in cols
    has_right = "geno_dis_right" in cols

    if has_left or has_right:
        # 缺失侧补 None
        if not has_left:
            merged = merged.with_columns(pl.lit(None).alias("geno_dis"))
        if not has_right:
            merged = merged.with_columns(pl.lit(None).alias("geno_dis_right"))
        # 将 -1 视为缺失 -> None，再做横向最大；最终缺失填回 -1
        merged = merged.with_columns([
            pl.max_horizontal([
                pl.when(pl.col("geno_dis") == -1).then(None).otherwise(pl.col("geno_dis")).cast(pl.Int64),
                pl.when(pl.col("geno_dis_right") == -1).then(None).otherwise(pl.col("geno_dis_right")).cast(pl.Int64),
            ]).fill_null(-1).alias("_gd")
        ])
        merged = merged.drop([c for c in ("geno_dis", "geno_dis_right") if c in merged.columns]) \
                       .rename({"_gd": "geno_dis"})
    else:
        # 两边都没有 -> 统一给 -1
        merged = merged.with_columns(pl.lit(-1).cast(pl.Int64).alias("geno_dis"))

    # 最终列与类型
    merged = merged.select([
        pl.col("locus1").cast(pl.Int64),
        pl.col("locus2").cast(pl.Int64),
        pl.col("v1").cast(pl.Float64),
        pl.col("v2").cast(pl.Float64),
        pl.col("geno_dis").cast(pl.Int64),
    ])

    return merged.to_pandas() if return_pandas else merged
