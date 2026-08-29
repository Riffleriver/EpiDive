#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block-matrix CPU implementation of weighted binary EpiDis.

This module keeps the public calling style of ``epidis_cpu`` while replacing
the per-pair sample loop with blocked matrix multiplication.  For a block of
source loci, one BLAS multiplication obtains the weighted joint-one count for
all source-target pairs.  The original EpiDis/JSD formula is then applied to
the entire score block with Numba.

Large runs can write each result block directly into one Parquet file::

    import epidis_cpu_fast as ecf

    ecf.run_epidis_normal_cpu(
        data_df=snp_pair,
        weight_ser=weight_ser,
        threshold=0.3,
        n_jobs=8,              # BLAS threads, not worker processes
        block_size=128,
        prefix="vp10k",
        out_format="parquet",
        stream_output=True,
        return_df=False,
        accumulation_dtype="float64",  # legacy-compatible default
    )
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union
import logging
import os
import time

import numpy as np
import pandas as pd
from joblib import cpu_count
from numba import config as numba_config
from numba import njit, prange, set_num_threads
from tqdm import tqdm

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional performance dependency
    threadpool_limits = None


__all__ = [
    "convert_snp_to_binary",
    "estimate_adaptive_epidis_threshold",
    "run_epidis_normal_cpu",
    "run_epidis_normal_cpu_fast",
    "run_from_files",
]


def estimate_adaptive_epidis_threshold(
    data_df: pd.DataFrame,
    weight_ser: Union[pd.Series, pd.DataFrame],
    *,
    target_max_pairs: int = 5_000_000,
    sample_pairs: int = 50_000,
    min_threshold: float = 0.3,
    safety_factor: float = 0.60,
    candidate_thresholds: Optional[list[float]] = None,
    random_state: int = 2026,
    n_jobs: int = 1,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
) -> dict[str, object]:
    """Estimate an output-limiting EpiDis threshold from random locus pairs."""
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame")
    if data_df.empty:
        raise ValueError("data_df cannot be empty")
    if not data_df.index.is_unique:
        raise ValueError("data_df index must contain unique locus identifiers")
    if target_max_pairs < 1:
        raise ValueError("target_max_pairs must be positive")
    if sample_pairs < 1_000:
        raise ValueError("sample_pairs must be at least 1000")
    if min_threshold < 0:
        raise ValueError("min_threshold cannot be negative")
    if not 0 < safety_factor <= 1:
        raise ValueError("safety_factor must be in (0, 1]")

    lg = logger or get_logger()
    locus_count = len(data_df)
    if locus_count < 2:
        raise ValueError("At least two loci are required")
    total_pairs = locus_count * (locus_count - 1) // 2
    requested_sample_size = min(int(sample_pairs), int(total_pairs))

    rng = np.random.default_rng(random_state)
    sampled_keys = np.empty(0, dtype=np.int64)
    while sampled_keys.size < requested_sample_size:
        remaining = requested_sample_size - sampled_keys.size
        draw_size = max(2_000, int(remaining * 1.30))
        row1 = rng.integers(0, locus_count, size=draw_size, dtype=np.int64)
        row2 = rng.integers(0, locus_count, size=draw_size, dtype=np.int64)
        valid = row1 != row2
        lower = np.minimum(row1[valid], row2[valid])
        upper = np.maximum(row1[valid], row2[valid])
        new_keys = lower * np.int64(locus_count) + upper
        sampled_keys = np.unique(np.concatenate([sampled_keys, new_keys]))
    sampled_keys = sampled_keys[:requested_sample_size]

    sampled_row1 = sampled_keys // locus_count
    sampled_row2 = sampled_keys % locus_count
    locus_index = data_df.index.to_numpy()
    sampled_pair_df = pd.DataFrame(
        {
            "locus1": locus_index[sampled_row1],
            "locus2": locus_index[sampled_row2],
        }
    )

    try:
        from epidis_cpu_test import calculate_complete_pair_scores
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "epidis_cpu_test.calculate_complete_pair_scores is required"
        ) from exc

    sampled_scores = calculate_complete_pair_scores(
        data_df=data_df,
        weight_ser=weight_ser,
        pairs=sampled_pair_df,
        label="Adaptive threshold sample",
        n_jobs=n_jobs,
        logger=lg,
    )
    values = sampled_scores["v"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The sampled EpiDis scores contain no finite values")

    target_retention = min(
        1.0,
        float(target_max_pairs) * float(safety_factor) / float(total_pairs),
    )
    unique_values, counts = np.unique(values, return_counts=True)
    survival_counts = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
    survival_fractions = survival_counts / float(values.size)
    eligible = np.flatnonzero(survival_fractions <= target_retention)
    if eligible.size:
        adaptive_threshold = float(unique_values[eligible[0]])
    else:
        adaptive_threshold = float(np.nextafter(unique_values[-1], np.inf))
    adaptive_threshold = max(float(min_threshold), adaptive_threshold)

    retained_fraction = float(np.mean(values >= adaptive_threshold))
    estimated_output_pairs = int(np.ceil(total_pairs * retained_fraction))

    if candidate_thresholds is None:
        candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = np.asarray(candidate_thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("candidate_thresholds must be a non-empty sequence")
    if not np.isfinite(thresholds).all() or (thresholds < 0).any():
        raise ValueError("candidate_thresholds must be finite and non-negative")

    threshold_summary = pd.DataFrame({"threshold": thresholds})
    threshold_summary["sample_retention"] = [
        float(np.mean(values >= threshold)) for threshold in thresholds
    ]
    threshold_summary["estimated_pairs"] = np.ceil(
        threshold_summary["sample_retention"].to_numpy() * total_pairs
    ).astype(np.int64)

    if verbose:
        print(f"Unique patterns: {locus_count:,}")
        print(f"Total possible pairs: {total_pairs:,}")
        print(f"Sampled pairs: {values.size:,}")
        print(f"Target maximum pairs: {target_max_pairs:,}")
        print(f"Safety factor: {safety_factor:.2f}")
        print(f"Adaptive threshold: {adaptive_threshold:.6f}")
        print(f"Estimated output pairs: {estimated_output_pairs:,}")
        print(threshold_summary.to_string(index=False))

    return {
        "threshold": float(adaptive_threshold),
        "total_pairs": int(total_pairs),
        "sample_pairs": int(values.size),
        "estimated_output_pairs": int(estimated_output_pairs),
        "estimated_retention": retained_fraction,
        "target_retention": float(target_retention),
        "threshold_summary": threshold_summary,
        "sampled_scores": sampled_scores,
    }


def get_logger(name: str = "EpiDisCPUFast", verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def _is_numeric_binary(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    values = df.to_numpy(copy=False)
    if not np.issubdtype(values.dtype, np.number):
        return False
    return bool(np.isfinite(values).all() and np.all((values == 0) | (values == 1)))


def convert_snp_to_binary(
    df_full: pd.DataFrame,
    *,
    n_jobs: int = -1,
    convert_chunksize: int = 20_000,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Convert loci-by-samples alleles to major-allele indicator 0/1.

    Numeric binary input takes a fast path.  ``n_jobs`` and
    ``convert_chunksize`` are accepted for compatibility; conversion itself
    is vectorized per row and does not start another process pool.
    """
    del n_jobs, convert_chunksize
    lg = logger or get_logger()
    if not isinstance(df_full, pd.DataFrame):
        raise TypeError("df_full must be a pandas DataFrame")
    if df_full.empty:
        raise ValueError("df_full cannot be empty")
    if _is_numeric_binary(df_full):
        lg.info("Detected numeric-binary matrix -> fast path")
        return pd.DataFrame(
            df_full.to_numpy(dtype=np.float32, copy=False),
            index=df_full.index,
            columns=df_full.columns,
        )

    lg.info("Converting nucleotide/categorical rows to major-allele 0/1")
    converted = df_full.apply(
        lambda row: (row == row.value_counts().idxmax()).astype(np.float32),
        axis=1,
        result_type="expand",
    )
    converted.index = df_full.index
    converted.columns = df_full.columns
    return converted.astype(np.float32, copy=False)


def _align_weights(
    weight_ser: Union[pd.Series, pd.DataFrame],
    sample_columns: pd.Index,
) -> np.ndarray:
    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError("weight_ser DataFrame must have exactly one column")
        weight_ser = weight_ser.iloc[:, 0]
    if not isinstance(weight_ser, pd.Series):
        raise TypeError("weight_ser must be a Series or one-column DataFrame")
    missing = sample_columns.difference(weight_ser.index).tolist()
    if missing:
        raise ValueError(
            f"weight_ser is missing {len(missing)} samples; first missing: {missing[:10]}"
        )
    weights = weight_ser.reindex(sample_columns).to_numpy(dtype=np.float32)
    if not np.isfinite(weights).all():
        raise ValueError("weight_ser contains NaN or infinite values")
    if np.any(weights < 0):
        raise ValueError("sample weights cannot be negative")
    if float(weights.sum(dtype=np.float64)) <= 0:
        raise ValueError("sample weights must have a positive sum")
    return np.ascontiguousarray(weights)


# Some external/network-style volumes do not expose a stable locator to
# Numba's disk cache.  Disabling disk caching keeps this single-file module
# importable beside user scripts; compilation still occurs only once per
# Python process.
@njit(parallel=True, fastmath=True, cache=False)
def _score_joint_block(
    joint11,
    source_dp,
    source_dq,
    target_dp,
):
    """Apply the legacy formula to a block and its right-side targets.

    Target column zero represents global locus ``block_start + 1``.  Thus row
    ``local_i`` begins at target column ``local_i`` to preserve the strict
    upper triangle without multiplying against already processed loci.
    """
    block_rows, target_count = joint11.shape
    scores = np.full((block_rows, target_count), -1.0, dtype=np.float32)
    for local_i in prange(block_rows):
        dp = source_dp[local_i]
        dq = source_dq[local_i]
        if dp <= 0.0 or dq <= 0.0:
            continue
        size_p = dp / (dp + dq)
        size_q = 1.0 - size_p
        for target_local in range(local_i, target_count):
            data_p = joint11[local_i, target_local]
            data_q = target_dp[target_local] - data_p
            p1 = max(min(data_p / dp, 1.0 - 1e-9), 1e-9)
            p0 = 1.0 - p1
            q1 = max(min(data_q / dq, 1.0 - 1e-9), 1e-9)
            q0 = 1.0 - q1
            st1_0 = p1 + 1e-27
            st1_1 = p0 + 1e-27
            st2_0 = q1 + 1e-27
            st2_1 = q0 + 1e-27
            st3_0 = st1_0 * size_p + st2_0 * size_q
            st3_1 = st1_1 * size_p + st2_1 * size_q
            r1_0 = st1_0 / st3_0
            r1_1 = st1_1 / st3_1
            r2_0 = st2_0 / st3_0
            r2_1 = st2_1 / st3_1
            js = (
                (st1_0 * np.log2(r1_0) + st1_1 * np.log2(r1_1)) * size_p
                + (st2_0 * np.log2(r2_0) + st2_1 * np.log2(r2_1)) * size_q
            )
            scores[local_i, target_local] = np.sqrt(js) if js > 0.0 else 0.0
    return scores


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "locus1": pd.Series(dtype=np.int64),
            "locus2": pd.Series(dtype=np.int64),
            "v": pd.Series(dtype=np.float32),
        }
    )


class _StreamingWriter:
    def __init__(self, output_path: str, out_format: str):
        self.output_path = output_path
        self.out_format = out_format
        self.writer = None
        self.wrote_txt = False
        if out_format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover
                raise ImportError("streamed Parquet output requires pyarrow") from exc
            self.pa = pa
            self.pq = pq

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        if self.out_format == "parquet":
            table = self.pa.Table.from_pandas(frame, preserve_index=False)
            if self.writer is None:
                self.writer = self.pq.ParquetWriter(self.output_path, table.schema)
            self.writer.write_table(table)
        else:
            frame.to_csv(
                self.output_path,
                sep="\t",
                index=False,
                header=False,
                mode="a" if self.wrote_txt else "w",
            )
            self.wrote_txt = True

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        elif self.out_format == "parquet":
            _empty_result().to_parquet(self.output_path, index=False)
        elif not self.wrote_txt:
            Path(self.output_path).touch()


def run_epidis_normal_cpu(
    data_df: pd.DataFrame,
    weight_ser: Union[pd.Series, pd.DataFrame],
    *,
    threshold: float = 0.2,
    n_jobs: int = 1,
    slices_per_job: int = 8,
    block_size: int = 128,
    use_memmap: bool = False,
    memmap_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    out_format: str = "txt",
    stream_output: bool = False,
    return_df: bool = True,
    accumulation_dtype: str = "float64",
    show_progress: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Compute weighted binary EpiDis with blocked matrix multiplication.

    ``n_jobs`` controls BLAS threads; no joblib worker processes are created.
    ``slices_per_job``, ``use_memmap`` and ``memmap_dir`` are accepted for
    calling compatibility.  Blocking already bounds temporary matrix memory.

    ``accumulation_dtype='float64'`` is the compatibility default and closely
    reproduces the legacy sequential accumulator.  ``'float32'`` is faster and
    uses less memory, but pairs whose values lie within roughly 1e-6 of the
    threshold can occasionally fall on the opposite side of that threshold.
    """
    del slices_per_job, use_memmap, memmap_dir
    lg = logger or get_logger()
    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame")
    if data_df.empty:
        raise ValueError("data_df cannot be empty")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if out_format not in {"txt", "parquet"}:
        raise ValueError("out_format must be 'txt' or 'parquet'")
    if stream_output and prefix is None:
        raise ValueError("prefix is required when stream_output=True")
    if not return_df and not stream_output:
        raise ValueError("return_df=False requires stream_output=True")
    if accumulation_dtype not in {"float32", "float64"}:
        raise ValueError("accumulation_dtype must be 'float32' or 'float64'")
    if n_jobs <= 0:
        n_jobs = cpu_count()
    n_jobs = max(1, int(n_jobs))
    numba_threads = min(n_jobs, int(numba_config.NUMBA_NUM_THREADS))
    set_num_threads(numba_threads)

    if not _is_numeric_binary(data_df):
        lg.info("Input is not numeric binary -> converting inside fast runner")
        data_df = convert_snp_to_binary(data_df, logger=lg)
    weights = _align_weights(weight_ser, data_df.columns)
    try:
        index_arr = np.ascontiguousarray(
            data_df.index.to_numpy(dtype=np.int64, copy=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("data_df index must be convertible to int64") from exc

    dtype = np.float32 if accumulation_dtype == "float32" else np.float64
    x = np.ascontiguousarray(data_df.to_numpy(dtype=dtype, copy=False))
    w = np.ascontiguousarray(weights, dtype=dtype)
    weighted_x = np.ascontiguousarray(x * w.reshape(1, -1), dtype=dtype)
    # Float64 reductions reproduce the legacy accumulator closely even when
    # SGEMM is selected for the dominant joint-count multiplication.
    target_dp = weighted_x.sum(axis=1, dtype=np.float64)
    target_dq = ((1.0 - x) * w.reshape(1, -1)).sum(axis=1, dtype=np.float64)

    total_loci, sample_count = x.shape
    temporary_mb = (
        block_size
        * total_loci
        * (np.dtype(dtype).itemsize + np.dtype(np.float32).itemsize + 1)
        / (1024**2)
    )
    lg.info(
        "EpiDis CPU fast | T=%d, N=%d, BLAS threads=%d, Numba threads=%d, block=%d, "
        "threshold=%.3f, accumulation=%s, estimated block temporaries=%.1f MiB",
        total_loci,
        sample_count,
        n_jobs,
        numba_threads,
        block_size,
        threshold,
        accumulation_dtype,
        temporary_mb,
    )

    output_path = None
    writer = None
    if prefix:
        output_path = (
            f"{prefix}_Epi_pairs.parquet"
            if out_format == "parquet"
            else f"{prefix}_Epi_pairs.txt"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if stream_output:
        writer = _StreamingWriter(output_path, out_format)

    frames = []
    result_count = 0
    start_time = time.time()
    limiter = (
        threadpool_limits(limits=n_jobs, user_api="blas")
        if threadpool_limits is not None
        else nullcontext()
    )
    block_starts = range(0, total_loci, block_size)
    iterator = tqdm(
        block_starts,
        total=(total_loci + block_size - 1) // block_size,
        desc="EpiDis matrix blocks",
        unit="blocks",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        with limiter:
            for start in iterator:
                stop = min(start + block_size, total_loci)
                target_start = start + 1
                if target_start >= total_loci:
                    continue
                # Only multiply against the current block's right side.  The
                # scoring kernel masks the small lower triangle inside the block.
                joint11 = np.ascontiguousarray(
                    x[start:stop] @ weighted_x[target_start:].T
                )
                scores = _score_joint_block(
                    joint11,
                    target_dp[start:stop],
                    target_dq[start:stop],
                    target_dp[target_start:],
                )
                local_i, target_local = np.nonzero(scores >= float(threshold))
                if local_i.size == 0:
                    continue
                source_i = start + local_i
                target_j = target_start + target_local
                block_result = pd.DataFrame(
                    {
                        "locus1": index_arr[source_i],
                        "locus2": index_arr[target_j],
                        "v": scores[local_i, target_local].astype(
                            np.float32, copy=False
                        ),
                    }
                )
                result_count += len(block_result)
                if writer is not None:
                    writer.write(block_result)
                if return_df:
                    frames.append(block_result)
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - start_time
    lg.info("EpiDis fast done in %.3f s; retained %d pairs", elapsed, result_count)
    if return_df:
        result = pd.concat(frames, ignore_index=True) if frames else _empty_result()
    else:
        result = _empty_result()
        result.attrs["rows_written"] = result_count
    if prefix and not stream_output:
        if out_format == "parquet":
            result.to_parquet(output_path, index=False)
        else:
            result.to_csv(output_path, sep="\t", index=False, header=False)
    if output_path:
        result.attrs["output_path"] = output_path
        lg.info("Results saved to %s", output_path)
    return result


run_epidis_normal_cpu_fast = run_epidis_normal_cpu


def run_from_files(
    snp_path: str,
    weight_path: str,
    **kwargs,
) -> pd.DataFrame:
    """Load SNP/weight files and call :func:`run_epidis_normal_cpu`."""
    extension = os.path.splitext(snp_path)[1].lower()
    data_df = (
        pd.read_parquet(snp_path)
        if extension == ".parquet"
        else pd.read_csv(snp_path, sep="\t", index_col=0)
    )
    weights_df = pd.read_csv(weight_path, index_col=0)
    weight_ser = (
        weights_df["wet"] if "wet" in weights_df.columns else weights_df.iloc[:, 0]
    )
    return run_epidis_normal_cpu(data_df, weight_ser, **kwargs)
