#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 13:01:16 2026

@author: riffleriver

CUDA/CuPy delayed-restoration version (2026-08-17):
- Multi-GPU CuPy computes representative-pattern all-pairs EpiDis.
- A reusable CuPy scorer handles missing and restored selected pairs on one GPU.
- VP analysis remains at representative-pattern level until high-order selection.
- Final restoration inherits invariant scores and recalculates only required scores.
- GWES, circular distances, Bootstrap thresholds, and plotting match the CPU version.
- The executable workflow is protected by main() for multiprocessing spawn safety.

GPU server package 1.0.14 (2026-08-21):
- Require both MAF and configurable per-locus MAC (default 5).
- Include MAC in cache and resume signatures.

GPU server package 1.0.13 (2026-08-20):
- Use memory-safe tiled CuPy SGEMM for SCC/HCC sample similarity.
- Auto-fallback to the validated CPU BLAS path when CUDA is unavailable.
- Preserve the 1.0.12 preprocessing, SCC, and downstream result semantics.

GPU server package 1.0.12 (2026-08-20):
- Cache categorical alleles once as uint8 and vectorize subgroup conversion.
- Preserve per-subgroup major-allele and legacy tie-breaking semantics.
- Pair the GPU workflow with an explicitly validated BLAS backend for SCC.
- Log SCC matrix runtime and estimated throughput.
"""
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import gc
import os
import logging
import csv
import hashlib
import json
import time
import traceback

import epidis_cpu as ec
import epidis_cpu_test as ect
import epidis_cpu_fast as ecf
import epidis_reweight as er
import epidis_gwes as eg
import epidis_gwes_high as egh
import epidis_gwes_vp as eg_vp
import epidis_filter as ef


def _configure_console_output(verbose=False):
    """Suppress INFO/DEBUG logging unless detailed progress is requested."""
    logging.disable(logging.NOTSET if verbose else logging.INFO)


def _load_gpu_backend():
    """Import CuPy and the project GPU runner only when GPU work starts."""
    try:
        import cupy as cp
        import epidis_gpu as egu
    except ImportError as exc:
        raise ImportError(
            "The GPU pipeline requires CuPy and epidis_gpu in the active "
            "CUDA environment"
        ) from exc
    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count < 1:
        raise RuntimeError("No CUDA GPU is available")
    return cp, egu, device_count
# -----------------------------------------------------------------------------
def transform_row(row):
    """
    Transform a single row by converting the most frequent element to 1 and the rest to 0.
    """
    value_counts = row.value_counts()
    max_value = value_counts.idxmax()
    return (row == max_value).astype(int)

def extra_loc(df_sp):
    """Extract and sort unique locus labels from both 'locus1' and 'locus2' columns."""
    t1 = np.unique(df_sp['locus1'])
    t2 = np.unique(df_sp['locus2'])
    sp_loc = np.unique(np.hstack((t1, t2)))
    return np.sort(sp_loc.astype(str))

def split_gwes(df_gwes,chr1_len=3288558,chr2_len=1877212):
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


def run_highorder_threshold_analysis(
    df_merged,
    prefix,
    *,
    k=2.0,
    min_abs_threshold=0.15,
    save_filtered=True,
    verbose=True,
):
    """Estimate global IQR thresholds, filter pairs, and plot both chromosomes."""
    required = {"locus1", "locus2", "v_f", "v_h", "geno_dis"}
    missing = sorted(required.difference(df_merged.columns))
    if missing:
        raise ValueError(f"df_merged is missing columns: {missing}")
    if k < 0:
        raise ValueError("k cannot be negative")
    if min_abs_threshold < 0:
        raise ValueError("min_abs_threshold cannot be negative")

    result_df = df_merged.copy()
    result_df["v_f"] = pd.to_numeric(result_df["v_f"], errors="coerce")
    result_df["v_h"] = pd.to_numeric(result_df["v_h"], errors="coerce")
    result_df["diff"] = result_df["v_h"] - result_df["v_f"]

    finite_diff = result_df.loc[
        np.isfinite(result_df["diff"]),
        "diff",
    ]
    if finite_diff.empty:
        raise ValueError("df_merged contains no finite EpiDis differences")

    q1 = float(finite_diff.quantile(0.25))
    q3 = float(finite_diff.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError("The EpiDis difference IQR must be positive")

    fence_width = 1.5 * float(k) * iqr
    upper_threshold = max(q3 + fence_width, float(min_abs_threshold))
    lower_threshold = min(q1 - fence_width, -float(min_abs_threshold))

    strengthen_mask = result_df["diff"] > upper_threshold
    weaken_mask = result_df["diff"] < lower_threshold
    selected_mask = strengthen_mask | weaken_mask

    df_strengthen = result_df.loc[strengthen_mask].copy()
    df_weaken = result_df.loc[weaken_mask].copy()
    df_filtered = result_df.loc[selected_mask].copy()

    df_chr1, df_chr2 = split_gwes(result_df)
    df_filtered_chr1, df_filtered_chr2 = split_gwes(df_filtered)

    plot_chr1 = egh.gwes_high_from_df(
        df_chr1,
        prefix=prefix + "_chr1",
        flag_chr="chr1",
        fixed_thresholds=(upper_threshold, lower_threshold),
    )
    plot_chr2 = egh.gwes_high_from_df(
        df_chr2,
        prefix=prefix + "_chr2",
        flag_chr="chr2",
        fixed_thresholds=(upper_threshold, lower_threshold),
    )

    filtered_path = None
    if save_filtered:
        filtered_path = prefix + "_filtered_pairs.parquet"
        df_filtered.to_parquet(filtered_path, index=False)

    if verbose:
        print(f"Q1: {q1:.6f}")
        print(f"Q3: {q3:.6f}")
        print(f"IQR: {iqr:.6f}")
        print(f"Upper threshold: {upper_threshold:.6f}")
        print(f"Lower threshold: {lower_threshold:.6f}")
        print(f"Strengthened pairs: {len(df_strengthen):,}")
        print(f"Weakened pairs: {len(df_weaken):,}")
        print(f"Selected pairs: {len(df_filtered):,}")

    return {
        "thresholds": {
            "upper": float(upper_threshold),
            "lower": float(lower_threshold),
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "k": float(k),
        },
        "filtered_pairs": df_filtered,
        "strengthened_pairs": df_strengthen,
        "weakened_pairs": df_weaken,
        "df_chr1": df_chr1,
        "df_chr2": df_chr2,
        "filtered_chr1": df_filtered_chr1,
        "filtered_chr2": df_filtered_chr2,
        "plot_chr1": plot_chr1,
        "plot_chr2": plot_chr2,
        "filtered_path": filtered_path,
    }


def _prepare_pair_score_context(
    data_df,
    weight_ser,
    *,
    label,
    gpu_device=0,
    batch_size=100_000,
):
    """Upload reusable float32 arrays for selected-pair GPU scoring."""
    if not isinstance(data_df, pd.DataFrame) or data_df.empty:
        raise ValueError(f"{label}: data_df must be a non-empty DataFrame")
    if not data_df.index.is_unique:
        raise ValueError(f"{label}: data_df index must be unique")

    values = data_df.to_numpy(copy=False)
    is_binary = (
        np.issubdtype(values.dtype, np.number)
        and np.isfinite(values).all()
        and np.all((values == 0) | (values == 1))
    )
    if not is_binary:
        data_binary = ect.convert_snp_to_binary(
            data_df,
            n_jobs=1,
            convert_chunksize=20_000,
        )
    else:
        data_binary = data_df

    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError(f"{label}: weight DataFrame must have one column")
        weight_ser = weight_ser.iloc[:, 0]
    weights = weight_ser.reindex(data_binary.columns).astype(np.float32)
    if weights.isna().any() or not np.isfinite(weights.to_numpy()).all():
        raise ValueError(f"{label}: sample weights are missing or non-finite")

    x_cpu = np.ascontiguousarray(
        data_binary.to_numpy(dtype=np.float32, copy=False)
    )
    w_cpu = np.ascontiguousarray(weights.to_numpy(), dtype=np.float32)
    cp, _, device_count = _load_gpu_backend()
    gpu_device = int(gpu_device)
    if not 0 <= gpu_device < device_count:
        raise ValueError(
            f"gpu_device={gpu_device} is outside [0, {device_count})"
        )
    if int(batch_size) < 1:
        raise ValueError("GPU selected-pair batch_size must be positive")
    with cp.cuda.Device(gpu_device):
        x_gpu = cp.asarray(x_cpu, dtype=cp.float32)
        w_gpu = cp.asarray(w_cpu, dtype=cp.float32)
        weighted_gpu = x_gpu * w_gpu[None, :]
    return {
        "label": label,
        "index": pd.Index(data_binary.index),
        "cp": cp,
        "gpu_device": gpu_device,
        "batch_size": int(batch_size),
        "x": x_gpu,
        "w": w_gpu,
        "weighted_x": weighted_gpu,
    }


def _calculate_pair_scores_cached(score_context, pairs):
    """Calculate requested pair scores using a prepared scoring context."""
    pair_columns = ["locus1", "locus2"]
    if not isinstance(pairs, pd.DataFrame):
        pairs = pd.DataFrame(pairs, columns=pair_columns)
    missing = sorted(set(pair_columns).difference(pairs.columns))
    if missing:
        raise ValueError(f"pairs is missing columns: {missing}")
    requested = pairs[pair_columns].copy()
    if requested.empty:
        return pd.DataFrame(columns=["locus1", "locus2", "v"])

    locus1 = requested["locus1"].to_numpy()
    locus2 = requested["locus2"].to_numpy()
    requested["locus1"] = np.minimum(locus1, locus2)
    requested["locus2"] = np.maximum(locus1, locus2)
    requested = requested.drop_duplicates(pair_columns).reset_index(drop=True)

    data_index = score_context["index"]
    row1 = data_index.get_indexer(requested["locus1"])
    row2 = data_index.get_indexer(requested["locus2"])
    valid = (row1 >= 0) & (row2 >= 0)
    if not valid.all():
        missing_pairs = requested.loc[~valid, pair_columns]
        raise ValueError(
            f"{score_context['label']}: {len(missing_pairs):,} pairs "
            "reference loci absent from the cached binary matrix"
        )

    cp = score_context["cp"]
    values = np.empty(len(requested), dtype=np.float32)
    eps = cp.float32(1e-27)
    prob_eps = cp.float32(1e-7)
    batch_size = score_context["batch_size"]
    with cp.cuda.Device(score_context["gpu_device"]):
        for start in range(0, len(requested), batch_size):
            stop = min(start + batch_size, len(requested))
            rows1 = cp.asarray(row1[start:stop], dtype=cp.int64)
            rows2 = cp.asarray(row2[start:stop], dtype=cp.int64)
            source = score_context["x"][rows1]
            target_weighted = score_context["weighted_x"][rows2]
            inverse_source = 1.0 - source

            dp = source @ score_context["w"]
            dq = inverse_source @ score_context["w"]
            valid_reference = (dp > 0) & (dq > 0)
            safe_dp = cp.where(valid_reference, dp, cp.float32(1.0))
            safe_dq = cp.where(valid_reference, dq, cp.float32(1.0))
            data_p = cp.sum(target_weighted * source, axis=1)
            target_marginal = cp.sum(target_weighted, axis=1)
            data_q = target_marginal - data_p

            p1 = cp.clip(data_p / safe_dp, prob_eps, 1 - prob_eps)
            q1 = cp.clip(data_q / safe_dq, prob_eps, 1 - prob_eps)
            p0 = 1 - p1
            q0 = 1 - q1
            size_p = safe_dp / (safe_dp + safe_dq)
            size_q = 1 - size_p

            st1_0 = p1 + eps
            st1_1 = p0 + eps
            st2_0 = q1 + eps
            st2_1 = q0 + eps
            mix0 = st1_0 * size_p + st2_0 * size_q
            mix1 = st1_1 * size_p + st2_1 * size_q
            r1_0 = cp.maximum(st1_0 / mix0, eps)
            r1_1 = cp.maximum(st1_1 / mix1, eps)
            r2_0 = cp.maximum(st2_0 / mix0, eps)
            r2_1 = cp.maximum(st2_1 / mix1, eps)
            js = (
                (st1_0 * cp.log2(r1_0) + st1_1 * cp.log2(r1_1))
                * size_p
                + (st2_0 * cp.log2(r2_0) + st2_1 * cp.log2(r2_1))
                * size_q
            )
            batch_values = cp.where(
                valid_reference,
                cp.sqrt(cp.maximum(js, 0)),
                cp.float32(0.0),
            )
            values[start:stop] = cp.asnumpy(batch_values)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{score_context['label']}: cached scorer returned non-finite values"
        )
    result = requested.copy()
    result["v"] = values
    return result


def _restore_selected_highorder_pairs(
    selected_pairs,
    pattern_map_full,
    pattern_map_high,
    full_score_context,
    high_score_context,
    thresholds,
    *,
    prefix,
    max_output_rows,
    pair_n_jobs,
    distance_n_jobs,
    distance_chunk_size,
    ld_threshold,
    chr1_len=3_288_558,
    chr2_len=1_877_212,
    save_output=True,
    verbose=True,
):
    """Restore selected representative pairs and recompute physical scores."""
    pair_columns = ["locus1", "locus2"]

    if selected_pairs.empty:
        empty = selected_pairs.copy()
        empty["applied_upper_threshold"] = pd.Series(dtype=np.float64)
        empty["applied_lower_threshold"] = pd.Series(dtype=np.float64)
        empty["change_direction"] = pd.Series(dtype="object")
        output_path = None
        if save_output:
            output_path = prefix + "_restored_filtered_pairs.parquet"
            empty.to_parquet(output_path, index=False)
        if verbose:
            print("No representative pair passed the high-order threshold")
            print("Final restored pairs: 0")
        return {
            "candidates": empty.copy(),
            "filtered": empty,
            "strengthened": empty.copy(),
            "weakened": empty.copy(),
            "output_path": output_path,
            "estimated_upper_bound": 0,
        }

    def load_pattern_map(value, name):
        if isinstance(value, pd.DataFrame):
            result = value
        elif isinstance(value, (str, os.PathLike)):
            result = pd.read_parquet(value)
        else:
            raise TypeError(
                f"{name} must be a DataFrame or Parquet path"
            )
        required = {"locus", "representative", "pattern_size"}
        missing = sorted(required.difference(result.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        return result

    full_map = load_pattern_map(pattern_map_full, "pattern_map_full")
    high_map = load_pattern_map(pattern_map_high, "pattern_map_high")

    def prepare_restore_input(frame, pattern_map, value_column):
        representatives = pd.Index(
            pattern_map["representative"].drop_duplicates()
        )
        valid = (
            frame["locus1"].isin(representatives)
            & frame["locus2"].isin(representatives)
        )
        restore_input = frame.loc[
            valid,
            pair_columns + ["v_f", "v_h"],
        ].copy()
        restore_input["v"] = frame.loc[valid, value_column].to_numpy()
        return restore_input

    full_input = prepare_restore_input(
        selected_pairs,
        full_map,
        "v_f",
    )
    high_input = prepare_restore_input(
        selected_pairs,
        high_map,
        "v_h",
    )

    if full_input.empty and high_input.empty:
        raise ValueError(
            "No selected pair is represented in either pattern map"
        )

    def estimate_rows(restore_input, pattern_map):
        if restore_input.empty:
            return 0
        sizes = (
            pattern_map.drop_duplicates("representative")
            .set_index("representative")["pattern_size"]
        )
        size1 = restore_input["locus1"].map(sizes).astype(np.int64)
        size2 = restore_input["locus2"].map(sizes).astype(np.int64)
        return int((size1 * size2).sum())

    estimated_full = estimate_rows(full_input, full_map)
    estimated_high = estimate_rows(high_input, high_map)
    estimated_upper_bound = estimated_full + estimated_high
    if (
        max_output_rows is not None
        and estimated_upper_bound > max_output_rows
    ):
        raise MemoryError(
            "Final restoration may create up to "
            f"{estimated_upper_bound:,} rows, exceeding "
            f"max_output_rows={max_output_rows:,}"
        )

    restored_parts = []
    for source, restore_input, pattern_map in (
        ("full", full_input, full_map),
        ("high", high_input, high_map),
    ):
        if restore_input.empty:
            continue
        restored = ef.restore_pattern_pairs(
            restore_input,
            pattern_map,
            input_stage="post_gwes",
            include_within_pattern=False,
            preserve_pair_columns=True,
            output_path=None,
            max_output_rows=max_output_rows,
            verbose=verbose,
        )
        if source == "full":
            restored_part = restored[pair_columns + ["v_f", "v_h"]].copy()
        else:
            restored_part = restored[pair_columns + ["v_h"]].copy()
            restored_part["v_f"] = np.float32(-1.0)
            restored_part = restored_part[pair_columns + ["v_f", "v_h"]]
        restored_parts.append(restored_part)

    physical_candidates = (
        pd.concat(restored_parts, ignore_index=True)
        .groupby(pair_columns, as_index=False, sort=False)
        .first()
        .reset_index(drop=True)
    )
    if (
        max_output_rows is not None
        and len(physical_candidates) > max_output_rows
    ):
        raise MemoryError(
            f"Final restoration created {len(physical_candidates):,} rows, "
            f"exceeding max_output_rows={max_output_rows:,}"
        )

    complete = physical_candidates
    missing_full = complete["v_f"] < 0
    if missing_full.any():
        full_scores = _calculate_pair_scores_cached(
            full_score_context,
            complete.loc[missing_full, pair_columns],
        )
        full_values = full_scores.set_index(pair_columns)["v"]
        missing_keys = pd.MultiIndex.from_frame(
            complete.loc[missing_full, pair_columns]
        )
        complete.loc[missing_full, "v_f"] = full_values.reindex(
            missing_keys
        ).to_numpy()

    missing_high = complete["v_h"].isna()
    if missing_high.any():
        high_scores = _calculate_pair_scores_cached(
            high_score_context,
            complete.loc[missing_high, pair_columns],
        )
        high_values = high_scores.set_index(pair_columns)["v"]
        missing_keys = pd.MultiIndex.from_frame(
            complete.loc[missing_high, pair_columns]
        )
        complete.loc[missing_high, "v_h"] = high_values.reindex(
            missing_keys
        ).to_numpy()

    invalid = (
        ~np.isfinite(complete["v_f"].to_numpy(dtype=np.float64))
        | ~np.isfinite(complete["v_h"].to_numpy(dtype=np.float64))
    )
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum()):,} restored pairs could not be scored"
        )

    complete = eg_vp.calculate_pair_distances(
        complete,
        n_jobs=distance_n_jobs,
        chunk_size=distance_chunk_size,
        ld_threshold=ld_threshold,
        verbose=verbose,
    )
    complete["diff"] = complete["v_h"] - complete["v_f"]

    upper = np.full(len(complete), np.nan, dtype=np.float64)
    lower = np.full(len(complete), np.nan, dtype=np.float64)
    if isinstance(thresholds, dict) and {"chr1", "chr2"}.issubset(
        thresholds
    ):
        chr2_end = chr1_len + chr2_len
        chr1_mask = (
            ((complete["locus1"] < chr1_len)
             | (complete["locus2"] < chr1_len))
            & (complete["geno_dis"] > 0)
        ).to_numpy()
        chr2_mask = (
            (complete["locus1"] > chr1_len)
            & (complete["locus2"] > chr1_len)
            & ((complete["locus1"] < chr2_end)
               | (complete["locus2"] < chr2_end))
            & (complete["geno_dis"] > 0)
        ).to_numpy()
        upper[chr1_mask] = float(thresholds["chr1"]["stable_upper"])
        lower[chr1_mask] = float(thresholds["chr1"]["stable_lower"])
        upper[chr2_mask] = float(thresholds["chr2"]["stable_upper"])
        lower[chr2_mask] = float(thresholds["chr2"]["stable_lower"])
    else:
        upper.fill(float(thresholds["stable_upper"]))
        lower.fill(float(thresholds["stable_lower"]))

    complete["applied_upper_threshold"] = upper
    complete["applied_lower_threshold"] = lower
    strengthened_mask = np.isfinite(upper) & (
        complete["diff"].to_numpy() >= upper
    )
    weakened_mask = np.isfinite(lower) & (
        complete["diff"].to_numpy() <= lower
    )
    complete["change_direction"] = "not_selected"
    complete.loc[strengthened_mask, "change_direction"] = "strengthened"
    complete.loc[weakened_mask, "change_direction"] = "weakened"
    filtered = complete.loc[
        strengthened_mask | weakened_mask
    ].copy().reset_index(drop=True)

    output_path = None
    if save_output:
        output_path = prefix + "_restored_filtered_pairs.parquet"
        filtered.to_parquet(output_path, index=False)

    if verbose:
        print(f"Selected representative pairs: {len(selected_pairs):,}")
        print(f"Restored physical candidates: {len(complete):,}")
        print(f"Final restored pairs: {len(filtered):,}")

    return {
        "candidates": complete,
        "filtered": filtered,
        "strengthened": filtered.loc[
            filtered["change_direction"] == "strengthened"
        ].copy(),
        "weakened": filtered.loc[
            filtered["change_direction"] == "weakened"
        ].copy(),
        "output_path": output_path,
        "estimated_upper_bound": int(estimated_upper_bound),
    }


def run_highorder_comparison_pipeline(
    df_pairs_full,
    df_pairs_high,
    data_full,
    data_high,
    weight_full,
    weight_high,
    dir_save,
    prefix,
    *,
    threshold_method="auto",
    k_upper=1.0,
    k_lower=2.0,
    upper_quantile=0.95,
    lower_quantile=0.05,
    n_bootstrap=200,
    bootstrap_sample_size=100_000,
    min_stability=0.80,
    separate_chromosomes=True,
    detection_sample_size=50_000,
    min_side_fraction=0.05,
    min_mode_abs=0.15,
    min_mode_separation=0.30,
    max_valley_ratio=0.50,
    pattern_map_full=None,
    pattern_map_high=None,
    restore_final_patterns=True,
    final_max_output_rows=50_000_000,
    pair_n_jobs=1,
    selected_pair_gpu_id=0,
    gpu_pair_batch_size=100_000,
    distance_n_jobs=1,
    distance_chunk_size=100_000,
    ld_threshold=5_000,
    save_comparison=True,
    save_filtered=True,
    make_plots=True,
    verbose=True,
):
    """Complete missing pair scores and run high-order threshold analysis."""
    import polars as pl

    pair_columns = ["locus1", "locus2"]

    for name, frame in (
        ("df_pairs_full", df_pairs_full),
        ("df_pairs_high", df_pairs_high),
        ("data_full", data_full),
        ("data_high", data_high),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if frame.empty:
            raise ValueError(f"{name} cannot be empty")

    for name, frame in (
        ("df_pairs_full", df_pairs_full),
        ("df_pairs_high", df_pairs_high),
    ):
        missing_columns = [
            column
            for column in ["locus1", "locus2", "v"]
            if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: {missing_columns}"
            )

    if not data_full.index.is_unique or not data_high.index.is_unique:
        raise ValueError("SNP matrix locus indices must be unique")
    if not isinstance(prefix, (str, int)):
        raise TypeError("prefix must be a string or integer")
    prefix = str(prefix).strip()
    if not prefix:
        raise ValueError("prefix cannot be empty")
    if os.path.basename(prefix) != prefix:
        raise ValueError("prefix must be a file-name component")
    threshold_method = str(threshold_method).strip().lower()
    if threshold_method not in {"auto", "iqr", "quantile"}:
        raise ValueError(
            "threshold_method must be auto, iqr, or quantile"
        )
    if threshold_method == "auto" and not separate_chromosomes:
        raise ValueError(
            "threshold_method='auto' requires separate_chromosomes=True"
        )
    os.makedirs(dir_save, exist_ok=True)
    comparison_prefix = os.path.join(dir_save, prefix)

    full_score_context = _prepare_pair_score_context(
        data_full,
        weight_full,
        label="Full dataset",
        gpu_device=selected_pair_gpu_id,
        batch_size=gpu_pair_batch_size,
    )
    high_score_context = _prepare_pair_score_context(
        data_high,
        weight_high,
        label="High-order subset",
        gpu_device=selected_pair_gpu_id,
        batch_size=gpu_pair_batch_size,
    )

    df_union_pl = eg_vp.merge_pairs_with_fill(
        df_pairs_full,
        df_pairs_high,
        fill_value=-1,
        return_pandas=False,
    )
    pairs_missing_full = (
        df_union_pl.filter(pl.col("v1") == -1)
        .select(pair_columns)
        .unique(maintain_order=True)
        .to_pandas()
    )
    pairs_missing_high = (
        df_union_pl.filter(pl.col("v2") == -1)
        .select(pair_columns)
        .unique(maintain_order=True)
        .to_pandas()
    )

    if verbose:
        print(
            "Pairs requiring calculation in the full dataset: ",
            f"{len(pairs_missing_full):,}",
            sep="",
        )
        print(
            "Pairs requiring calculation in the high-order subset: ",
            f"{len(pairs_missing_high):,}",
            sep="",
        )

    if pairs_missing_full.empty:
        scores_missing_full = pd.DataFrame(
            columns=["locus1", "locus2", "v"]
        )
    else:
        scores_missing_full = _calculate_pair_scores_cached(
            full_score_context,
            pairs_missing_full,
        )

    if pairs_missing_high.empty:
        scores_missing_high = pd.DataFrame(
            columns=["locus1", "locus2", "v"]
        )
    else:
        scores_missing_high = _calculate_pair_scores_cached(
            high_score_context,
            pairs_missing_high,
        )

    full_complete_pl = pl.concat(
        [pl.from_pandas(df_pairs_full), pl.from_pandas(scores_missing_full)],
        how="diagonal_relaxed",
    ).unique(subset=pair_columns, keep="first", maintain_order=True)
    high_complete_pl = pl.concat(
        [pl.from_pandas(df_pairs_high), pl.from_pandas(scores_missing_high)],
        how="diagonal_relaxed",
    ).unique(subset=pair_columns, keep="first", maintain_order=True)

    df_merged_pl = eg_vp.merge_pairs_with_fill(
        full_complete_pl,
        high_complete_pl,
        fill_value=-1,
        return_pandas=False,
    )
    if "geno_dis" in df_union_pl.columns:
        distance_lookup_pl = (
            df_union_pl.select(pair_columns + ["geno_dis"])
            .unique(subset=pair_columns, keep="first", maintain_order=True)
        )
        df_merged_pl = (
            df_merged_pl.drop("geno_dis")
            .join(distance_lookup_pl, on=pair_columns, how="left")
        )
    df_merged = df_merged_pl.to_pandas()
    invalid_full = (
        (df_merged["v1"] == -1)
        | ~np.isfinite(df_merged["v1"].to_numpy(dtype=np.float64))
    )
    invalid_high = (
        (df_merged["v2"] == -1)
        | ~np.isfinite(df_merged["v2"].to_numpy(dtype=np.float64))
    )
    missing_full_count = int(invalid_full.sum())
    missing_high_count = int(invalid_high.sum())
    if missing_full_count or missing_high_count:
        raise ValueError(
            "Some candidate pairs could not be calculated in both datasets: "
            f"full={missing_full_count}, high={missing_high_count}"
        )

    df_merged = df_merged.rename(columns={"v1": "v_f", "v2": "v_h"})
    existing_distance = pd.to_numeric(
        df_merged.get("geno_dis"),
        errors="coerce",
    ) if "geno_dis" in df_merged.columns else None
    if existing_distance is None or existing_distance.isna().any():
        df_merged = eg_vp.calculate_pair_distances(
            df_merged,
            n_jobs=distance_n_jobs,
            chunk_size=distance_chunk_size,
            ld_threshold=ld_threshold,
            verbose=verbose,
        )
    else:
        df_merged["geno_dis"] = existing_distance
        if verbose:
            print("Reusing existing representative-pair genome distances")
    df_merged["diff"] = df_merged["v_h"] - df_merged["v_f"]

    comparison_path = None
    if save_comparison:
        comparison_path = comparison_prefix + "_comparison.parquet"
        df_merged.to_parquet(comparison_path, index=False)

    if threshold_method == "auto":
        threshold_result = egh.run_highorder_threshold_analysis_auto(
            df_merged=df_merged,
            prefix=comparison_prefix,
            k_upper=k_upper,
            k_lower=k_lower,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            n_bootstrap=n_bootstrap,
            bootstrap_sample_size=bootstrap_sample_size,
            min_stability=min_stability,
            detection_sample_size=detection_sample_size,
            min_side_fraction=min_side_fraction,
            min_mode_abs=min_mode_abs,
            min_mode_separation=min_mode_separation,
            max_valley_ratio=max_valley_ratio,
            save_filtered=save_filtered,
            make_plots=make_plots,
            verbose=verbose,
        )
    elif threshold_method == "quantile":
        threshold_result = (
            egh.run_highorder_threshold_analysis_bootstrap_quantile(
                df_merged=df_merged,
                prefix=comparison_prefix,
                upper_quantile=upper_quantile,
                lower_quantile=lower_quantile,
                n_bootstrap=n_bootstrap,
                bootstrap_sample_size=bootstrap_sample_size,
                min_stability=min_stability,
                separate_chromosomes=separate_chromosomes,
                save_filtered=save_filtered,
                make_plots=make_plots,
                verbose=verbose,
            )
        )
    else:
        threshold_result = egh.run_highorder_threshold_analysis_bootstrap(
            df_merged=df_merged,
            prefix=comparison_prefix,
            k_upper=k_upper,
            k_lower=k_lower,
            n_bootstrap=n_bootstrap,
            bootstrap_sample_size=bootstrap_sample_size,
            min_stability=min_stability,
            separate_chromosomes=separate_chromosomes,
            save_filtered=save_filtered,
            make_plots=make_plots,
            verbose=verbose,
        )

    restoration_result = None
    if restore_final_patterns:
        if pattern_map_full is None or pattern_map_high is None:
            raise ValueError(
                "pattern_map_full and pattern_map_high are required when "
                "restore_final_patterns=True"
            )
        restoration_result = _restore_selected_highorder_pairs(
            selected_pairs=threshold_result["filtered_pairs"],
            pattern_map_full=pattern_map_full,
            pattern_map_high=pattern_map_high,
            full_score_context=full_score_context,
            high_score_context=high_score_context,
            thresholds=threshold_result["thresholds"],
            prefix=comparison_prefix,
            max_output_rows=final_max_output_rows,
            pair_n_jobs=pair_n_jobs,
            distance_n_jobs=distance_n_jobs,
            distance_chunk_size=distance_chunk_size,
            ld_threshold=ld_threshold,
            save_output=save_filtered,
            verbose=verbose,
        )
        final_filtered = restoration_result["filtered"]
        final_strengthened = restoration_result["strengthened"]
        final_weakened = restoration_result["weakened"]
    else:
        final_filtered = threshold_result["filtered_pairs"]
        final_strengthened = threshold_result["strengthened_pairs"]
        final_weakened = threshold_result["weakened_pairs"]

    return {
        "merged_pairs": df_merged,
        "filtered_pairs": final_filtered,
        "strengthened_pairs": final_strengthened,
        "weakened_pairs": final_weakened,
        "representative_filtered_pairs": threshold_result["filtered_pairs"],
        "restoration_result": restoration_result,
        "thresholds": threshold_result["thresholds"],
        "threshold_result": threshold_result,
        "pairs_missing_full": pairs_missing_full,
        "pairs_missing_high": pairs_missing_high,
        "scores_missing_full": scores_missing_full,
        "scores_missing_high": scores_missing_high,
        "comparison_prefix": comparison_prefix,
        "paths": {
            "comparison": comparison_path,
            "filtered": (
                restoration_result["output_path"]
                if restoration_result is not None
                else threshold_result["filtered_path"]
            ),
            "representative_filtered": threshold_result["filtered_path"],
            "chr1_plot": (
                threshold_result["plot_chr1"]["figure_path"]
                if threshold_result["plot_chr1"] is not None else None
            ),
            "chr2_plot": (
                threshold_result["plot_chr2"]["figure_path"]
                if threshold_result["plot_chr2"] is not None else None
            ),
        },
        "counts": {
            "union_pairs": int(len(df_union_pl)),
            "missing_full": int(len(pairs_missing_full)),
            "missing_high": int(len(pairs_missing_high)),
            "complete_pairs": int(len(df_merged)),
            "representative_filtered_pairs": int(
                len(threshold_result["filtered_pairs"])
            ),
            "filtered_pairs": int(len(final_filtered)),
            "strengthened_pairs": int(
                len(final_strengthened)
            ),
            "weakened_pairs": int(len(final_weakened)),
        },
    }

def r_plot(df, prefix):
    sns.set_style("white")
    plt.figure(figsize=(8,8))
    plt.scatter(df['v_f'], df['v_h'], s=5, alpha=0.7)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('First')
    plt.ylabel('High')
    plt.title('EpiDive Correlation')
    plt.savefig(prefix+".Corr_Epi_high_1st.png",dpi=200)
    plt.close()


def classify_subgroup(
    n_samples,
    *,
    weight_ser=None,
    original_samples=None,
    ess_mode="absolute",
    formal_min_ess="auto",
    exploratory_min_ess="auto",
    formal_max_se=0.08,
    exploratory_max_se=0.10,
    formal_min_fraction=0.05,
    exploratory_min_fraction=0.02,
):
    """Classify a subgroup using raw or weighted effective sample size."""
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    def resolve_minimum(value, max_se, name):
        if isinstance(value, str):
            if value.strip().lower() != "auto":
                raise ValueError(f'{name} must be numeric or "auto"')
            max_se = float(max_se)
            if not 0 < max_se <= 0.5:
                raise ValueError(f"{name.replace('_min_ess', '_max_se')} must be in (0, 0.5]")
            return int(np.ceil(0.25 / (max_se * max_se)))
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    formal_min_ess = resolve_minimum(
        formal_min_ess,
        formal_max_se,
        "formal_min_ess",
    )
    exploratory_min_ess = resolve_minimum(
        exploratory_min_ess,
        exploratory_max_se,
        "exploratory_min_ess",
    )
    ess_mode = str(ess_mode).strip().lower()
    if ess_mode not in {"absolute", "hybrid"}:
        raise ValueError('ess_mode must be "absolute" or "hybrid"')
    if ess_mode == "hybrid":
        if original_samples is None:
            raise ValueError("original_samples is required for hybrid ESS mode")
        original_samples = int(original_samples)
        if original_samples < n_samples:
            raise ValueError("original_samples cannot be smaller than n_samples")
        for value, name in (
            (formal_min_fraction, "formal_min_fraction"),
            (exploratory_min_fraction, "exploratory_min_fraction"),
        ):
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        formal_min_ess = max(
            formal_min_ess,
            int(np.ceil(original_samples * float(formal_min_fraction))),
        )
        exploratory_min_ess = max(
            exploratory_min_ess,
            int(np.ceil(original_samples * float(exploratory_min_fraction))),
        )
    if exploratory_min_ess <= 0:
        raise ValueError("exploratory_min_ess must be positive")
    if formal_min_ess < exploratory_min_ess:
        raise ValueError(
            "formal_min_ess cannot be smaller than exploratory_min_ess"
        )

    if weight_ser is None:
        effective_n = float(n_samples)
    else:
        weights = np.asarray(weight_ser, dtype=np.float64).reshape(-1)
        if weights.size != n_samples:
            raise ValueError(
                "weight_ser length does not match n_samples"
            )
        if not np.isfinite(weights).all():
            raise ValueError("weight_ser contains non-finite values")
        weight_sum = weights.sum()
        weight_square_sum = np.square(weights).sum()
        if weight_sum <= 0 or weight_square_sum <= 0:
            raise ValueError("Invalid sample weights")
        effective_n = weight_sum * weight_sum / weight_square_sum

    if effective_n >= formal_min_ess:
        analysis_level = "formal"
        run_analysis = True
    elif effective_n >= exploratory_min_ess:
        analysis_level = "exploratory"
        run_analysis = True
    else:
        analysis_level = "skip"
        run_analysis = False

    return {
        "n_samples": n_samples,
        "effective_n": float(effective_n),
        "formal_min_ess": float(formal_min_ess),
        "exploratory_min_ess": float(exploratory_min_ess),
        "formal_max_se": float(formal_max_se),
        "exploratory_max_se": float(exploratory_max_se),
        "ess_mode": ess_mode,
        "formal_min_fraction": float(formal_min_fraction),
        "exploratory_min_fraction": float(exploratory_min_fraction),
        "analysis_level": analysis_level,
        "run_analysis": run_analysis,
    }

def run_vp_pipeline(
    snp_m,
    dir_save,
    *,
    prefix="vp10k",
    maf=0.02,
    min_mac=5,
    model="SCC",
    scc_threshold=0.98,
    hcc_threshold=0.75,
    threshold=0.3,
    auto_target_max_pairs=5_000_000,
    auto_sample_pairs=50_000,
    auto_safety_factor=0.60,
    auto_min_threshold=0.3,
    auto_random_state=2026,
    k=1,
    apply_gwes_filter=True,
    highorder=None,
    highorder_formal_min_ess="auto",
    highorder_exploratory_min_ess="auto",
    highorder_formal_max_se=0.08,
    highorder_exploratory_max_se=0.10,
    highorder_ess_mode="absolute",
    highorder_formal_min_fraction=0.05,
    highorder_exploratory_min_fraction=0.02,
    reweight_n_jobs=-1,
    gpu_block_size=5_000,
    gpu_stream_output=True,
    gpu_convert_chunksize=40_000,
    gpu_convert_n_jobs=-1,
    distance_n_jobs=1,
    distance_chunk_size=100_000,
    ld_chr1=5_000,
    ld_chr2=17_500,
    max_output_rows=50_000_000,
    restore_patterns=False,
    include_cross_chromosome=True,
    make_plots=True,
    verbose=True,
):
    """
    Run the complete VP EpiDive workflow.

    Parameters
    ----------
    snp_m : pandas.DataFrame
        Locus-by-sample SNP matrix.
    dir_save : str
        Output directory.
    prefix : str or int, default="vp10k"
        File-name prefix used to identify the dataset or background.
    maf : float, default=0.02
        Minimum minor allele frequency.
    min_mac : int, default=5
        Minimum minor allele count. Both the MAF and MAC filters must pass.
        Set to 0 to disable the MAC filter.
    model : {"SCC", "HCC", "NW"}, default="SCC"
        Sample reweighting model.
    scc_threshold : float, default=0.98
        SCC similarity threshold.
    hcc_threshold : float, default=0.75
        HCC distance threshold.
    threshold : float or "auto", default=0.3
        Initial EpiDis output threshold. Use "auto" to estimate it
        from random representative-pattern pairs.
    k : int, default=1
        IQR level used to estimate the final GWES threshold.
    apply_gwes_filter : bool, default=True
        If True, final_pairs must pass the initial threshold, GWES threshold,
        and LD rules. If False, final_pairs must pass the initial threshold
        and LD rules, but the calculated GWES threshold is not applied.
    highorder : int or None, default=None
        Original sample count for high-order subgroup screening. When set,
        subgroups smaller than highorder_min_samples are skipped before SNP
        conversion, sample reweighting, and pairwise calculation.
    highorder_formal_min_ess : float or "auto", default="auto"
        Effective sample size required for a formal analysis label.
    highorder_exploratory_min_ess : float or "auto", default="auto"
        Minimum effective sample size required to run the analysis.
    highorder_formal_max_se : float, default=0.08
        Maximum worst-case binomial standard error used to derive the formal
        ESS threshold when highorder_formal_min_ess="auto".
    highorder_exploratory_max_se : float, default=0.10
        Maximum worst-case binomial standard error used to derive the
        exploratory ESS threshold when its minimum is "auto".
    gpu_block_size : int, default=5000
        CUDA tile size used by the multi-GPU representative-pair runner.
    gpu_stream_output : bool, default=True
        Stream GPU result blocks to Parquet before loading the GWES table.

    Returns
    -------
    dict
        Final pairs, threshold, plots, weights, pattern map,
        full binary matrix, representative matrix, counts, and output paths.
    """
    # --------------------------------------------------------------
    # 0. Validate input and prepare output paths
    # --------------------------------------------------------------
    if not isinstance(snp_m, pd.DataFrame):
        raise TypeError("snp_m must be a pandas DataFrame")

    if snp_m.empty:
        raise ValueError("snp_m cannot be empty")

    if not snp_m.index.is_unique:
        raise ValueError(
            "snp_m contains duplicate locus identifiers"
        )

    if not snp_m.columns.is_unique:
        raise ValueError(
            "snp_m contains duplicate sample identifiers"
        )

    highorder_screen = None
    if highorder is not None:
        if isinstance(highorder, (bool, np.bool_)):
            raise TypeError("highorder must be an integer sample count")
        original_samples = int(highorder)
        if original_samples < 1 or original_samples != highorder:
            raise ValueError("highorder must be a positive integer")
        subgroup_samples = int(snp_m.shape[1])
        if subgroup_samples > original_samples:
            raise ValueError(
                "The subgroup sample count cannot exceed highorder"
            )
        highorder_screen = classify_subgroup(
            subgroup_samples,
            original_samples=original_samples,
            ess_mode=highorder_ess_mode,
            formal_min_ess=highorder_formal_min_ess,
            exploratory_min_ess=highorder_exploratory_min_ess,
            formal_max_se=highorder_formal_max_se,
            exploratory_max_se=highorder_exploratory_max_se,
            formal_min_fraction=highorder_formal_min_fraction,
            exploratory_min_fraction=highorder_exploratory_min_fraction,
        )
        highorder_screen.update({
            "original_samples": original_samples,
            "subgroup_samples": subgroup_samples,
            "subgroup_fraction": subgroup_samples / original_samples,
            "screen_stage": "raw",
        })

    if not 0 <= maf <= 0.5:
        raise ValueError("maf must be between 0 and 0.5")

    if isinstance(min_mac, (bool, np.bool_)):
        raise TypeError("min_mac must be a non-negative integer")
    min_mac_value = float(min_mac)
    if (
        not np.isfinite(min_mac_value)
        or min_mac_value < 0
        or not min_mac_value.is_integer()
    ):
        raise ValueError("min_mac must be a non-negative integer")
    min_mac = int(min_mac_value)

    auto_threshold = isinstance(threshold, str)
    if auto_threshold:
        if threshold.strip().lower() != "auto":
            raise ValueError('string threshold must be "auto"')
    else:
        threshold = float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

    if auto_target_max_pairs < 1:
        raise ValueError("auto_target_max_pairs must be positive")
    if auto_sample_pairs < 1_000:
        raise ValueError("auto_sample_pairs must be at least 1000")
    if not 0 < auto_safety_factor <= 1:
        raise ValueError("auto_safety_factor must be in (0, 1]")
    if not 0 <= auto_min_threshold <= 1:
        raise ValueError("auto_min_threshold must be between 0 and 1")

    if k < 0:
        raise ValueError("k cannot be negative")

    if not isinstance(apply_gwes_filter, (bool, np.bool_)):
        raise TypeError("apply_gwes_filter must be boolean")

    if gpu_block_size < 1 or gpu_convert_chunksize < 1:
        raise ValueError(
            "gpu_block_size and gpu_convert_chunksize must be positive"
        )
    if not isinstance(gpu_stream_output, (bool, np.bool_)):
        raise TypeError("gpu_stream_output must be boolean")

    model = model.upper()

    if model not in {"SCC", "HCC", "NW"}:
        raise ValueError(
            "model must be SCC, HCC, or NW"
        )

    if not isinstance(prefix, (str, int)):
        raise TypeError("prefix must be a string or integer")

    prefix = str(prefix).strip()

    if not prefix:
        raise ValueError("prefix cannot be empty")

    if os.path.basename(prefix) != prefix:
        raise ValueError(
            "prefix must be a file-name component, not a path"
        )

    os.makedirs(dir_save, exist_ok=True)

    run_prefix = os.path.join(dir_save, prefix)

    pattern_map_path = (
        run_prefix + "_pattern_map.parquet"
    )
    representative_pairs_path = (
        run_prefix + "_Epi_pairs_w.parquet"
    )
    final_pairs_path = (
        run_prefix + "_final_pairs.parquet"
    )
    initial_output_path = (
        run_prefix + "_initial_ld_pairs.parquet"
    )
    restored_plot_prefix = (
        run_prefix + "_restored"
    )

    if highorder_screen is not None and not highorder_screen["run_analysis"]:
        empty_pairs = pd.DataFrame(
            columns=["locus1", "locus2", "v"]
        )
        skip_reason = (
            "High-order subgroup has "
            f"effective_n={highorder_screen['effective_n']:.3f}, below "
            "the minimum of "
            f"{highorder_screen['exploratory_min_ess']:.3f}"
        )
        if verbose:
            print(skip_reason)
            print("VP EpiDive pipeline skipped")
        return {
            "prefix": prefix,
            "skipped": True,
            "skip_reason": skip_reason,
            "highorder_screen": highorder_screen,
            "pair_level": None,
            "output_stage": "skipped",
            "apply_gwes_filter": bool(apply_gwes_filter),
            "final_pairs": empty_pairs,
            "initial_threshold": None,
            "threshold_estimate": None,
            "threshold": None,
            "plot_result": None,
            "weight_ser": None,
            "pattern_map": None,
            "snp_bin": None,
            "representative_matrix": None,
            "counts": {
                "input_loci": int(len(snp_m)),
                "input_samples": int(snp_m.shape[1]),
                "final_pairs": 0,
            },
            "paths": {
                "pattern_map": None,
                "representative_pairs": None,
                "final_pairs": None,
                "gwes_filtered_pairs": None,
                "initial_threshold_pairs": None,
                "initial_ld_pairs": None,
                "chr1_plot": None,
                "chr2_plot": None,
            },
        }

    # --------------------------------------------------------------
    # 1. Convert the SNP matrix and apply MAF filtering
    # --------------------------------------------------------------
    if verbose:
        print("Converting SNP matrix to binary ...")

    snp_bin = ec.convert_snp_to_binary(snp_m)
    snp_bin = snp_bin.astype(
        np.uint8,
        copy=False,
    )

    allele_count = snp_bin.sum(axis=1).astype(np.int64, copy=False)
    sample_count = int(snp_bin.shape[1])
    minor_allele_count = np.minimum(
        allele_count,
        sample_count - allele_count,
    )
    minor_frequency = minor_allele_count / float(sample_count)

    maf_only_mask = (
        minor_frequency.notna()
        & (minor_frequency >= float(maf))
    )
    mac_only_mask = minor_allele_count >= min_mac
    maf_mask = maf_only_mask & mac_only_mask

    input_locus_count = len(snp_m)
    filtered_locus_count = int(maf_mask.sum())
    maf_pass_locus_count = int(maf_only_mask.sum())
    mac_pass_locus_count = int(mac_only_mask.sum())

    snp_filtered = snp_m.loc[maf_mask].copy()
    snp_bin_filtered = snp_bin.loc[maf_mask].copy()

    if snp_filtered.empty:
        raise ValueError(
            "No loci remain after MAF/MAC filtering"
        )

    if verbose:
        print(
            f"Input loci: {input_locus_count:,}"
        )
        print(
            f"Loci passing MAF >= {float(maf):.6g}: "
            f"{maf_pass_locus_count:,}"
        )
        print(
            f"Loci passing MAC >= {min_mac}: "
            f"{mac_pass_locus_count:,}"
        )
        print(
            "Loci removed by combined MAF/MAC filtering: "
            f"{input_locus_count - filtered_locus_count:,}"
        )
        print(
            "Loci retained after MAF/MAC filtering: "
            f"{filtered_locus_count:,}"
        )

    # --------------------------------------------------------------
    # 2. Calculate sample weights
    # --------------------------------------------------------------
    reweight_result = er.run_sample_reweight(
        snp_filtered,
        model=model,
        scc_threshold=scc_threshold,
        hcc_threshold=hcc_threshold,
        prefix=run_prefix + "_reweight",
        n_jobs=reweight_n_jobs,
    )

    weight_ser = reweight_result["snp_wet"]

    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError(
                "snp_wet must contain exactly one column"
            )

        weight_ser = weight_ser.iloc[:, 0]

    weight_ser = weight_ser.reindex(
        snp_filtered.columns
    )

    if weight_ser.isna().any():
        raise ValueError(
            "Sample weights cannot be aligned with "
            "the SNP matrix columns"
        )

    if not np.isfinite(
        weight_ser.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Sample weights contain non-finite values"
        )

    if (weight_ser < 0).any():
        raise ValueError(
            "Sample weights cannot be negative"
        )

    if weight_ser.sum() <= 0:
        raise ValueError(
            "Sample weights must have a positive sum"
        )

    if highorder is not None:
        highorder_screen = classify_subgroup(
            snp_m.shape[1],
            weight_ser=weight_ser,
            original_samples=int(highorder),
            ess_mode=highorder_ess_mode,
            formal_min_ess=highorder_formal_min_ess,
            exploratory_min_ess=highorder_exploratory_min_ess,
            formal_max_se=highorder_formal_max_se,
            exploratory_max_se=highorder_exploratory_max_se,
            formal_min_fraction=highorder_formal_min_fraction,
            exploratory_min_fraction=highorder_exploratory_min_fraction,
        )
        highorder_screen.update({
            "original_samples": int(highorder),
            "subgroup_samples": int(snp_m.shape[1]),
            "subgroup_fraction": snp_m.shape[1] / float(highorder),
            "screen_stage": "weighted",
        })
        if verbose:
            print(
                "High-order weighted ESS: "
                f"{highorder_screen['effective_n']:.3f} "
                f"({highorder_screen['analysis_level']})"
            )
        if not highorder_screen["run_analysis"]:
            empty_pairs = pd.DataFrame(
                columns=["locus1", "locus2", "v"]
            )
            skip_reason = (
                "High-order subgroup weighted effective_n="
                f"{highorder_screen['effective_n']:.3f}, below the "
                "minimum of "
                f"{highorder_screen['exploratory_min_ess']:.3f}"
            )
            if verbose:
                print(skip_reason)
                print("VP EpiDive pipeline skipped before pairwise")
            return {
                "prefix": prefix,
                "skipped": True,
                "skip_reason": skip_reason,
                "highorder_screen": highorder_screen,
                "pair_level": None,
                "output_stage": "skipped",
                "apply_gwes_filter": bool(apply_gwes_filter),
                "final_pairs": empty_pairs,
                "initial_threshold": None,
                "threshold_estimate": None,
                "threshold": None,
                "plot_result": None,
                "weight_ser": weight_ser,
                "pattern_map": None,
                "snp_bin": snp_bin,
                "representative_matrix": None,
                "counts": {
                    "input_loci": int(input_locus_count),
                    "maf_filtered_loci": int(filtered_locus_count),
                    "input_samples": int(snp_m.shape[1]),
                    "final_pairs": 0,
                },
                "paths": {
                    "pattern_map": None,
                    "representative_pairs": None,
                    "final_pairs": None,
                    "gwes_filtered_pairs": None,
                    "initial_threshold_pairs": None,
                    "initial_ld_pairs": None,
                    "chr1_plot": None,
                    "chr2_plot": None,
                },
            }

    # --------------------------------------------------------------
    # 3. Collapse identical genotype patterns
    # --------------------------------------------------------------
    snp_pair, pattern_map = (
        ef.collapse_duplicate_patterns(
            snp_bin_filtered,
            weight_ser=weight_ser,
            map_path=pattern_map_path,
            verbose=verbose,
        )
    )

    unique_pattern_count = len(snp_pair)
    collapsed_locus_count = (
        filtered_locus_count - unique_pattern_count
    )

    if verbose:
        print(
            f"Unique genotype patterns: "
            f"{unique_pattern_count:,}"
        )
        print(
            f"Collapsed loci: "
            f"{collapsed_locus_count:,}"
        )

    del snp_bin_filtered
    gc.collect()

    # --------------------------------------------------------------
    # 4. Select the initial threshold and calculate representative pairs
    # --------------------------------------------------------------
    threshold_estimate = None
    if auto_threshold:
        threshold_estimate = ecf.estimate_adaptive_epidis_threshold(
            data_df=snp_pair,
            weight_ser=weight_ser,
            target_max_pairs=auto_target_max_pairs,
            sample_pairs=auto_sample_pairs,
            min_threshold=auto_min_threshold,
            safety_factor=auto_safety_factor,
            random_state=auto_random_state,
            n_jobs=1,
            verbose=verbose,
        )
        pairwise_threshold = float(threshold_estimate["threshold"])
    else:
        pairwise_threshold = float(threshold)

    if verbose:
        print(
            "Initial EpiDis threshold: "
            f"{pairwise_threshold:.6f}"
        )

    _, egu, gpu_count = _load_gpu_backend()
    if verbose:
        print(f"CUDA GPUs available: {gpu_count}")
    if gpu_stream_output:
        representative_pairs_path = egu.run_epidis_gpu_df(
            snp_df=snp_pair,
            weight_df=weight_ser,
            outlier=pairwise_threshold,
            prefix=run_prefix,
            fmt="parquet",
            block_size=gpu_block_size,
            convert_chunksize=gpu_convert_chunksize,
            convert_n_jobs=gpu_convert_n_jobs,
            maf=None,
            return_df=False,
            verbose=verbose,
        )
        df_pairs = pd.read_parquet(representative_pairs_path)
    else:
        df_pairs = egu.run_epidis_gpu_df(
            snp_df=snp_pair,
            weight_df=weight_ser,
            outlier=pairwise_threshold,
            prefix=run_prefix,
            fmt="parquet",
            block_size=gpu_block_size,
            convert_chunksize=gpu_convert_chunksize,
            convert_n_jobs=gpu_convert_n_jobs,
            maf=None,
            return_df=True,
            verbose=verbose,
        )
        df_pairs.to_parquet(representative_pairs_path, index=False)

    if df_pairs.empty:
        raise ValueError(
            "No representative pairs reached the "
            f"initial EpiDis threshold {pairwise_threshold}"
        )

    representative_pair_count = len(df_pairs)

    if verbose:
        print(
            "Representative pairs: "
            f"{representative_pair_count:,}"
        )
        print(
            "Representative pair file: "
            f"{representative_pairs_path}"
        )

    gc.collect()

    # --------------------------------------------------------------
    # 5. Estimate the GWES threshold
    # --------------------------------------------------------------
    df_pairs_all, v_threshold = (
        eg_vp.run_gwes_vp_full(
            df_input=df_pairs,
            k=k,
            n_jobs=1,
            ld_chr1=ld_chr1,
            ld_chr2=ld_chr2,
            verbose=verbose,
        )
    )

    if not np.isfinite(v_threshold):
        raise ValueError(
            "The calculated GWES threshold is not finite"
        )

    if verbose:
        print(f"GWES IQR level: {k}")
        print(
            "GWES EpiDis threshold: "
            f"{v_threshold:.6f}"
        )

    # --------------------------------------------------------------
    # 6. Optionally restore physical locus pairs
    # --------------------------------------------------------------
    if restore_patterns:
        analysis_pairs = (
            ef.restore_pattern_pairs(
                df_pairs_all,
                pattern_map,
                input_stage="post_gwes",
                include_within_pattern=False,
                output_path=None,
                max_output_rows=max_output_rows,
                verbose=verbose,
            )
        )
        restored_pair_count = len(analysis_pairs)
        analysis_pairs = (
            eg_vp.calculate_pair_distances(
                analysis_pairs,
                n_jobs=distance_n_jobs,
                chunk_size=distance_chunk_size,
                ld_threshold=ld_chr1,
                verbose=verbose,
            )
        )
        pair_level = "physical"
        plot_prefix = restored_plot_prefix
    else:
        analysis_pairs = df_pairs_all
        restored_pair_count = 0
        pair_level = "representative"
        plot_prefix = run_prefix + "_representative"

    del df_pairs_all
    gc.collect()

    # --------------------------------------------------------------
    # 7. Plot the GWES background
    # --------------------------------------------------------------
    plot_result = None
    if make_plots:
        plot_result = eg_vp.plot_gwes_with_threshold(
            df_input=analysis_pairs,
            threshold=v_threshold,
            prefix=plot_prefix,
            ld_chr1=ld_chr1,
            ld_chr2=ld_chr2,
            verbose=verbose,
        )

    # --------------------------------------------------------------
    # 8. Select the requested output stage
    # --------------------------------------------------------------
    if apply_gwes_filter:
        df_pairs_final = (
            eg_vp.filter_gwes_pairs_with_threshold(
                df_input=analysis_pairs,
                threshold=v_threshold,
                ld_chr1=ld_chr1,
                ld_chr2=ld_chr2,
                include_cross_chromosome=(
                    include_cross_chromosome
                ),
                output_path=final_pairs_path,
                verbose=verbose,
            )
        )
        output_stage = "initial_and_gwes"
        output_pairs_path = final_pairs_path
    else:
        df_pairs_final = (
            eg_vp.filter_gwes_pairs_with_threshold(
                df_input=analysis_pairs,
                threshold=-1.0,
                ld_chr1=ld_chr1,
                ld_chr2=ld_chr2,
                include_cross_chromosome=(
                    include_cross_chromosome
                ),
                output_path=initial_output_path,
                verbose=verbose,
            )
        )
        output_stage = "initial_and_ld"
        output_pairs_path = initial_output_path

    final_pair_count = len(df_pairs_final)

    if verbose:
        print()
        print("VP EpiDive pipeline completed")
        print(
            "Representative pairs: "
            f"{representative_pair_count:,}"
        )
        print(
            "Pair level: "
            f"{pair_level}"
        )
        print(
            "Output stage: "
            f"{output_stage}"
        )
        print(
            "Final pairs: "
            f"{final_pair_count:,}"
        )
        print(
            "Final EpiDis threshold: "
            f"{v_threshold:.6f}"
        )
        print(
            f"Output pair file: {output_pairs_path}"
        )

    del analysis_pairs
    del df_pairs
    gc.collect()

    return {
        "prefix": prefix,
        "skipped": False,
        "skip_reason": None,
        "highorder_screen": highorder_screen,
        "pair_level": pair_level,
        "output_stage": output_stage,
        "apply_gwes_filter": bool(apply_gwes_filter),
        "maf": float(maf),
        "min_mac": int(min_mac),
        "final_pairs": df_pairs_final,
        "initial_threshold": float(pairwise_threshold),
        "threshold_estimate": threshold_estimate,
        "threshold": float(v_threshold),
        "plot_result": plot_result,
        "weight_ser": weight_ser,
        "pattern_map": pattern_map,
        "snp_bin": snp_bin,
        "representative_matrix": snp_pair,
        "counts": {
            "input_loci": int(input_locus_count),
            "maf_filtered_loci": int(
                filtered_locus_count
            ),
            "maf_pass_loci": int(
                maf_pass_locus_count
            ),
            "mac_pass_loci": int(
                mac_pass_locus_count
            ),
            "maf_mac_filtered_loci": int(
                filtered_locus_count
            ),
            "unique_patterns": int(
                unique_pattern_count
            ),
            "collapsed_loci": int(
                collapsed_locus_count
            ),
            "representative_pairs": int(
                representative_pair_count
            ),
            "restored_pairs": int(
                restored_pair_count
            ),
            "final_pairs": int(
                final_pair_count
            ),
        },
        "paths": {
            "pattern_map": pattern_map_path,
            "representative_pairs": (
                representative_pairs_path
            ),
            "final_pairs": output_pairs_path,
            "gwes_filtered_pairs": (
                final_pairs_path if apply_gwes_filter else None
            ),
            "initial_threshold_pairs": (
                initial_output_path if not apply_gwes_filter else None
            ),
            "initial_ld_pairs": (
                initial_output_path if not apply_gwes_filter else None
            ),
            "chr1_plot": (
                plot_result["chr1_path"] if plot_result is not None else None
            ),
            "chr2_plot": (
                plot_result["chr2_path"] if plot_result is not None else None
            ),
        },
    }

def _parse_initial_threshold(value):
    """Accept either ``auto`` or a non-negative numeric threshold."""
    if str(value).strip().lower() == "auto":
        return "auto"
    threshold = float(value)
    if threshold < 0:
        raise argparse.ArgumentTypeError("threshold must be non-negative")
    return threshold


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the GPU VP/high-order EpiDive workflow. The input must be "
            "a locus-by-sample SNP matrix accepted by "
            "epidis_reweight.load_snp_matrix."
        )
    )
    parser.add_argument("--input", required=True, help="Input SNP matrix")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for all result files"
    )
    background_group = parser.add_mutually_exclusive_group(required=True)
    background_group.add_argument(
        "--background", type=int,
        help="One background locus used to split samples into allele 1 and 0",
    )
    background_group.add_argument(
        "--background-file",
        help="TSV/CSV file containing multiple background loci",
    )
    parser.add_argument(
        "--background-column",
        default="representative_snp",
        help="Column in --background-file (default: representative_snp)",
    )
    parser.add_argument(
        "--shard-count", type=int, default=1,
        help="Split batch backgrounds across this many servers (default: 1)",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Zero-based server shard index (default: 0)",
    )
    parser.add_argument(
        "--background-limit", type=int,
        help="Process at most this many backgrounds after sharding",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Recompute backgrounds even when a matching _SUCCESS.json exists",
    )
    parser.add_argument("--prefix", default="vp10k", help="Output prefix")
    parser.add_argument("--maf", type=float, default=0.02)
    parser.add_argument(
        "--min-mac",
        type=int,
        default=5,
        help="Minimum minor allele count per locus (default: 5; 0 disables)",
    )
    parser.add_argument("--model", choices=["SCC", "HCC", "NW"], default="SCC")
    parser.add_argument("--scc-threshold", type=float, default=0.98)
    parser.add_argument("--hcc-threshold", type=float, default=0.75)
    parser.add_argument(
        "--threshold", type=_parse_initial_threshold, default=0.3,
        help='Initial full-data threshold: a number or "auto" (default: 0.3)',
    )
    parser.add_argument(
        "--subgroup-threshold", type=_parse_initial_threshold, default="auto",
        help='Initial subgroup threshold (default: "auto")',
    )
    parser.add_argument("--full-k", type=float, default=1.0)
    parser.add_argument("--subgroup-1-k", type=float, default=1.0)
    parser.add_argument("--subgroup-0-k", type=float, default=2.0)
    parser.add_argument(
        "--threshold-method", choices=["auto", "iqr", "quantile"], default="auto"
    )
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-sample-size", type=int, default=100_000)
    parser.add_argument("--min-stability", type=float, default=0.80)
    parser.add_argument(
        "--ess-mode", choices=["absolute", "hybrid"], default="absolute",
        help="Use fixed ESS minima or combine them with total-sample fractions",
    )
    parser.add_argument("--formal-min-fraction", type=float, default=0.05)
    parser.add_argument("--exploratory-min-fraction", type=float, default=0.02)
    parser.add_argument("--selected-pair-gpu", type=int, default=0)
    parser.add_argument("--gpu-block-size", type=int, default=5_000)
    parser.add_argument("--reweight-jobs", type=int, default=-1)
    parser.add_argument("--gpu-convert-jobs", type=int, default=-1)
    parser.add_argument("--distance-jobs", type=int, default=1)
    parser.add_argument("--distance-chunk-size", type=int, default=100_000)
    parser.add_argument("--ld-threshold", type=int, default=5_000)
    parser.add_argument(
        "--apply-gwes-filter", action="store_true",
        help="Apply the calculated GWES threshold to VP pairs (default: disabled)",
    )
    parser.add_argument(
        "--no-restore-final-patterns", action="store_true",
        help="Keep final high-order output at representative-pattern level",
    )
    parser.add_argument(
        "--final-only", action="store_true",
        help="Keep only final filtered background pairs and status markers",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot",
        dest="make_plots",
        action="store_true",
        help="Generate VP and high-order GWES figures (default)",
    )
    plot_group.add_argument(
        "--no-plot",
        dest="make_plots",
        action="store_false",
        help="Disable all VP and high-order figure generation",
    )
    parser.set_defaults(make_plots=True)
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed progress, INFO logs, and stage summaries",
    )
    return parser


def _load_background_ids(args):
    """Load, deduplicate and deterministically shard background loci."""
    if args.background is not None:
        backgrounds = [int(args.background)]
    else:
        suffix = os.path.splitext(args.background_file)[1].lower()
        if suffix in {".tsv", ".tab"}:
            frame = pd.read_csv(args.background_file, sep="\t")
        elif suffix == ".csv":
            frame = pd.read_csv(args.background_file)
        else:
            frame = pd.read_csv(args.background_file, sep=None, engine="python")
        if args.background_column not in frame.columns:
            raise ValueError(
                f"Background file has no column {args.background_column!r}; "
                f"available columns: {list(frame.columns)}"
            )
        values = pd.to_numeric(
            frame[args.background_column], errors="coerce"
        )
        if values.isna().any():
            rows = (values.index[values.isna()] + 2).tolist()[:10]
            raise ValueError(
                "Background column contains missing or non-numeric values; "
                f"first file rows: {rows}"
            )
        backgrounds = list(dict.fromkeys(values.astype(np.int64).tolist()))

    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    if args.background_limit is not None and args.background_limit < 1:
        raise ValueError("--background-limit must be positive")

    backgrounds = [
        background
        for position, background in enumerate(backgrounds)
        if position % args.shard_count == args.shard_index
    ]
    if args.background_limit is not None:
        backgrounds = backgrounds[:args.background_limit]
    if not backgrounds:
        raise ValueError("No backgrounds were selected for this shard")
    return backgrounds


def _analysis_signature(args):
    """Fingerprint result-affecting options for safe resume checks."""
    fields = {
        "input": os.path.abspath(args.input),
        "prefix": args.prefix,
        "maf": args.maf,
        "min_mac": args.min_mac,
        "model": args.model,
        "scc_threshold": args.scc_threshold,
        "hcc_threshold": args.hcc_threshold,
        "threshold": args.threshold,
        "subgroup_threshold": args.subgroup_threshold,
        "full_k": args.full_k,
        "subgroup_1_k": args.subgroup_1_k,
        "subgroup_0_k": args.subgroup_0_k,
        "threshold_method": args.threshold_method,
        "bootstrap": args.bootstrap,
        "bootstrap_sample_size": args.bootstrap_sample_size,
        "min_stability": args.min_stability,
        "apply_gwes_filter": args.apply_gwes_filter,
        "restore_final_patterns": not args.no_restore_final_patterns,
        "ld_threshold": args.ld_threshold,
        "make_plots": args.make_plots,
        "ess_mode": args.ess_mode,
        "formal_min_fraction": args.formal_min_fraction,
        "exploratory_min_fraction": args.exploratory_min_fraction,
        "final_only": args.final_only,
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _full_analysis_signature(args):
    fields = {
        "input": os.path.abspath(args.input),
        "prefix": args.prefix,
        "maf": args.maf,
        "min_mac": args.min_mac,
        "model": args.model,
        "scc_threshold": args.scc_threshold,
        "hcc_threshold": args.hcc_threshold,
        "threshold": args.threshold,
        "full_k": args.full_k,
        "apply_gwes_filter": args.apply_gwes_filter,
        "gpu_block_size": args.gpu_block_size,
        "make_plots": args.make_plots,
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_full_cache(cache_dir, signature):
    manifest_path = os.path.join(cache_dir, "full_cache.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("full_analysis_signature") != signature:
            return None
        paths = {
            name: os.path.join(cache_dir, filename)
            for name, filename in manifest["files"].items()
        }
        if not all(os.path.isfile(path) for path in paths.values()):
            return None
        weight_frame = pd.read_parquet(paths["weight_ser"])
        return {
            "final_pairs": pd.read_parquet(paths["final_pairs"]),
            "snp_bin": pd.read_parquet(paths["snp_bin"]),
            "weight_ser": weight_frame.iloc[:, 0],
            "pattern_map": pd.read_parquet(paths["pattern_map"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_full_cache(full, cache_dir, signature):
    os.makedirs(cache_dir, exist_ok=True)
    files = {
        "final_pairs": "full_cache_final_pairs.parquet",
        "snp_bin": "full_cache_snp_bin.parquet",
        "weight_ser": "full_cache_weight.parquet",
        "pattern_map": "full_cache_pattern_map.parquet",
    }
    full["final_pairs"].to_parquet(
        os.path.join(cache_dir, files["final_pairs"]), index=False
    )
    full["snp_bin"].to_parquet(
        os.path.join(cache_dir, files["snp_bin"]), index=True
    )
    full["weight_ser"].rename("weight").to_frame().to_parquet(
        os.path.join(cache_dir, files["weight_ser"]), index=True
    )
    full["pattern_map"].to_parquet(
        os.path.join(cache_dir, files["pattern_map"]), index=False
    )
    _write_success_marker(
        os.path.join(cache_dir, "full_cache.json"),
        {
            "status": "complete",
            "full_analysis_signature": signature,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": files,
        },
    )


def _matching_success_marker(path, signature):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            marker = json.load(handle)
        return marker.get("analysis_signature") == signature
    except (OSError, ValueError, TypeError):
        return False


def _write_success_marker(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _append_batch_record(path, record):
    columns = [
        "time", "background", "status", "allele_1", "allele_0", "message"
    ]
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({column: record.get(column, "") for column in columns})


def _cleanup_background_intermediates(high, comparison):
    """Remove known per-allele intermediates while preserving the final result."""
    comparison_paths = comparison.get("paths", {}) if comparison else {}
    final_path = comparison_paths.get("filtered")
    candidates = set()
    for path in high.get("paths", {}).values():
        if isinstance(path, str):
            candidates.add(path)
    for key in ("comparison", "representative_filtered", "chr1_plot", "chr2_plot"):
        path = comparison_paths.get(key)
        if isinstance(path, str):
            candidates.add(path)
    removed = []
    for path in sorted(candidates):
        if path == final_path or not os.path.isfile(path):
            continue
        os.remove(path)
        removed.append(path)
    return final_path, removed


def _run_background_analysis(args, snp_df, full, background, output_dir):
    """Run both exact all-pairs subgroup analyses for one background."""
    background_binary = transform_row(snp_df.loc[background])
    mask_1 = (background_binary == 1).to_numpy()
    mask_0 = (background_binary == 0).to_numpy()
    subgroup_counts = {"1": int(mask_1.sum()), "0": int(mask_0.sum())}
    analysis_signature = _analysis_signature(args)

    common = dict(
        maf=args.maf,
        min_mac=args.min_mac,
        model=args.model,
        scc_threshold=args.scc_threshold,
        hcc_threshold=args.hcc_threshold,
        apply_gwes_filter=args.apply_gwes_filter,
        restore_patterns=False,
        reweight_n_jobs=args.reweight_jobs,
        gpu_block_size=args.gpu_block_size,
        gpu_convert_n_jobs=args.gpu_convert_jobs,
        distance_n_jobs=args.distance_jobs,
        distance_chunk_size=args.distance_chunk_size,
        make_plots=args.make_plots,
        verbose=args.verbose,
        highorder_ess_mode=args.ess_mode,
        highorder_formal_min_fraction=args.formal_min_fraction,
        highorder_exploratory_min_fraction=args.exploratory_min_fraction,
    )
    for allele, mask, subgroup_k in (
        (1, mask_1, args.subgroup_1_k),
        (0, mask_0, args.subgroup_0_k),
    ):
        allele_marker = os.path.join(
            output_dir, f"_ALLELE_{allele}_SUCCESS.json"
        )
        if (
            not args.no_resume
            and _matching_success_marker(allele_marker, analysis_signature)
        ):
            if args.verbose:
                print(f"Reusing completed background {background} allele {allele}")
            continue
        subgroup = snp_df.loc[:, mask].copy(deep=False)
        subgroup_prefix = f"{args.prefix}_background_{background}_{allele}"
        high = run_vp_pipeline(
            snp_m=subgroup,
            dir_save=output_dir,
            prefix=subgroup_prefix,
            threshold=args.subgroup_threshold,
            k=subgroup_k,
            highorder=snp_df.shape[1],
            **common,
        )
        comparison = None
        if not high.get("skipped"):
            comparison = run_highorder_comparison_pipeline(
                df_pairs_full=full["final_pairs"],
                df_pairs_high=high["final_pairs"],
                data_full=full["snp_bin"],
                data_high=high["snp_bin"],
                weight_full=full["weight_ser"],
                weight_high=high["weight_ser"],
                dir_save=output_dir,
                prefix=subgroup_prefix,
                pattern_map_full=full["pattern_map"],
                pattern_map_high=high["pattern_map"],
                restore_final_patterns=not args.no_restore_final_patterns,
                threshold_method=args.threshold_method,
                k_upper=1,
                k_lower=2,
                n_bootstrap=args.bootstrap,
                bootstrap_sample_size=args.bootstrap_sample_size,
                min_stability=args.min_stability,
                separate_chromosomes=True,
                selected_pair_gpu_id=args.selected_pair_gpu,
                pair_n_jobs=1,
                distance_n_jobs=args.distance_jobs,
                distance_chunk_size=args.distance_chunk_size,
                ld_threshold=args.ld_threshold,
                save_comparison=not args.final_only,
                save_filtered=True,
                make_plots=args.make_plots,
                verbose=args.verbose,
            )
        final_output = None
        removed_files = []
        if args.final_only:
            final_output, removed_files = _cleanup_background_intermediates(
                high, comparison
            )
        _write_success_marker(allele_marker, {
            "background": int(background),
            "allele": int(allele),
            "status": "complete",
            "analysis_signature": analysis_signature,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": subgroup_counts[str(allele)],
            "skipped": bool(high.get("skipped")),
            "skip_reason": high.get("skip_reason"),
            "highorder_screen": high.get("highorder_screen"),
            "final_only": bool(args.final_only),
            "final_output": final_output,
            "removed_intermediate_files": removed_files,
        })
        del high
        del comparison
        del subgroup
        gc.collect()
    return subgroup_counts


def main(argv=None):
    args = _build_cli_parser().parse_args(argv)
    _configure_console_output(verbose=args.verbose)
    if not 0 <= args.maf <= 0.5:
        raise ValueError("--maf must be between 0 and 0.5")
    if args.min_mac < 0:
        raise ValueError("--min-mac must be non-negative")
    os.makedirs(args.output_dir, exist_ok=True)

    snp_df = er.load_snp_matrix(args.input)
    encoding_start = time.perf_counter()
    snp_df = er.encode_snp_alleles_uint8(snp_df)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Allele encoding cache ready: dtype={snp_df.dtypes.iloc[0]}, "
        f"elapsed={time.perf_counter() - encoding_start:.3f}s",
        flush=True,
    )
    backgrounds = _load_background_ids(args)
    missing = [background for background in backgrounds if background not in snp_df.index]
    if missing and args.background_file is None:
        raise KeyError(f"Background locus {missing[0]} is absent from the input matrix")
    backgrounds = [background for background in backgrounds if background in snp_df.index]
    if not backgrounds:
        raise ValueError("None of the selected backgrounds occur in the SNP matrix")

    batch_mode = args.background_file is not None
    signature = _analysis_signature(args)
    shard_suffix = f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}"
    status_path = os.path.join(args.output_dir, f"batch_status_{shard_suffix}.tsv")
    error_path = os.path.join(args.output_dir, f"batch_errors_{shard_suffix}.tsv")
    for background in missing:
        _append_batch_record(error_path, {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "background": background,
            "status": "missing",
            "message": "Background locus is absent from the input matrix",
        })

    selected_count = len(backgrounds)
    pending_backgrounds = []
    skipped = 0
    for background in backgrounds:
        background_dir = (
            os.path.join(args.output_dir, f"background_{background}")
            if batch_mode else args.output_dir
        )
        marker_path = os.path.join(background_dir, "_SUCCESS.json")
        if not args.no_resume and _matching_success_marker(marker_path, signature):
            skipped += 1
        else:
            pending_backgrounds.append(background)
    backgrounds = pending_backgrounds
    print(
        f"Selected {selected_count:,} backgrounds for shard "
        f"{args.shard_index}/{args.shard_count}; "
        f"pending={len(backgrounds):,}, resumed={skipped:,}"
    )
    if not backgrounds:
        print("All selected backgrounds are already complete")
        return 0

    full_signature = _full_analysis_signature(args)
    full_dir = (
        os.path.join(args.output_dir, "_full", full_signature[:16])
        if batch_mode else args.output_dir
    )
    os.makedirs(full_dir, exist_ok=True)
    full_common = dict(
        dir_save=full_dir,
        maf=args.maf,
        min_mac=args.min_mac,
        model=args.model,
        scc_threshold=args.scc_threshold,
        hcc_threshold=args.hcc_threshold,
        apply_gwes_filter=args.apply_gwes_filter,
        restore_patterns=False,
        reweight_n_jobs=args.reweight_jobs,
        gpu_block_size=args.gpu_block_size,
        gpu_convert_n_jobs=args.gpu_convert_jobs,
        distance_n_jobs=args.distance_jobs,
        distance_chunk_size=args.distance_chunk_size,
        make_plots=args.make_plots,
        verbose=args.verbose,
    )
    full = (
        _load_full_cache(full_dir, full_signature)
        if batch_mode and not args.no_resume else None
    )
    if full is None:
        full = run_vp_pipeline(
            snp_m=snp_df, prefix=args.prefix, threshold=args.threshold,
            k=args.full_k, **full_common,
        )
        if batch_mode:
            _save_full_cache(full, full_dir, full_signature)
    else:
        print(f"Reusing full-data VP cache: {full_dir}")
    completed = 0
    failed = len(missing)
    for position, background in enumerate(backgrounds, start=1):
        background_dir = (
            os.path.join(args.output_dir, f"background_{background}")
            if batch_mode else args.output_dir
        )
        os.makedirs(background_dir, exist_ok=True)
        marker_path = os.path.join(background_dir, "_SUCCESS.json")
        if not args.no_resume and _matching_success_marker(marker_path, signature):
            skipped += 1
            continue
        if args.verbose:
            print(f"Background {position}/{len(backgrounds)}: {background}")
        try:
            subgroup_counts = _run_background_analysis(
                args, snp_df, full, background, background_dir
            )
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            previous_error_path = os.path.join(background_dir, "_ERROR.txt")
            if os.path.isfile(previous_error_path):
                os.remove(previous_error_path)
            _write_success_marker(marker_path, {
                "background": int(background),
                "status": "complete",
                "analysis_signature": signature,
                "completed_at": now,
                "allele_1_samples": subgroup_counts["1"],
                "allele_0_samples": subgroup_counts["0"],
            })
            _append_batch_record(status_path, {
                "time": now,
                "background": background,
                "status": "complete",
                "allele_1": subgroup_counts["1"],
                "allele_0": subgroup_counts["0"],
                "message": "",
            })
            completed += 1
        except Exception as exc:
            failed += 1
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            message = f"{type(exc).__name__}: {exc}"
            _append_batch_record(error_path, {
                "time": now,
                "background": background,
                "status": "failed",
                "message": message,
            })
            with open(
                os.path.join(background_dir, "_ERROR.txt"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(traceback.format_exc())
            logging.error("Background %s failed: %s", background, message)
            if not batch_mode:
                raise
    del full
    print(
        f"Batch finished: completed={completed}, resumed={skipped}, failed={failed}"
    )
    if batch_mode and failed:
        print(f"Failure log: {error_path}")
    return 1 if failed else 0


def _legacy_main_example():
    """Original workstation example retained for reference only."""
    # -----------------------------------------------------------------------------
    # load data
    # This function is not used by the installed command. Supply real paths if
    # copying the example into a notebook; the CLI main() uses --input and
    # --output-dir instead.
    snp_df = er.load_snp_matrix("INPUT_SNP_MATRIX")
    dir_save = "OUTPUT_DIRECTORY"

    # highorder dataset
    # background = 1026343
    # background = 4938638
    background = 7685000

    roww = transform_row(snp_df.loc[background])
    ta = (roww == 1)
    snp_m_1 = snp_df.loc[:, ta.values].copy(deep=False)
    ta = (roww == 0)
    snp_m_0 = snp_df.loc[:, ta.values].copy(deep=False)

    # pairwise
    vp_result = run_vp_pipeline(
        snp_m=snp_df,
        dir_save=dir_save,
        prefix="vp10k",
        maf=0.02,
        model="SCC",
        scc_threshold=0.98,
        threshold=0.3,
        k=1,
        apply_gwes_filter=False,
        restore_patterns=False,
    )
    df_pairs_final = vp_result["final_pairs"]
    v_threshold = vp_result["threshold"]
    pattern_map = vp_result["pattern_map"]


    # highorder 1
    vp_result1 = run_vp_pipeline(
        snp_m=snp_m_1,
        dir_save=dir_save,
        prefix=f"vp10k_background_{background}_1",
        maf=0.02,
        model="SCC",
        scc_threshold=0.98,
        highorder=snp_df.shape[1],
        threshold="auto",
        k=1,
        apply_gwes_filter=False,
        restore_patterns=False,
    )
    df_pairs_final1 = vp_result1["final_pairs"]
    v_threshold1 = vp_result1["threshold"]
    pattern_map1 = vp_result1["pattern_map"]


    comparison_result = run_highorder_comparison_pipeline(
        df_pairs_full=df_pairs_final,
        df_pairs_high=df_pairs_final1,
        data_full=vp_result["snp_bin"],
        data_high=vp_result1["snp_bin"],
        weight_full=vp_result["weight_ser"],
        weight_high=vp_result1["weight_ser"],
        dir_save=dir_save,
        prefix=f"vp10k_background_{background}_1",
        pattern_map_full=vp_result["pattern_map"],
        pattern_map_high=vp_result1["pattern_map"],
        restore_final_patterns=True,
        threshold_method="auto",
        k_upper=1,
        k_lower=2,
        n_bootstrap=200,
        bootstrap_sample_size=100_000,
        min_stability=0.80,
        separate_chromosomes=True,
        pair_n_jobs=1,
        distance_n_jobs=1,
        distance_chunk_size=100_000,
        ld_threshold=5_000,
        save_comparison=True,
        save_filtered=True,
        verbose=True,
    )

    df_merged = comparison_result["merged_pairs"]
    threshold_result = comparison_result["threshold_result"]
    df_highorder_final = comparison_result["filtered_pairs"]
    comparison_prefix = comparison_result["comparison_prefix"]


    # highorder 0
    vp_result0 = run_vp_pipeline(
        snp_m=snp_m_0,
        dir_save=dir_save,
        prefix=f"vp10k_background_{background}_0",
        maf=0.02,
        model="SCC",
        scc_threshold=0.98,
        highorder=snp_df.shape[1],
        threshold="auto",
        k=2,
        apply_gwes_filter=False,
        restore_patterns=False,
    )
    df_pairs_final0 = vp_result0["final_pairs"]
    v_threshold0 = vp_result0["threshold"]
    pattern_map0 = vp_result0["pattern_map"]


    comparison_result = run_highorder_comparison_pipeline(
        df_pairs_full=df_pairs_final,
        df_pairs_high=df_pairs_final0,
        data_full=vp_result["snp_bin"],
        data_high=vp_result0["snp_bin"],
        weight_full=vp_result["weight_ser"],
        weight_high=vp_result0["weight_ser"],
        dir_save=dir_save,
        prefix=f"vp10k_background_{background}_0",
        pattern_map_full=vp_result["pattern_map"],
        pattern_map_high=vp_result0["pattern_map"],
        restore_final_patterns=True,
        threshold_method="auto",
        k_upper=1,
        k_lower=2,
        n_bootstrap=200,
        bootstrap_sample_size=100_000,
        min_stability=0.80,
        separate_chromosomes=True,
        pair_n_jobs=1,
        distance_n_jobs=1,
        distance_chunk_size=100_000,
        ld_threshold=5_000,
        save_comparison=True,
        save_filtered=True,
        verbose=True,
    )

    df_merged = comparison_result["merged_pairs"]
    threshold_result = comparison_result["threshold_result"]
    df_highorder_final = comparison_result["filtered_pairs"]
    comparison_prefix = comparison_result["comparison_prefix"]


if __name__ == "__main__":
    main()
