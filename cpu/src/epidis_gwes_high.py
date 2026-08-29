#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:45:49 2024

@author: rivers_imac

Directional conservative threshold version (2026-08-17):
- Chromosomes are assessed independently for signed bimodality.
- Bimodal data combine IQR and separate-tail quantile Bootstrap thresholds.
- The stricter upper and lower directional thresholds are selected separately.
- Bootstrap sample sizes and plotting behavior remain unchanged.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import gaussian_kde


# -----------------------------------------------------------------------------
# Functions
def gwes_high_from_df(
    df_unipairs1: pd.DataFrame,
    prefix: str,
    flag_chr: str = "chr1",
    save_path: Optional[str] = None,
    outlier_mode: str = "iqr",
    kde_margin: float = 0.1,
    k: float = 2,
    fixed_thresholds: Optional[Tuple[float, float]] = None,
) -> Dict[str, object]:
    """
    Minimal-change callable entrypoint.

    Parameters
    ----------
    df_unipairs : pd.DataFrame
        Required columns: ['v_h', 'v_f', 'geno_dis'].
    prefix : str
        Base filename when saving.
    flag_chr : {'chr1','chr2'}
        Controls figure width.
    save_path : str | None
        If provided, used as base save path; otherwise use `prefix`.
    outlier_mode : {'iqr','kde'}
    kde_margin : float

    Returns
    -------
    Dict[str, object]
        {
          "mask_strengthen": pd.Series(bool),
          "mask_weaken": pd.Series(bool),
          "df_strengthen": pd.DataFrame,   # rows where diff > upper
          "df_weaken": pd.DataFrame,       # rows where diff < lower
          "thresholds": {"upper": float, "lower": float},
          "figure_path": str
        }
    """


    # Ensure numeric
    for col in ["v_h", "v_f", "geno_dis"]:
        if col not in df_unipairs1.columns:
            raise KeyError(f"Missing required column: {col}")
        df_unipairs1[col] = pd.to_numeric(df_unipairs1[col], errors="coerce")

    # diff & filter
    df_unipairs1["diff"] = df_unipairs1["v_h"] - df_unipairs1["v_f"]
    df_unipairs = df_unipairs1.copy()
    # df_unipairs = df_unipairs[df_unipairs["geno_dis"] >= 0]

    out_base = save_path if save_path is not None else prefix

    # Compute thresholds and plot (returns (upper, lower))
    upper, lower = plot_GWES_Diff_pure(
        df_unipairs, flag_chr, out_base,
        outlier_mode=outlier_mode,
        kde_margin=kde_margin,
        k=k,
        fixed_thresholds=fixed_thresholds,
    )

    # upper, lower = plot_GWES_Diff(
    #     df_unipairs, flag_chr, out_base,
    #     outlier_mode=outlier_mode, kde_margin=kde_margin, k=k
    # )

    # Boolean masks on diff
    if np.isfinite(upper):
        mask_strengthen = df_unipairs1["diff"] > upper
    else:
        mask_strengthen = pd.Series(False, index=df_unipairs1.index)

    if np.isfinite(lower):
        mask_weaken = df_unipairs1["diff"] < lower
    else:
        mask_weaken = pd.Series(False, index=df_unipairs1.index)

    df_strengthen = df_unipairs1.loc[mask_strengthen]
    df_weaken = df_unipairs1.loc[mask_weaken]

    return {
        "mask_strengthen": mask_strengthen,
        "mask_weaken": mask_weaken,
        "df_strengthen": df_strengthen,
        "df_weaken": df_weaken,
        "thresholds": {"upper": float(upper), "lower": float(lower)},
        "figure_path": f"{out_base}.GWES_HighOrder_diff.png",
    }


def run_highorder_threshold_analysis(
    df_merged: pd.DataFrame,
    prefix: str,
    *,
    k: float = 2.0,
    min_abs_threshold: float = 0.15,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    save_filtered: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Estimate one global IQR threshold pair, filter, split, and plot."""
    required = {"locus1", "locus2", "v_f", "v_h", "geno_dis"}
    missing = sorted(required.difference(df_merged.columns))
    if missing:
        raise ValueError(f"df_merged is missing columns: {missing}")
    if k < 0:
        raise ValueError("k cannot be negative")
    if min_abs_threshold < 0:
        raise ValueError("min_abs_threshold cannot be negative")

    data = df_merged.copy()
    data["v_f"] = pd.to_numeric(data["v_f"], errors="coerce")
    data["v_h"] = pd.to_numeric(data["v_h"], errors="coerce")
    data["geno_dis"] = pd.to_numeric(data["geno_dis"], errors="coerce")
    data["diff"] = data["v_h"] - data["v_f"]

    finite_diff = data.loc[np.isfinite(data["diff"]), "diff"]
    if finite_diff.empty:
        raise ValueError("df_merged contains no finite EpiDis differences")

    q1 = float(finite_diff.quantile(0.25))
    q3 = float(finite_diff.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError("The EpiDis difference IQR must be positive")

    fence_width = 1.5 * float(k) * iqr
    upper = max(q3 + fence_width, float(min_abs_threshold))
    lower = min(q1 - fence_width, -float(min_abs_threshold))

    strengthen = data.loc[data["diff"] > upper].copy()
    weaken = data.loc[data["diff"] < lower].copy()
    filtered = data.loc[
        (data["diff"] > upper) | (data["diff"] < lower)
    ].copy()

    chr1_mask = (
        ((data["locus1"] < chr1_len) | (data["locus2"] < chr1_len))
        & (data["geno_dis"] > 0)
    )
    chr2_end = chr1_len + chr2_len
    chr2_mask = (
        (data["locus1"] > chr1_len)
        & (data["locus2"] > chr1_len)
        & ((data["locus1"] < chr2_end) | (data["locus2"] < chr2_end))
        & (data["geno_dis"] > 0)
    )
    df_chr1 = data.loc[chr1_mask].copy()
    df_chr2 = data.loc[chr2_mask].copy()

    filtered_chr1 = filtered.loc[filtered.index.intersection(df_chr1.index)].copy()
    filtered_chr2 = filtered.loc[filtered.index.intersection(df_chr2.index)].copy()

    plot_chr1 = None
    plot_chr2 = None
    if make_plots:
        plot_chr1 = gwes_high_from_df(
            df_chr1,
            prefix=prefix + "_chr1",
            flag_chr="chr1",
            fixed_thresholds=(upper, lower),
        )
        plot_chr2 = gwes_high_from_df(
            df_chr2,
            prefix=prefix + "_chr2",
            flag_chr="chr2",
            fixed_thresholds=(upper, lower),
        )

    filtered_path = None
    if save_filtered:
        filtered_path = prefix + "_filtered_pairs.parquet"
        filtered.to_parquet(filtered_path, index=False)

    if verbose:
        print(f"Q1: {q1:.6f}")
        print(f"Q3: {q3:.6f}")
        print(f"IQR: {iqr:.6f}")
        print(f"Upper threshold: {upper:.6f}")
        print(f"Lower threshold: {lower:.6f}")
        print(f"Strengthened pairs: {len(strengthen):,}")
        print(f"Weakened pairs: {len(weaken):,}")
        print(f"Selected pairs: {len(filtered):,}")

    return {
        "thresholds": {
            "upper": float(upper),
            "lower": float(lower),
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "k": float(k),
        },
        "filtered_pairs": filtered,
        "strengthened_pairs": strengthen,
        "weakened_pairs": weaken,
        "df_chr1": df_chr1,
        "df_chr2": df_chr2,
        "filtered_chr1": filtered_chr1,
        "filtered_chr2": filtered_chr2,
        "plot_chr1": plot_chr1,
        "plot_chr2": plot_chr2,
        "filtered_path": filtered_path,
    }


def run_highorder_threshold_analysis_bootstrap(
    df_merged: pd.DataFrame,
    prefix: str,
    *,
    k_upper: float = 1.0,
    k_lower: float = 2.0,
    n_bootstrap: int = 200,
    bootstrap_sample_size: int = 100_000,
    min_stability: float = 0.80,
    min_abs_threshold: float = 0.15,
    zero_anchor_iqr: bool = True,
    random_state: int = 2026,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    save_filtered: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
    separate_chromosomes: bool = False,
) -> Dict[str, object]:
    """Estimate stable IQR thresholds using a fast row-bootstrap."""
    if separate_chromosomes:
        return run_highorder_threshold_analysis_bootstrap_by_chromosome(
            df_merged=df_merged,
            prefix=prefix,
            k_upper=k_upper,
            k_lower=k_lower,
            n_bootstrap=n_bootstrap,
            bootstrap_sample_size=bootstrap_sample_size,
            min_stability=min_stability,
            min_abs_threshold=min_abs_threshold,
            zero_anchor_iqr=zero_anchor_iqr,
            random_state=random_state,
            chr1_len=chr1_len,
            chr2_len=chr2_len,
            save_filtered=save_filtered,
            make_plots=make_plots,
            verbose=verbose,
        )
    required = {"locus1", "locus2", "v_f", "v_h", "geno_dis"}
    missing = sorted(required.difference(df_merged.columns))
    if missing:
        raise ValueError(f"df_merged is missing columns: {missing}")
    if k_upper < 0 or k_lower < 0:
        raise ValueError("k_upper and k_lower cannot be negative")
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if bootstrap_sample_size < 1_000:
        raise ValueError("bootstrap_sample_size must be at least 1000")
    if not 0.5 <= min_stability < 1.0:
        raise ValueError("min_stability must be in [0.5, 1.0)")
    if min_abs_threshold < 0:
        raise ValueError("min_abs_threshold cannot be negative")

    data = df_merged.copy()
    data["v_f"] = pd.to_numeric(data["v_f"], errors="coerce")
    data["v_h"] = pd.to_numeric(data["v_h"], errors="coerce")
    data["geno_dis"] = pd.to_numeric(data["geno_dis"], errors="coerce")
    data["diff"] = data["v_h"] - data["v_f"]

    finite_mask = np.isfinite(data["diff"].to_numpy(dtype=np.float64))
    values = data.loc[finite_mask, "diff"].to_numpy(dtype=np.float64)
    if values.size < 1_000:
        raise ValueError("At least 1000 finite EpiDis differences are required")

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError("The EpiDis difference IQR must be positive")

    upper_base = max(q3, 0.0) if zero_anchor_iqr else q3
    lower_base = min(q1, 0.0) if zero_anchor_iqr else q1
    raw_upper = max(
        upper_base + 1.5 * float(k_upper) * iqr,
        min_abs_threshold,
    )
    raw_lower = min(
        lower_base - 1.5 * float(k_lower) * iqr,
        -min_abs_threshold,
    )

    sample_size = min(int(bootstrap_sample_size), values.size)
    rng = np.random.default_rng(random_state)
    upper_boot = np.empty(n_bootstrap, dtype=np.float64)
    lower_boot = np.empty(n_bootstrap, dtype=np.float64)

    for bootstrap_index in range(n_bootstrap):
        sample = rng.choice(values, size=sample_size, replace=True)
        sample_q1, sample_q3 = np.quantile(sample, [0.25, 0.75])
        sample_iqr = sample_q3 - sample_q1
        sample_upper_base = (
            max(sample_q3, 0.0) if zero_anchor_iqr else sample_q3
        )
        sample_lower_base = (
            min(sample_q1, 0.0) if zero_anchor_iqr else sample_q1
        )
        upper_boot[bootstrap_index] = max(
            sample_upper_base + 1.5 * float(k_upper) * sample_iqr,
            min_abs_threshold,
        )
        lower_boot[bootstrap_index] = min(
            sample_lower_base - 1.5 * float(k_lower) * sample_iqr,
            -min_abs_threshold,
        )

    median_upper = float(np.median(upper_boot))
    median_lower = float(np.median(lower_boot))
    effective_upper = float(np.quantile(upper_boot, min_stability))
    effective_lower = float(np.quantile(lower_boot, 1.0 - min_stability))

    sorted_upper = np.sort(upper_boot)
    sorted_lower = np.sort(lower_boot)
    diff_values = data["diff"].to_numpy(dtype=np.float64)

    strengthen_stability = np.searchsorted(
        sorted_upper,
        diff_values,
        side="left",
    ) / float(n_bootstrap)
    weaken_stability = (
        n_bootstrap
        - np.searchsorted(sorted_lower, diff_values, side="right")
    ) / float(n_bootstrap)
    strengthen_stability[~finite_mask] = 0.0
    weaken_stability[~finite_mask] = 0.0

    data["strengthen_stability"] = strengthen_stability
    data["weaken_stability"] = weaken_stability
    data["threshold_stability"] = np.maximum(
        strengthen_stability,
        weaken_stability,
    )

    strengthen_mask = (
        (data["diff"] > effective_upper)
        & (data["strengthen_stability"] >= min_stability)
    )
    weaken_mask = (
        (data["diff"] < effective_lower)
        & (data["weaken_stability"] >= min_stability)
    )
    data["change_direction"] = "not_selected"
    data.loc[strengthen_mask, "change_direction"] = "strengthened"
    data.loc[weaken_mask, "change_direction"] = "weakened"

    strengthen = data.loc[strengthen_mask].copy()
    weaken = data.loc[weaken_mask].copy()
    filtered = data.loc[strengthen_mask | weaken_mask].copy()

    chr2_end = chr1_len + chr2_len
    chr1_mask = (
        ((data["locus1"] < chr1_len) | (data["locus2"] < chr1_len))
        & (data["geno_dis"] > 0)
    )
    chr2_mask = (
        (data["locus1"] > chr1_len)
        & (data["locus2"] > chr1_len)
        & ((data["locus1"] < chr2_end) | (data["locus2"] < chr2_end))
        & (data["geno_dis"] > 0)
    )
    df_chr1 = data.loc[chr1_mask].copy()
    df_chr2 = data.loc[chr2_mask].copy()
    filtered_chr1 = filtered.loc[filtered.index.intersection(df_chr1.index)].copy()
    filtered_chr2 = filtered.loc[filtered.index.intersection(df_chr2.index)].copy()

    plot_chr1 = None
    plot_chr2 = None
    if make_plots:
        plot_chr1 = gwes_high_from_df(
            df_chr1,
            prefix=prefix + "_bootstrap_chr1",
            flag_chr="chr1",
            fixed_thresholds=(effective_upper, effective_lower),
        )
        plot_chr2 = gwes_high_from_df(
            df_chr2,
            prefix=prefix + "_bootstrap_chr2",
            flag_chr="chr2",
            fixed_thresholds=(effective_upper, effective_lower),
        )

    filtered_path = None
    if save_filtered:
        filtered_path = prefix + "_bootstrap_filtered_pairs.parquet"
        filtered.to_parquet(filtered_path, index=False)

    upper_ci = np.quantile(upper_boot, [0.025, 0.975]).astype(float)
    lower_ci = np.quantile(lower_boot, [0.025, 0.975]).astype(float)

    if verbose:
        print(f"Raw upper threshold: {raw_upper:.6f}")
        print(f"Raw lower threshold: {raw_lower:.6f}")
        print(f"Bootstrap median upper: {median_upper:.6f}")
        print(f"Bootstrap median lower: {median_lower:.6f}")
        print(f"Stable upper threshold: {effective_upper:.6f}")
        print(f"Stable lower threshold: {effective_lower:.6f}")
        print(
            f"Upper 95% interval: [{upper_ci[0]:.6f}, {upper_ci[1]:.6f}]"
        )
        print(
            f"Lower 95% interval: [{lower_ci[0]:.6f}, {lower_ci[1]:.6f}]"
        )
        print(f"Strengthened pairs: {len(strengthen):,}")
        print(f"Weakened pairs: {len(weaken):,}")
        print(f"Selected pairs: {len(filtered):,}")

    return {
        "thresholds": {
            "raw_upper": float(raw_upper),
            "raw_lower": float(raw_lower),
            "bootstrap_median_upper": median_upper,
            "bootstrap_median_lower": median_lower,
            "stable_upper": effective_upper,
            "stable_lower": effective_lower,
            "upper_ci95": upper_ci.tolist(),
            "lower_ci95": lower_ci.tolist(),
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "upper_base": float(upper_base),
            "lower_base": float(lower_base),
            "zero_anchor_iqr": bool(zero_anchor_iqr),
            "k_upper": float(k_upper),
            "k_lower": float(k_lower),
            "min_stability": float(min_stability),
        },
        "filtered_pairs": filtered,
        "strengthened_pairs": strengthen,
        "weakened_pairs": weaken,
        "df_chr1": df_chr1,
        "df_chr2": df_chr2,
        "filtered_chr1": filtered_chr1,
        "filtered_chr2": filtered_chr2,
        "plot_chr1": plot_chr1,
        "plot_chr2": plot_chr2,
        "filtered_path": filtered_path,
    }


def _bootstrap_iqr_threshold_subset(
    data: pd.DataFrame,
    *,
    k_upper: float,
    k_lower: float,
    n_bootstrap: int,
    bootstrap_sample_size: int,
    min_stability: float,
    min_abs_threshold: float,
    zero_anchor_iqr: bool,
    random_state: int,
) -> Dict[str, object]:
    values = data["diff"].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size < 1_000:
        raise ValueError(
            "Each chromosome requires at least 1000 finite EpiDis differences"
        )

    q1, q3 = np.quantile(finite_values, [0.25, 0.75]).astype(float)
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError("The chromosome-specific EpiDis IQR must be positive")

    upper_base = max(q3, 0.0) if zero_anchor_iqr else q3
    lower_base = min(q1, 0.0) if zero_anchor_iqr else q1
    raw_upper = max(
        upper_base + 1.5 * k_upper * iqr,
        min_abs_threshold,
    )
    raw_lower = min(
        lower_base - 1.5 * k_lower * iqr,
        -min_abs_threshold,
    )
    sample_size = min(int(bootstrap_sample_size), finite_values.size)
    rng = np.random.default_rng(random_state)
    upper_boot = np.empty(n_bootstrap, dtype=np.float64)
    lower_boot = np.empty(n_bootstrap, dtype=np.float64)

    for bootstrap_index in range(n_bootstrap):
        sample = rng.choice(finite_values, size=sample_size, replace=True)
        sample_q1, sample_q3 = np.quantile(sample, [0.25, 0.75])
        sample_iqr = sample_q3 - sample_q1
        sample_upper_base = (
            max(sample_q3, 0.0) if zero_anchor_iqr else sample_q3
        )
        sample_lower_base = (
            min(sample_q1, 0.0) if zero_anchor_iqr else sample_q1
        )
        upper_boot[bootstrap_index] = max(
            sample_upper_base + 1.5 * k_upper * sample_iqr,
            min_abs_threshold,
        )
        lower_boot[bootstrap_index] = min(
            sample_lower_base - 1.5 * k_lower * sample_iqr,
            -min_abs_threshold,
        )

    stable_upper = float(np.quantile(upper_boot, min_stability))
    stable_lower = float(np.quantile(lower_boot, 1.0 - min_stability))
    sorted_upper = np.sort(upper_boot)
    sorted_lower = np.sort(lower_boot)
    strengthen_stability = np.searchsorted(
        sorted_upper,
        values,
        side="left",
    ) / float(n_bootstrap)
    weaken_stability = (
        n_bootstrap
        - np.searchsorted(sorted_lower, values, side="right")
    ) / float(n_bootstrap)
    strengthen_stability[~finite_mask] = 0.0
    weaken_stability[~finite_mask] = 0.0

    annotated = data.copy()
    annotated["strengthen_stability"] = strengthen_stability
    annotated["weaken_stability"] = weaken_stability
    annotated["threshold_stability"] = np.maximum(
        strengthen_stability,
        weaken_stability,
    )
    strengthen_mask = (
        (annotated["diff"] > stable_upper)
        & (annotated["strengthen_stability"] >= min_stability)
    )
    weaken_mask = (
        (annotated["diff"] < stable_lower)
        & (annotated["weaken_stability"] >= min_stability)
    )
    annotated["change_direction"] = "not_selected"
    annotated.loc[strengthen_mask, "change_direction"] = "strengthened"
    annotated.loc[weaken_mask, "change_direction"] = "weakened"

    upper_ci = np.quantile(upper_boot, [0.025, 0.975]).astype(float)
    lower_ci = np.quantile(lower_boot, [0.025, 0.975]).astype(float)
    return {
        "data": annotated,
        "strengthened": annotated.loc[strengthen_mask].copy(),
        "weakened": annotated.loc[weaken_mask].copy(),
        "filtered": annotated.loc[strengthen_mask | weaken_mask].copy(),
        "thresholds": {
            "raw_upper": float(raw_upper),
            "raw_lower": float(raw_lower),
            "bootstrap_median_upper": float(np.median(upper_boot)),
            "bootstrap_median_lower": float(np.median(lower_boot)),
            "stable_upper": stable_upper,
            "stable_lower": stable_lower,
            "upper_ci95": upper_ci.tolist(),
            "lower_ci95": lower_ci.tolist(),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": iqr,
            "upper_base": float(upper_base),
            "lower_base": float(lower_base),
            "zero_anchor_iqr": bool(zero_anchor_iqr),
            "k_upper": float(k_upper),
            "k_lower": float(k_lower),
            "min_stability": float(min_stability),
        },
    }


def run_highorder_threshold_analysis_bootstrap_by_chromosome(
    df_merged: pd.DataFrame,
    prefix: str,
    *,
    k_upper: float = 1.0,
    k_lower: float = 2.0,
    n_bootstrap: int = 200,
    bootstrap_sample_size: int = 100_000,
    min_stability: float = 0.80,
    min_abs_threshold: float = 0.15,
    zero_anchor_iqr: bool = True,
    random_state: int = 2026,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    save_filtered: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Estimate independent stable thresholds for chromosome 1 and 2."""
    required = {"locus1", "locus2", "v_f", "v_h", "geno_dis"}
    missing = sorted(required.difference(df_merged.columns))
    if missing:
        raise ValueError(f"df_merged is missing columns: {missing}")
    if k_upper < 0 or k_lower < 0:
        raise ValueError("k_upper and k_lower cannot be negative")
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if bootstrap_sample_size < 1_000:
        raise ValueError("bootstrap_sample_size must be at least 1000")
    if not 0.5 <= min_stability < 1.0:
        raise ValueError("min_stability must be in [0.5, 1.0)")

    data = df_merged.copy()
    data["v_f"] = pd.to_numeric(data["v_f"], errors="coerce")
    data["v_h"] = pd.to_numeric(data["v_h"], errors="coerce")
    data["geno_dis"] = pd.to_numeric(data["geno_dis"], errors="coerce")
    data["diff"] = data["v_h"] - data["v_f"]

    chr2_end = chr1_len + chr2_len
    chr1_mask = (
        ((data["locus1"] < chr1_len) | (data["locus2"] < chr1_len))
        & (data["geno_dis"] > 0)
    )
    chr2_mask = (
        (data["locus1"] > chr1_len)
        & (data["locus2"] > chr1_len)
        & ((data["locus1"] < chr2_end) | (data["locus2"] < chr2_end))
        & (data["geno_dis"] > 0)
    )
    chr1 = data.loc[chr1_mask].copy()
    chr2 = data.loc[chr2_mask].copy()

    chr1_result = _bootstrap_iqr_threshold_subset(
        chr1,
        k_upper=float(k_upper),
        k_lower=float(k_lower),
        n_bootstrap=n_bootstrap,
        bootstrap_sample_size=bootstrap_sample_size,
        min_stability=min_stability,
        min_abs_threshold=min_abs_threshold,
        zero_anchor_iqr=zero_anchor_iqr,
        random_state=random_state,
    )
    chr2_result = _bootstrap_iqr_threshold_subset(
        chr2,
        k_upper=float(k_upper),
        k_lower=float(k_lower),
        n_bootstrap=n_bootstrap,
        bootstrap_sample_size=bootstrap_sample_size,
        min_stability=min_stability,
        min_abs_threshold=min_abs_threshold,
        zero_anchor_iqr=zero_anchor_iqr,
        random_state=random_state + 1,
    )

    chr1_thresholds = chr1_result["thresholds"]
    chr2_thresholds = chr2_result["thresholds"]
    plot_chr1 = None
    plot_chr2 = None
    if make_plots:
        plot_chr1 = gwes_high_from_df(
            chr1_result["data"],
            prefix=prefix + "_bootstrap_chr1",
            flag_chr="chr1",
            fixed_thresholds=(
                chr1_thresholds["stable_upper"],
                chr1_thresholds["stable_lower"],
            ),
        )
        plot_chr2 = gwes_high_from_df(
            chr2_result["data"],
            prefix=prefix + "_bootstrap_chr2",
            flag_chr="chr2",
            fixed_thresholds=(
                chr2_thresholds["stable_upper"],
                chr2_thresholds["stable_lower"],
            ),
        )

    filtered = pd.concat(
        [chr1_result["filtered"], chr2_result["filtered"]],
        ignore_index=True,
    )
    strengthened = pd.concat(
        [chr1_result["strengthened"], chr2_result["strengthened"]],
        ignore_index=True,
    )
    weakened = pd.concat(
        [chr1_result["weakened"], chr2_result["weakened"]],
        ignore_index=True,
    )

    filtered_path = None
    if save_filtered:
        filtered_path = prefix + "_bootstrap_filtered_pairs.parquet"
        filtered.to_parquet(filtered_path, index=False)

    if verbose:
        for chromosome, result in (("chr1", chr1_result), ("chr2", chr2_result)):
            thresholds = result["thresholds"]
            print(f"{chromosome} stable upper: {thresholds['stable_upper']:.6f}")
            print(f"{chromosome} stable lower: {thresholds['stable_lower']:.6f}")
            print(
                f"{chromosome} strengthened pairs: "
                f"{len(result['strengthened']):,}"
            )
            print(
                f"{chromosome} weakened pairs: {len(result['weakened']):,}"
            )
        print(f"Selected pairs: {len(filtered):,}")

    return {
        "threshold_scope": "chromosome",
        "thresholds": {
            "chr1": chr1_thresholds,
            "chr2": chr2_thresholds,
        },
        "filtered_pairs": filtered,
        "strengthened_pairs": strengthened,
        "weakened_pairs": weakened,
        "df_chr1": chr1_result["data"],
        "df_chr2": chr2_result["data"],
        "filtered_chr1": chr1_result["filtered"],
        "filtered_chr2": chr2_result["filtered"],
        "plot_chr1": plot_chr1,
        "plot_chr2": plot_chr2,
        "filtered_path": filtered_path,
    }


# Backward-compatible aliases for notebooks using the previous test names.
run_highorder_threshold_analysis_test = (
    run_highorder_threshold_analysis_bootstrap
)
run_highorder_threshold_analysis_test_by_chromosome = (
    run_highorder_threshold_analysis_bootstrap_by_chromosome
)

def _bootstrap_tail_quantile_subset(
    data: pd.DataFrame,
    *,
    upper_quantile: float,
    lower_quantile: float,
    n_bootstrap: int,
    bootstrap_sample_size: int,
    min_stability: float,
    random_state: int,
    min_tail_pairs: int,
) -> Dict[str, object]:
    values = data["diff"].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    negative = finite[finite < 0]
    if positive.size < min_tail_pairs:
        raise ValueError(
            f"Positive tail has {positive.size} pairs; "
            f"at least {min_tail_pairs} are required"
        )
    if negative.size < min_tail_pairs:
        raise ValueError(
            f"Negative tail has {negative.size} pairs; "
            f"at least {min_tail_pairs} are required"
        )

    raw_upper = float(np.quantile(positive, upper_quantile))
    raw_lower = float(np.quantile(negative, lower_quantile))
    positive_sample_size = min(bootstrap_sample_size, positive.size)
    negative_sample_size = min(bootstrap_sample_size, negative.size)
    rng = np.random.default_rng(random_state)
    upper_boot = np.empty(n_bootstrap, dtype=np.float64)
    lower_boot = np.empty(n_bootstrap, dtype=np.float64)

    for bootstrap_index in range(n_bootstrap):
        positive_sample = rng.choice(
            positive,
            size=positive_sample_size,
            replace=True,
        )
        negative_sample = rng.choice(
            negative,
            size=negative_sample_size,
            replace=True,
        )
        upper_boot[bootstrap_index] = np.quantile(
            positive_sample,
            upper_quantile,
        )
        lower_boot[bootstrap_index] = np.quantile(
            negative_sample,
            lower_quantile,
        )

    stable_upper = float(np.quantile(upper_boot, min_stability))
    stable_lower = float(np.quantile(lower_boot, 1.0 - min_stability))
    sorted_upper = np.sort(upper_boot)
    sorted_lower = np.sort(lower_boot)
    strengthen_stability = np.searchsorted(
        sorted_upper,
        values,
        side="left",
    ) / float(n_bootstrap)
    weaken_stability = (
        n_bootstrap
        - np.searchsorted(sorted_lower, values, side="right")
    ) / float(n_bootstrap)
    finite_mask = np.isfinite(values)
    strengthen_stability[~finite_mask] = 0.0
    weaken_stability[~finite_mask] = 0.0

    annotated = data.copy()
    annotated["strengthen_stability"] = strengthen_stability
    annotated["weaken_stability"] = weaken_stability
    annotated["threshold_stability"] = np.maximum(
        strengthen_stability,
        weaken_stability,
    )
    strengthen_mask = (
        (annotated["diff"] >= stable_upper)
        & (annotated["strengthen_stability"] >= min_stability)
    )
    weaken_mask = (
        (annotated["diff"] <= stable_lower)
        & (annotated["weaken_stability"] >= min_stability)
    )
    annotated["change_direction"] = "not_selected"
    annotated.loc[strengthen_mask, "change_direction"] = "strengthened"
    annotated.loc[weaken_mask, "change_direction"] = "weakened"

    return {
        "data": annotated,
        "filtered": annotated.loc[strengthen_mask | weaken_mask].copy(),
        "strengthened": annotated.loc[strengthen_mask].copy(),
        "weakened": annotated.loc[weaken_mask].copy(),
        "thresholds": {
            "method": "separate_tail_quantile_bootstrap",
            "raw_upper": raw_upper,
            "raw_lower": raw_lower,
            "bootstrap_median_upper": float(np.median(upper_boot)),
            "bootstrap_median_lower": float(np.median(lower_boot)),
            "stable_upper": stable_upper,
            "stable_lower": stable_lower,
            "upper_ci95": np.quantile(
                upper_boot, [0.025, 0.975]
            ).astype(float).tolist(),
            "lower_ci95": np.quantile(
                lower_boot, [0.025, 0.975]
            ).astype(float).tolist(),
            "upper_quantile": float(upper_quantile),
            "lower_quantile": float(lower_quantile),
            "min_stability": float(min_stability),
            "positive_tail_pairs": int(positive.size),
            "negative_tail_pairs": int(negative.size),
        },
    }


def _combine_directional_conservative_analyses(
    iqr_analysis: Dict[str, object],
    quantile_analysis: Dict[str, object],
    *,
    min_stability: float,
) -> Dict[str, object]:
    """Use the stricter IQR/quantile threshold independently per tail."""
    iqr_thresholds = iqr_analysis["thresholds"]
    quantile_thresholds = quantile_analysis["thresholds"]
    iqr_upper = float(iqr_thresholds["stable_upper"])
    iqr_lower = float(iqr_thresholds["stable_lower"])
    quantile_upper = float(quantile_thresholds["stable_upper"])
    quantile_lower = float(quantile_thresholds["stable_lower"])

    diff_values = pd.to_numeric(
        iqr_analysis["data"]["diff"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    finite_diff = diff_values[np.isfinite(diff_values)]
    if finite_diff.size == 0:
        raise ValueError("No finite EpiDis differences are available")
    observed_upper = float(finite_diff.max())
    observed_lower = float(finite_diff.min())

    # Strict mode: an extrapolated IQR fence remains authoritative. If it is
    # outside the observed/theoretical range, that direction deliberately
    # produces zero discoveries instead of falling back to a quantile fence.
    iqr_upper_usable = (
        np.isfinite(iqr_upper)
        and iqr_upper <= observed_upper
        and iqr_upper <= 1.0
    )
    iqr_lower_usable = (
        np.isfinite(iqr_lower)
        and iqr_lower >= observed_lower
        and iqr_lower >= -1.0
    )

    if not iqr_upper_usable:
        stable_upper = iqr_upper
        upper_source = "iqr_bootstrap_out_of_range_empty"
        upper_analysis = iqr_analysis
    elif iqr_upper >= quantile_upper:
        stable_upper = iqr_upper
        upper_source = "iqr_bootstrap"
        upper_analysis = iqr_analysis
    else:
        stable_upper = quantile_upper
        upper_source = "separate_tail_quantile_bootstrap"
        upper_analysis = quantile_analysis

    if not iqr_lower_usable:
        stable_lower = iqr_lower
        lower_source = "iqr_bootstrap_out_of_range_empty"
        lower_analysis = iqr_analysis
    elif iqr_lower <= quantile_lower:
        stable_lower = iqr_lower
        lower_source = "iqr_bootstrap"
        lower_analysis = iqr_analysis
    else:
        stable_lower = quantile_lower
        lower_source = "separate_tail_quantile_bootstrap"
        lower_analysis = quantile_analysis

    annotated = iqr_analysis["data"].copy()
    annotated["strengthen_stability"] = upper_analysis["data"][
        "strengthen_stability"
    ].to_numpy(copy=False)
    annotated["weaken_stability"] = lower_analysis["data"][
        "weaken_stability"
    ].to_numpy(copy=False)
    annotated["threshold_stability"] = np.maximum(
        annotated["strengthen_stability"].to_numpy(),
        annotated["weaken_stability"].to_numpy(),
    )
    strengthen_mask = (
        (annotated["diff"] >= stable_upper)
        & (annotated["strengthen_stability"] >= min_stability)
    )
    weaken_mask = (
        (annotated["diff"] <= stable_lower)
        & (annotated["weaken_stability"] >= min_stability)
    )
    annotated["change_direction"] = "not_selected"
    annotated.loc[strengthen_mask, "change_direction"] = "strengthened"
    annotated.loc[weaken_mask, "change_direction"] = "weakened"

    return {
        "data": annotated,
        "filtered": annotated.loc[
            strengthen_mask | weaken_mask
        ].copy(),
        "strengthened": annotated.loc[strengthen_mask].copy(),
        "weakened": annotated.loc[weaken_mask].copy(),
        "thresholds": {
            "method": "directional_conservative_bootstrap",
            "stable_upper": stable_upper,
            "stable_lower": stable_lower,
            "upper_source": upper_source,
            "lower_source": lower_source,
            "iqr_stable_upper": iqr_upper,
            "iqr_stable_lower": iqr_lower,
            "quantile_stable_upper": quantile_upper,
            "quantile_stable_lower": quantile_lower,
            "observed_upper": observed_upper,
            "observed_lower": observed_lower,
            "iqr_upper_usable": bool(iqr_upper_usable),
            "iqr_lower_usable": bool(iqr_lower_usable),
            "min_stability": float(min_stability),
            "iqr_thresholds": dict(iqr_thresholds),
            "quantile_thresholds": dict(quantile_thresholds),
        },
    }


def run_highorder_threshold_analysis_bootstrap_quantile(
    df_merged: pd.DataFrame,
    prefix: str,
    *,
    upper_quantile: float = 0.95,
    lower_quantile: float = 0.05,
    n_bootstrap: int = 200,
    bootstrap_sample_size: int = 100_000,
    min_stability: float = 0.80,
    random_state: int = 2026,
    separate_chromosomes: bool = True,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    min_tail_pairs: int = 100,
    save_filtered: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Bootstrap independent positive and negative diff-tail quantiles."""
    required = {"locus1", "locus2", "v_f", "v_h", "geno_dis"}
    missing = sorted(required.difference(df_merged.columns))
    if missing:
        raise ValueError(f"df_merged is missing columns: {missing}")
    if not 0.5 < upper_quantile < 1.0:
        raise ValueError("upper_quantile must be in (0.5, 1.0)")
    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must be in (0.0, 0.5)")
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if bootstrap_sample_size < 1_000:
        raise ValueError("bootstrap_sample_size must be at least 1000")
    if not 0.5 <= min_stability < 1.0:
        raise ValueError("min_stability must be in [0.5, 1.0)")
    if min_tail_pairs < 20:
        raise ValueError("min_tail_pairs must be at least 20")

    data = df_merged.copy()
    data["v_f"] = pd.to_numeric(data["v_f"], errors="coerce")
    data["v_h"] = pd.to_numeric(data["v_h"], errors="coerce")
    data["geno_dis"] = pd.to_numeric(data["geno_dis"], errors="coerce")
    data["diff"] = data["v_h"] - data["v_f"]

    chr2_end = chr1_len + chr2_len
    chr1_mask = (
        ((data["locus1"] < chr1_len) | (data["locus2"] < chr1_len))
        & (data["geno_dis"] > 0)
    )
    chr2_mask = (
        (data["locus1"] > chr1_len)
        & (data["locus2"] > chr1_len)
        & ((data["locus1"] < chr2_end) | (data["locus2"] < chr2_end))
        & (data["geno_dis"] > 0)
    )

    if separate_chromosomes:
        subsets = {
            "chr1": data.loc[chr1_mask].copy(),
            "chr2": data.loc[chr2_mask].copy(),
        }
    else:
        subsets = {"global": data}

    analyses = {}
    for offset, (name, subset) in enumerate(subsets.items()):
        analyses[name] = _bootstrap_tail_quantile_subset(
            subset,
            upper_quantile=upper_quantile,
            lower_quantile=lower_quantile,
            n_bootstrap=n_bootstrap,
            bootstrap_sample_size=bootstrap_sample_size,
            min_stability=min_stability,
            random_state=random_state + offset,
            min_tail_pairs=min_tail_pairs,
        )

    if separate_chromosomes:
        chr1_result = analyses["chr1"]
        chr2_result = analyses["chr2"]
    else:
        global_result = analyses["global"]
        global_data = global_result["data"]
        global_filtered = global_result["filtered"]
        chr1_result = {
            "data": global_data.loc[chr1_mask].copy(),
            "filtered": global_filtered.loc[
                global_filtered.index.intersection(data.index[chr1_mask])
            ].copy(),
            "thresholds": global_result["thresholds"],
        }
        chr2_result = {
            "data": global_data.loc[chr2_mask].copy(),
            "filtered": global_filtered.loc[
                global_filtered.index.intersection(data.index[chr2_mask])
            ].copy(),
            "thresholds": global_result["thresholds"],
        }

    chr1_thresholds = chr1_result["thresholds"]
    chr2_thresholds = chr2_result["thresholds"]
    plot_chr1 = None
    plot_chr2 = None
    if make_plots:
        plot_chr1 = gwes_high_from_df(
            chr1_result["data"],
            prefix=prefix + "_quantile_bootstrap_chr1",
            flag_chr="chr1",
            fixed_thresholds=(
                chr1_thresholds["stable_upper"],
                chr1_thresholds["stable_lower"],
            ),
        )
        plot_chr2 = gwes_high_from_df(
            chr2_result["data"],
            prefix=prefix + "_quantile_bootstrap_chr2",
            flag_chr="chr2",
            fixed_thresholds=(
                chr2_thresholds["stable_upper"],
                chr2_thresholds["stable_lower"],
            ),
        )

    if separate_chromosomes:
        filtered = pd.concat(
            [chr1_result["filtered"], chr2_result["filtered"]],
            ignore_index=True,
        )
        strengthened = pd.concat(
            [chr1_result["strengthened"], chr2_result["strengthened"]],
            ignore_index=True,
        )
        weakened = pd.concat(
            [chr1_result["weakened"], chr2_result["weakened"]],
            ignore_index=True,
        )
        thresholds = {
            "chr1": chr1_thresholds,
            "chr2": chr2_thresholds,
        }
    else:
        filtered = analyses["global"]["filtered"]
        strengthened = analyses["global"]["strengthened"]
        weakened = analyses["global"]["weakened"]
        thresholds = analyses["global"]["thresholds"]

    filtered_path = None
    if save_filtered:
        filtered_path = prefix + "_quantile_bootstrap_filtered_pairs.parquet"
        filtered.to_parquet(filtered_path, index=False)

    if verbose:
        for name, result in analyses.items():
            threshold_info = result["thresholds"]
            print(f"{name} stable upper: {threshold_info['stable_upper']:.6f}")
            print(f"{name} stable lower: {threshold_info['stable_lower']:.6f}")
            print(f"{name} strengthened pairs: {len(result['strengthened']):,}")
            print(f"{name} weakened pairs: {len(result['weakened']):,}")
        print(f"Selected pairs: {len(filtered):,}")

    return {
        "threshold_method": "separate_tail_quantile_bootstrap",
        "threshold_scope": (
            "chromosome" if separate_chromosomes else "global"
        ),
        "thresholds": thresholds,
        "filtered_pairs": filtered,
        "strengthened_pairs": strengthened,
        "weakened_pairs": weakened,
        "df_chr1": chr1_result["data"],
        "df_chr2": chr2_result["data"],
        "filtered_chr1": chr1_result["filtered"],
        "filtered_chr2": chr2_result["filtered"],
        "plot_chr1": plot_chr1,
        "plot_chr2": plot_chr2,
        "filtered_path": filtered_path,
    }


def detect_signed_bimodality(
    diff,
    *,
    sample_size: int = 50_000,
    bins: int = 128,
    smoothing_sigma: float = 2.0,
    min_side_fraction: float = 0.05,
    min_mode_abs: float = 0.15,
    min_mode_separation: float = 0.30,
    max_valley_ratio: float = 0.50,
    random_state: int = 2026,
) -> Dict[str, object]:
    """Detect a separated negative/positive two-mode diff distribution."""
    from scipy.ndimage import gaussian_filter1d

    values = pd.to_numeric(pd.Series(diff), errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    if values.size < 1_000:
        raise ValueError("At least 1000 finite diff values are required")
    if sample_size < 1_000 or bins < 32:
        raise ValueError("sample_size must be >=1000 and bins must be >=32")

    if values.size > sample_size:
        rng = np.random.default_rng(random_state)
        values = rng.choice(values, size=sample_size, replace=False)

    negative_fraction = float(np.mean(values < 0))
    positive_fraction = float(np.mean(values > 0))
    negative_count = int(np.sum(values < 0))
    positive_count = int(np.sum(values > 0))
    central_fraction = float(np.mean(np.abs(values) <= min_mode_abs))
    lower = min(-1.0, float(np.min(values)))
    upper = max(1.0, float(np.max(values)))
    histogram, edges = np.histogram(values, bins=bins, range=(lower, upper))
    smoothed = gaussian_filter1d(
        histogram.astype(np.float64),
        sigma=float(smoothing_sigma),
        mode="nearest",
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    negative_indices = np.flatnonzero(centers < 0)
    positive_indices = np.flatnonzero(centers > 0)

    negative_peak_index = negative_indices[
        np.argmax(smoothed[negative_indices])
    ]
    positive_peak_index = positive_indices[
        np.argmax(smoothed[positive_indices])
    ]
    negative_mode = float(centers[negative_peak_index])
    positive_mode = float(centers[positive_peak_index])
    negative_peak_height = float(smoothed[negative_peak_index])
    positive_peak_height = float(smoothed[positive_peak_index])
    interval = smoothed[negative_peak_index:positive_peak_index + 1]
    valley_height = float(np.min(interval)) if interval.size else np.inf
    smaller_peak = min(negative_peak_height, positive_peak_height)
    valley_ratio = (
        valley_height / smaller_peak if smaller_peak > 0 else np.inf
    )
    mode_separation = positive_mode - negative_mode

    is_bimodal = bool(
        negative_fraction >= min_side_fraction
        and positive_fraction >= min_side_fraction
        and negative_mode <= -min_mode_abs
        and positive_mode >= min_mode_abs
        and mode_separation >= min_mode_separation
        and valley_ratio <= max_valley_ratio
    )
    return {
        "is_bimodal": is_bimodal,
        "negative_fraction": negative_fraction,
        "positive_fraction": positive_fraction,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "central_fraction": central_fraction,
        "negative_mode": negative_mode,
        "positive_mode": positive_mode,
        "mode_separation": float(mode_separation),
        "valley_ratio": float(valley_ratio),
        "sample_size": int(values.size),
        "criteria": {
            "min_side_fraction": float(min_side_fraction),
            "min_mode_abs": float(min_mode_abs),
            "min_mode_separation": float(min_mode_separation),
            "max_valley_ratio": float(max_valley_ratio),
        },
    }


def run_highorder_threshold_analysis_auto(
    df_merged: pd.DataFrame,
    prefix: str,
    *,
    k_upper: float = 1.0,
    k_lower: float = 2.0,
    upper_quantile: float = 0.95,
    lower_quantile: float = 0.05,
    n_bootstrap: int = 200,
    bootstrap_sample_size: int = 100_000,
    min_stability: float = 0.80,
    random_state: int = 2026,
    detection_sample_size: int = 50_000,
    min_side_fraction: float = 0.05,
    min_mode_abs: float = 0.15,
    zero_anchor_iqr: bool = True,
    min_mode_separation: float = 0.30,
    max_valley_ratio: float = 0.50,
    chr1_len: int = 3_288_558,
    chr2_len: int = 1_877_212,
    min_tail_pairs: int = 100,
    save_filtered: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Select conservative directional Bootstrap thresholds per chromosome."""
    required = {"locus1", "locus2", "v_f", "v_h", "geno_dis"}
    missing = sorted(required.difference(df_merged.columns))
    if missing:
        raise ValueError(f"df_merged is missing columns: {missing}")
    if k_upper < 0 or k_lower < 0:
        raise ValueError("k_upper and k_lower cannot be negative")
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if bootstrap_sample_size < 1_000:
        raise ValueError("bootstrap_sample_size must be at least 1000")
    if not 0.5 <= min_stability < 1.0:
        raise ValueError("min_stability must be in [0.5, 1.0)")

    data = df_merged.copy()
    data["v_f"] = pd.to_numeric(data["v_f"], errors="coerce")
    data["v_h"] = pd.to_numeric(data["v_h"], errors="coerce")
    data["geno_dis"] = pd.to_numeric(data["geno_dis"], errors="coerce")
    data["diff"] = data["v_h"] - data["v_f"]
    chr2_end = chr1_len + chr2_len
    chr1_mask = (
        ((data["locus1"] < chr1_len) | (data["locus2"] < chr1_len))
        & (data["geno_dis"] > 0)
    )
    chr2_mask = (
        (data["locus1"] > chr1_len)
        & (data["locus2"] > chr1_len)
        & ((data["locus1"] < chr2_end) | (data["locus2"] < chr2_end))
        & (data["geno_dis"] > 0)
    )
    subsets = {
        "chr1": data.loc[chr1_mask].copy(),
        "chr2": data.loc[chr2_mask].copy(),
    }

    detections = {}
    analyses = {}
    methods = {}
    for offset, (name, subset) in enumerate(subsets.items()):
        detection = detect_signed_bimodality(
            subset["diff"],
            sample_size=detection_sample_size,
            min_side_fraction=min_side_fraction,
            min_mode_abs=min_mode_abs,
            min_mode_separation=min_mode_separation,
            max_valley_ratio=max_valley_ratio,
            random_state=random_state + offset,
        )
        detections[name] = detection
        if detection["is_bimodal"]:
            method = "directional_conservative_bootstrap"
            iqr_analysis = _bootstrap_iqr_threshold_subset(
                subset,
                k_upper=float(k_upper),
                k_lower=float(k_lower),
                n_bootstrap=n_bootstrap,
                bootstrap_sample_size=bootstrap_sample_size,
                min_stability=min_stability,
                min_abs_threshold=min_mode_abs,
                zero_anchor_iqr=zero_anchor_iqr,
                random_state=random_state + offset,
            )
            quantile_analysis = _bootstrap_tail_quantile_subset(
                subset,
                upper_quantile=upper_quantile,
                lower_quantile=lower_quantile,
                n_bootstrap=n_bootstrap,
                bootstrap_sample_size=bootstrap_sample_size,
                min_stability=min_stability,
                random_state=random_state + offset,
                min_tail_pairs=min_tail_pairs,
            )
            analysis = _combine_directional_conservative_analyses(
                iqr_analysis,
                quantile_analysis,
                min_stability=min_stability,
            )
        else:
            method = "iqr_bootstrap"
            analysis = _bootstrap_iqr_threshold_subset(
                subset,
                k_upper=float(k_upper),
                k_lower=float(k_lower),
                n_bootstrap=n_bootstrap,
                bootstrap_sample_size=bootstrap_sample_size,
                min_stability=min_stability,
                min_abs_threshold=min_mode_abs,
                zero_anchor_iqr=zero_anchor_iqr,
                random_state=random_state + offset,
            )
            iqr_thresholds = analysis["thresholds"]
            upper_out_of_range = (
                iqr_thresholds["stable_upper"] > 1.0
                or iqr_thresholds["stable_upper"]
                > float(subset["diff"].max())
            )
            lower_out_of_range = (
                iqr_thresholds["stable_lower"] < -1.0
                or iqr_thresholds["stable_lower"]
                < float(subset["diff"].min())
            )
            if upper_out_of_range or lower_out_of_range:
                method = "iqr_bootstrap_out_of_range_empty"
                detection["out_of_range_policy"] = "empty"
                analysis["thresholds"]["upper_out_of_range"] = bool(
                    upper_out_of_range
                )
                analysis["thresholds"]["lower_out_of_range"] = bool(
                    lower_out_of_range
                )
        methods[name] = method
        analysis["thresholds"]["selected_method"] = method
        for frame_name in ("data", "filtered", "strengthened", "weakened"):
            analysis[frame_name]["threshold_method"] = method
        analyses[name] = analysis

    plots = {}
    for name, flag in (("chr1", "chr1"), ("chr2", "chr2")):
        analysis = analyses[name]
        thresholds = analysis["thresholds"]
        plots[name] = None
        if make_plots:
            plots[name] = gwes_high_from_df(
                analysis["data"],
                prefix=prefix + f"_auto_{name}",
                flag_chr=flag,
                fixed_thresholds=(
                    thresholds["stable_upper"],
                    thresholds["stable_lower"],
                ),
            )

    filtered = pd.concat(
        [analyses["chr1"]["filtered"], analyses["chr2"]["filtered"]],
        ignore_index=True,
    )
    strengthened = pd.concat(
        [
            analyses["chr1"]["strengthened"],
            analyses["chr2"]["strengthened"],
        ],
        ignore_index=True,
    )
    weakened = pd.concat(
        [analyses["chr1"]["weakened"], analyses["chr2"]["weakened"]],
        ignore_index=True,
    )
    filtered_path = None
    if save_filtered:
        filtered_path = prefix + "_auto_filtered_pairs.parquet"
        filtered.to_parquet(filtered_path, index=False)

    if verbose:
        for name in ("chr1", "chr2"):
            detection = detections[name]
            thresholds = analyses[name]["thresholds"]
            print(f"{name} detected bimodal: {detection['is_bimodal']}")
            print(f"{name} selected method: {methods[name]}")
            print(f"{name} valley ratio: {detection['valley_ratio']:.4f}")
            print(f"{name} stable upper: {thresholds['stable_upper']:.6f}")
            print(f"{name} stable lower: {thresholds['stable_lower']:.6f}")
            if "upper_source" in thresholds:
                print(f"{name} upper source: {thresholds['upper_source']}")
                print(f"{name} lower source: {thresholds['lower_source']}")
        print(f"Selected pairs: {len(filtered):,}")

    return {
        "methods": methods,
        "bimodality": detections,
        "thresholds": {
            "chr1": analyses["chr1"]["thresholds"],
            "chr2": analyses["chr2"]["thresholds"],
        },
        "filtered_pairs": filtered,
        "strengthened_pairs": strengthened,
        "weakened_pairs": weakened,
        "df_chr1": analyses["chr1"]["data"],
        "df_chr2": analyses["chr2"]["data"],
        "filtered_chr1": analyses["chr1"]["filtered"],
        "filtered_chr2": analyses["chr2"]["filtered"],
        "plot_chr1": plots["chr1"],
        "plot_chr2": plots["chr2"],
        "filtered_path": filtered_path,
    }



def find_diff_outliers(df_sp1: pd.DataFrame, k1: float, k2: float) -> Tuple[float, float]:
    s = pd.Series(df_sp1["diff"]).dropna()
    q3 = s.quantile(0.75)
    q1 = s.quantile(0.25)
    iqr = q3 - q1
    outlier1 = q3 + k1 * 1.5 * iqr
    outlier2 = q1 - k2 * 1.5 * iqr
    return float(outlier1), float(outlier2)


def find_diff_outliers_KDE(df_sp1: pd.DataFrame, margin: float = 0.1) -> Tuple[float, float]:
    """
    Symmetric KDE baseline: peak of KDE over all 'diff', then +/- margin.
    """
    s = pd.Series(df_sp1["diff"]).dropna().values
    if s.size == 0:
        return np.nan, np.nan
    xs = np.linspace(s.min(), s.max(), 512)
    kde = gaussian_kde(s)
    ys = kde(xs)
    peak = xs[np.argmax(ys)]
    def find_kde_margin(df_sp1):
        s = pd.Series(df_sp1["diff"]).dropna()
        q3 = s.quantile(0.75)
        q1 = s.quantile(0.25)
        IQR = (q3 - q1)*1.5
        return IQR

    # margin1 = find_kde_margin( df_sp1[df_sp1["diff"] > 0].copy())
    # margin2 = find_kde_margin( df_sp1[df_sp1["diff"] < 0].copy())
    margin1 = max(margin,find_kde_margin(df_sp1))
    margin1 = min(margin1,0.55)
    upper = peak + margin1
    lower = peak - margin1
    return float(upper), float(lower)


def _safe_kde_peak(values) -> Optional[float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.std(v) == 0:
        return None
    xs = np.linspace(v.min(), v.max(), 512)
    kde = gaussian_kde(v, bw_method=0.3)
    # kde = gaussian_kde(v)
    ys = kde(xs)
    return float(xs[np.argmax(ys)])


def find_diff_outliers_KDE_two_sided(df_sp1: pd.DataFrame, margin: float = 0.1) -> Tuple[float, float]:
    """
    Two-sided KDE:
      - diff>0 -> upper = peak_pos + margin
      - diff<0 -> lower = peak_neg - margin
      fallback: symmetric KDE -> IQR(k=3)
    """
    diff_all = pd.Series(df_sp1["diff"]).dropna()
    diff_mid = 0.0
    diff_mid = max(0,np.median(df_sp1["diff"]))
    pos = diff_all[diff_all > diff_mid].values
    neg = diff_all[diff_all < diff_mid].values

    peak_pos = _safe_kde_peak(pos) if pos.size else None
    peak_neg = _safe_kde_peak(neg) if neg.size else None

    # def find_kde_margin(df_sp1):
    #     s = pd.Series(df_sp1["diff"]).dropna()
    #     if df_sp1["diff"].all() > 0:
    #         kde_min = 0.2
    #     else:
    #         if df_sp1["diff"].all() < 0:
    #             s = abs(s)
    #             kde_min = 0.1
    #         else:
    #             return 0.1

    #     q3 = s.quantile(0.75)
    #     q1 = s.quantile(0.25)
    #     IQR = q3 - q1
    #     return max(kde_min, 1 * IQR)

    def find_kde_margin(df_sp1):
        s = pd.Series(df_sp1["diff"]).dropna()
        q3 = s.quantile(0.75)
        q1 = s.quantile(0.25)
        IQR = (q3 - q1)*1.5
        return IQR

    # margin1 = find_kde_margin( df_sp1[df_sp1["diff"] > 0].copy())
    # margin2 = find_kde_margin( df_sp1[df_sp1["diff"] < 0].copy())
    margin1 = max(margin,find_kde_margin( df_sp1[df_sp1["diff"] > diff_mid].copy()))
    margin2 = max(margin,find_kde_margin( df_sp1[df_sp1["diff"] < diff_mid].copy()))
    margin1 = min(margin1,0.55)
    margin2 = min(margin2,0.55)
    # print("margin11:", find_kde_margin( df_sp1[df_sp1["diff"] > diff_mid].copy()))
    # print("margin1:", margin1)
    # print("margin2:", margin2)
    upper = peak_pos + margin1 if peak_pos is not None else None
    lower = peak_neg - margin2 if peak_neg is not None else None

    if upper is None or lower is None:
        u_sym, l_sym = find_diff_outliers_KDE(df_sp1, margin=margin)
        if upper is None and np.isfinite(u_sym):
            upper = u_sym
        if lower is None and np.isfinite(l_sym):
            lower = l_sym

    if upper is None or lower is None:
        iqr_u, iqr_l = find_diff_outliers(df_sp1, 3, 3)
        if upper is None:
            upper = iqr_u
        if lower is None:
            lower = iqr_l

    return float(upper), float(lower)

def get_extreme_values(df_sp: pd.DataFrame) -> Tuple[float, float]:
    df_filtered = df_sp[df_sp["geno_dis"] > 0]
    return float(df_filtered["diff"].max()), float(df_filtered["diff"].min())

def plot_GWES_Diff_pure(
    df_sp2: pd.DataFrame,
    flag_chr: str,
    dir_save: str,
    outlier_mode: str = "iqr",
    kde_margin: float = 0.1,
    k: int = 2,
    fixed_thresholds: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    plot_l = 28 if flag_chr == "chr1" else 16

    df_sp1 = df_sp2.copy()

    # Compute outlier thresholds consistently
    if fixed_thresholds is not None:
        upper, lower = map(float, fixed_thresholds)
        if not np.isfinite(upper) or not np.isfinite(lower):
            raise ValueError("fixed_thresholds must contain finite values")
        if lower >= upper:
            raise ValueError("fixed lower threshold must be below upper threshold")
        outliers_s = [(upper, lower)]
    elif outlier_mode == "kde":
        # df_sp1 = df_sp[(df_sp['locus1']<3288558)&(df_sp['locus2']<3288558)]
        upper, lower = find_diff_outliers_KDE_two_sided(df_sp1, margin=kde_margin)
        # upper, lower = find_diff_outliers_KDE(df_sp1, margin=kde_margin)
        outliers_s = [(upper, lower)] * 5  # keep drawing style consistent
    else:
        outliers_s = [find_diff_outliers(df_sp1, i + 1, i + 1) for i in range(5)]
        # use ks=3 (index=2) as main thresholds like before
        upper, lower = outliers_s[2]

    if fixed_thresholds is None:
        upper = max(upper,0.15)
        lower = min(lower,-0.15)
    # for highlighting points
    df_sp = df_sp1[df_sp1["geno_dis"] >= 0]
    df_sp_ex = df_sp[(df_sp["diff"] > upper) | (df_sp["diff"] < lower)]
    max_diff, min_diff = get_extreme_values(df_sp)

    sns.set_style("white")
    fig, axs = plt.subplots(1, 2, figsize=(plot_l, 7), gridspec_kw={"width_ratios": [1, 3]})

    # Boxplot
    sns.boxplot(y=df_sp2["diff"], color="skyblue", ax=axs[0])
    axs[0].set_title("Boxplot of EpiDis diff", fontsize=18)
    axs[0].set_xlabel("HighOrder EpiDis diff", fontsize=18)
    axs[0].set_ylabel("EpiDis diff", fontsize=18)
    axs[0].grid(True)

    # Scatter
    axs[1].plot(df_sp["geno_dis"].values, df_sp["diff"].values, ".", alpha=0.5, markersize=5, label="Common pairs")
    axs[1].plot(df_sp_ex["geno_dis"].values, df_sp_ex["diff"].values, ".", alpha=0.5, markersize=5, label="Strong pairs (SNP–SNP)")

    # Threshold lines
    if fixed_thresholds is not None:
        axs[1].axhline(y=upper, c="r", ls="--")
        axs[1].axhline(y=lower, c="r", ls="--")
        axs[1].text(
            axs[1].get_xlim()[1], upper,
            f"Fixed upper: {upper:.3f}",
            va="bottom", ha="right", color="red", fontsize=13,
        )
        axs[1].text(
            axs[1].get_xlim()[1], lower,
            f"Fixed lower: {lower:.3f}",
            va="bottom", ha="right", color="red", fontsize=13,
        )
    elif outlier_mode == "kde":
        lbl_u = f"KDE upper: {upper:.3f}"
        lbl_l = f"KDE lower: {lower:.3f}"
        if np.isfinite(upper):
            axs[1].axhline(y=upper, c="r", ls="--")
            axs[1].text(axs[1].get_xlim()[1], upper, lbl_u, va="bottom", ha="right", color="red", fontsize=13)
        if np.isfinite(lower):
            axs[1].axhline(y=lower, c="r", ls="--")
            axs[1].text(axs[1].get_xlim()[1], lower, lbl_l, va="bottom", ha="right", color="red", fontsize=13)
    else:
        for i, (u, l) in enumerate(outliers_s, start=1):
            if np.isfinite(u):
                if u<1 and u>0.15:
                    axs[1].axhline(y=u, c="r", ls="--")
                    axs[1].text(axs[1].get_xlim()[1], u, f"Outlier(ks={i}): {u:.3f}", va="bottom", ha="right", color="red", fontsize=13)
            if np.isfinite(l):
                if l>-1 and l<-0.15:
                    axs[1].axhline(y=l, c="r", ls="--")
                    axs[1].text(axs[1].get_xlim()[1], l, f"Outlier(kw={i}): {l:.3f}", va="bottom", ha="right", color="red", fontsize=13)

    axs[1].set_xlabel("Genome Distance", fontsize=18)
    axs[1].set_ylabel("EpiDis Difference", fontsize=18)
    axs[1].set_title("EpiDive_Diff_GWES", fontsize=18)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].tick_params(axis="both", which="both", direction="out", length=6, width=2, colors="black", labelsize=15)
    axs[1].legend(loc="lower right", fontsize=15)
    axs[1].set_ylim([-1, 1])
    # if (max_diff < 0.15) or (min_diff > -0.15):
    #     axs[1].set_ylim([-1, 1])

    plt.tight_layout()
    plt.savefig(dir_save + ".GWES_HighOrder_diff.png", dpi=200)
    plt.close()
    return float(upper), float(lower)

def plot_GWES_Diff(
    df_sp2: pd.DataFrame,
    flag_chr: str,
    dir_save: str,
    outlier_mode: str = "iqr",
    kde_margin: float = 0.1,
    k: int = 2,
) -> Tuple[float, float]:
    plot_l = 28 if flag_chr == "chr1" else 16

    df_sp1 = df_sp2.copy()

    # Compute outlier thresholds consistently
    if outlier_mode == "kde":
        # df_sp1 = df_sp[(df_sp['locus1']<3288558)&(df_sp['locus2']<3288558)]
        upper, lower = find_diff_outliers_KDE_two_sided(df_sp1, margin=kde_margin)
        # upper, lower = find_diff_outliers_KDE(df_sp1, margin=kde_margin)
        outliers_s = [(upper, lower)] * 5  # keep drawing style consistent
    else:
        ##########  old  ###################################################################
        # df_sp1 = df_sp[(df_sp['locus1']<3288558)&(df_sp['locus2']<3288558)]
        # outliers_s = [find_diff_outliers(df_sp1, i + 1, i + 1) for i in range(5)]
        # # use ks=3 (index=2) as main thresholds like before
        # upper, lower = outliers_s[2]

        ##########  new  ###################################################################
        diff_mid = max(0,np.median(df_sp1["diff"]))
        # diff_mid = np.median(df_sp1["diff"])
        df_unipairsP = df_sp1[df_sp1["diff"]>diff_mid].copy()
        df_unipairsP = df_unipairsP[df_unipairsP["diff"]>np.median(df_unipairsP["diff"])].copy()
        df_unipairsN = df_sp1[df_sp1["diff"]<diff_mid].copy()
        df_unipairsN = df_unipairsN[df_unipairsN["diff"]<np.median(df_unipairsN["diff"])].copy()

        outliers_s_upper = [find_diff_outliers(df_unipairsP, i + 1, i + 1) for i in range(5)]
        outliers_s_lower = [find_diff_outliers(df_unipairsN, i + 1, i + 1) for i in range(5)]

        outliers_s = [
            (u[0], l[1])
            for u, l in zip(outliers_s_upper, outliers_s_lower)
        ]
        # use ks=3 (index=2) as main thresholds like before
        if k>5:
            k=5
        if k<1:
            k=1
        upper, lower = outliers_s[int(k)-1]
        upper = np.quantile(df_sp1["diff"], 0.99995)
        lower = np.quantile(df_sp1["diff"], 0.001)
        upper = max(upper,0.15)
        lower = min(lower,-0.15)
        # y = df_sp1["diff"].values
        # low_clip  = np.quantile(y, 0.0005)
        # high_clip = np.quantile(y, 0.9995)
        # y_clip = np.clip(y, low_clip, high_clip)
        # upper = np.quantile(y_clip, 0.999)
        # lower = np.quantile(y_clip, 0.001)
        outliers_s = [(upper, lower)] * 5

    upper = max(upper,0.15)
    lower = min(lower,-0.15)
    # for highlighting points
    df_sp = df_sp1[df_sp1["geno_dis"] >= 0]
    df_sp_ex = df_sp[(df_sp["diff"] > upper) | (df_sp["diff"] < lower)]
    max_diff, min_diff = get_extreme_values(df_sp)

    sns.set_style("white")
    fig, axs = plt.subplots(1, 2, figsize=(plot_l, 7), gridspec_kw={"width_ratios": [1, 3]})

    # Boxplot
    sns.boxplot(y=df_sp2["diff"], color="skyblue", ax=axs[0])
    axs[0].set_title("Boxplot of EpiDis diff", fontsize=18)
    axs[0].set_xlabel("HighOrder EpiDis diff", fontsize=18)
    axs[0].set_ylabel("EpiDis diff", fontsize=18)
    axs[0].grid(True)

    # Scatter
    axs[1].plot(df_sp["geno_dis"].values, df_sp["diff"].values, ".", alpha=0.5, markersize=5, label="Common pairs")
    axs[1].plot(df_sp_ex["geno_dis"].values, df_sp_ex["diff"].values, ".", alpha=0.5, markersize=5, label="Strong pairs (SNP–SNP)")

    # Threshold lines
    if outlier_mode == "kde":
        lbl_u = f"KDE upper: {upper:.3f}"
        lbl_l = f"KDE lower: {lower:.3f}"
        if np.isfinite(upper):
            axs[1].axhline(y=upper, c="r", ls="--")
            axs[1].text(axs[1].get_xlim()[1], upper, lbl_u, va="bottom", ha="right", color="red", fontsize=13)
        if np.isfinite(lower):
            axs[1].axhline(y=lower, c="r", ls="--")
            axs[1].text(axs[1].get_xlim()[1], lower, lbl_l, va="bottom", ha="right", color="red", fontsize=13)
    else:
        for i, (u, l) in enumerate(outliers_s, start=1):
            if np.isfinite(u):
                if u<1 and u>0.15:
                    axs[1].axhline(y=u, c="r", ls="--")
                    axs[1].text(axs[1].get_xlim()[1], u, f"Outlier(ks={i}): {u:.3f}", va="bottom", ha="right", color="red", fontsize=13)
            if np.isfinite(l):
                if l>-1 and l<-0.15:
                    axs[1].axhline(y=l, c="r", ls="--")
                    axs[1].text(axs[1].get_xlim()[1], l, f"Outlier(kw={i}): {l:.3f}", va="bottom", ha="right", color="red", fontsize=13)

    axs[1].set_xlabel("Genome Distance", fontsize=18)
    axs[1].set_ylabel("EpiDis Difference", fontsize=18)
    axs[1].set_title("EpiDis_Diff_GWES", fontsize=18)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].tick_params(axis="both", which="both", direction="out", length=6, width=2, colors="black", labelsize=15)
    axs[1].legend(loc="lower right", fontsize=15)
    if (max_diff < 0.15) or (min_diff > -0.15):
        axs[1].set_ylim([-1, 1])

    plt.tight_layout()
    plt.savefig(dir_save + ".GWES_HighOrder_diff.png", dpi=200)
    plt.close()

    return float(upper), float(lower)


if __name__ == "__main__":
    # Backward-compatible CLI: python epidis_gwes_high.py <csv_path> <prefix>
    if len(sys.argv) < 3:
        print("Usage: epidis_gwes_high.py <csv_path> <prefix>")
        sys.exit(1)

    dir_fs = sys.argv[1]
    prefix = sys.argv[2]

    print("Loading the data ...")
    df_unipairs = pd.read_csv(dir_fs)
    res = gwes_high_from_df(df_unipairs, prefix, flag_chr="chr1")
    print("Thresholds:", res["thresholds"])
    print("Figure:", res["figure_path"])
    print("DONE!!")
