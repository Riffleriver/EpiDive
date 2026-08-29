#!/usr/bin/env python3
"""Batch, resume, cache, and sharding CLI for the CPU EpiDive workflow."""

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import time
import traceback

import numpy as np
import pandas as pd

import epidis_reweight as er
from EpiDive_vp_ho_cpu import (
    run_highorder_comparison_pipeline,
    run_vp_pipeline,
    transform_row,
)


def _parse_initial_threshold(value):
    if str(value).strip().lower() == "auto":
        return "auto"
    threshold = float(value)
    if threshold < 0:
        raise argparse.ArgumentTypeError("threshold must be non-negative")
    return threshold


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the multi-CPU VP/high-order EpiDive workflow with optional "
            "batch backgrounds, deterministic sharding, cache, and resume."
        )
    )
    parser.add_argument("--input", required=True, help="Input SNP matrix")
    parser.add_argument("--output-dir", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--background", type=int)
    group.add_argument("--background-file")
    parser.add_argument("--background-column", default="representative_snp")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--background-limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--prefix", default="vp10k")
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
        help='Initial full-data threshold: a number or "auto"',
    )
    parser.add_argument(
        "--subgroup-threshold", type=_parse_initial_threshold, default="auto"
    )
    parser.add_argument("--full-k", type=float, default=1.0)
    parser.add_argument("--subgroup-1-k", type=float, default=1.0)
    parser.add_argument("--subgroup-0-k", type=float, default=2.0)
    parser.add_argument(
        "--threshold-method", choices=["auto", "iqr", "quantile"],
        default="auto",
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
    parser.add_argument("--pair-jobs", type=int, default=-1)
    parser.add_argument("--reweight-jobs", type=int, default=-1)
    parser.add_argument("--cpu-block-size", type=int, default=4096)
    parser.add_argument(
        "--accumulation-dtype", choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument("--distance-jobs", type=int, default=1)
    parser.add_argument("--distance-chunk-size", type=int, default=100_000)
    parser.add_argument("--ld-threshold", type=int, default=5_000)
    parser.add_argument("--apply-gwes-filter", action="store_true")
    parser.add_argument("--no-restore-final-patterns", action="store_true")
    parser.add_argument("--no-stream-output", action="store_true")
    parser.add_argument(
        "--final-only", action="store_true",
        help="Keep only final filtered background pairs and status markers",
    )
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--plot", dest="make_plots", action="store_true")
    plots.add_argument("--no-plot", dest="make_plots", action="store_false")
    parser.set_defaults(make_plots=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _load_background_ids(args):
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
                f"Missing background column {args.background_column!r}; "
                f"available={list(frame.columns)}"
            )
        values = pd.to_numeric(frame[args.background_column], errors="coerce")
        if values.isna().any():
            raise ValueError("Background column contains non-numeric values")
        backgrounds = list(dict.fromkeys(values.astype(np.int64).tolist()))
    if args.shard_count < 1:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    backgrounds = [
        value for position, value in enumerate(backgrounds)
        if position % args.shard_count == args.shard_index
    ]
    if args.background_limit is not None:
        if args.background_limit < 1:
            raise ValueError("--background-limit must be positive")
        backgrounds = backgrounds[:args.background_limit]
    if not backgrounds:
        raise ValueError("No backgrounds were selected for this shard")
    return backgrounds


def _signature(args, *, full_only=False):
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
        "pair_jobs": args.pair_jobs,
        "cpu_block_size": args.cpu_block_size,
        "accumulation_dtype": args.accumulation_dtype,
        "stream_output": not args.no_stream_output,
        "make_plots": args.make_plots,
    }
    if not full_only:
        fields.update({
            "subgroup_threshold": args.subgroup_threshold,
            "subgroup_1_k": args.subgroup_1_k,
            "subgroup_0_k": args.subgroup_0_k,
            "threshold_method": args.threshold_method,
            "bootstrap": args.bootstrap,
            "bootstrap_sample_size": args.bootstrap_sample_size,
            "min_stability": args.min_stability,
            "restore_final_patterns": not args.no_restore_final_patterns,
            "ld_threshold": args.ld_threshold,
            "ess_mode": args.ess_mode,
            "formal_min_fraction": args.formal_min_fraction,
            "exploratory_min_fraction": args.exploratory_min_fraction,
            "final_only": args.final_only,
        })
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _matching_marker(path, signature):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("analysis_signature") == signature
    except (OSError, ValueError, TypeError):
        return False


def _append_record(path, record):
    columns = ["time", "background", "status", "allele_1", "allele_0", "message"]
    exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow({name: record.get(name, "") for name in columns})


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


def _save_full_cache(full, cache_dir, signature):
    os.makedirs(cache_dir, exist_ok=True)
    files = {
        "final_pairs": "full_cache_final_pairs.parquet",
        "snp_bin": "full_cache_snp_bin.parquet",
        "weight_ser": "full_cache_weight.parquet",
        "pattern_map": "full_cache_pattern_map.parquet",
    }
    full["final_pairs"].to_parquet(os.path.join(cache_dir, files["final_pairs"]), index=False)
    full["snp_bin"].to_parquet(os.path.join(cache_dir, files["snp_bin"]), index=True)
    full["weight_ser"].rename("weight").to_frame().to_parquet(
        os.path.join(cache_dir, files["weight_ser"]), index=True
    )
    full["pattern_map"].to_parquet(os.path.join(cache_dir, files["pattern_map"]), index=False)
    _write_json_atomic(os.path.join(cache_dir, "full_cache.json"), {
        "status": "complete", "full_analysis_signature": signature,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files,
    })


def _load_full_cache(cache_dir, signature):
    manifest_path = os.path.join(cache_dir, "full_cache.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("full_analysis_signature") != signature:
            return None
        paths = {key: os.path.join(cache_dir, value) for key, value in manifest["files"].items()}
        if not all(os.path.isfile(path) for path in paths.values()):
            return None
        weight = pd.read_parquet(paths["weight_ser"]).iloc[:, 0]
        return {
            "final_pairs": pd.read_parquet(paths["final_pairs"]),
            "snp_bin": pd.read_parquet(paths["snp_bin"]),
            "weight_ser": weight,
            "pattern_map": pd.read_parquet(paths["pattern_map"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _vp_common(args):
    return {
        "maf": args.maf,
        "min_mac": args.min_mac,
        "model": args.model,
        "scc_threshold": args.scc_threshold,
        "hcc_threshold": args.hcc_threshold,
        "apply_gwes_filter": args.apply_gwes_filter,
        "restore_patterns": False,
        "reweight_n_jobs": args.reweight_jobs,
        "pair_n_jobs": args.pair_jobs,
        "block_size": args.cpu_block_size,
        "accumulation_dtype": args.accumulation_dtype,
        "pair_stream_output": not args.no_stream_output,
        "distance_n_jobs": args.distance_jobs,
        "distance_chunk_size": args.distance_chunk_size,
        "make_plots": args.make_plots,
        "verbose": args.verbose,
        "highorder_ess_mode": args.ess_mode,
        "highorder_formal_min_fraction": args.formal_min_fraction,
        "highorder_exploratory_min_fraction": args.exploratory_min_fraction,
    }


def _run_background(args, snp_df, full, background, output_dir):
    binary = transform_row(snp_df.loc[background])
    masks = {1: (binary == 1).to_numpy(), 0: (binary == 0).to_numpy()}
    counts = {str(allele): int(mask.sum()) for allele, mask in masks.items()}
    signature = _signature(args)
    for allele, subgroup_k in ((1, args.subgroup_1_k), (0, args.subgroup_0_k)):
        marker = os.path.join(output_dir, f"_ALLELE_{allele}_SUCCESS.json")
        if not args.no_resume and _matching_marker(marker, signature):
            continue
        subgroup = snp_df.loc[:, masks[allele]].copy(deep=False)
        subgroup.attrs.update(snp_df.attrs)
        subgroup_prefix = f"{args.prefix}_background_{background}_{allele}"
        high = run_vp_pipeline(
            snp_m=subgroup,
            dir_save=output_dir,
            prefix=subgroup_prefix,
            threshold=args.subgroup_threshold,
            k=subgroup_k,
            highorder=snp_df.shape[1],
            **_vp_common(args),
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
                pair_n_jobs=args.pair_jobs,
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
        _write_json_atomic(marker, {
            "background": int(background), "allele": allele, "status": "complete",
            "analysis_signature": signature,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": counts[str(allele)],
            "skipped": bool(high.get("skipped")),
            "skip_reason": high.get("skip_reason"),
            "highorder_screen": high.get("highorder_screen"),
            "final_only": bool(args.final_only),
            "final_output": final_output,
            "removed_intermediate_files": removed_files,
        })
        del high, subgroup, comparison
        gc.collect()
    return counts


def main(argv=None):
    args = _build_cli_parser().parse_args(argv)
    logging.disable(logging.NOTSET if args.verbose else logging.INFO)
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
    selected = _load_background_ids(args)
    missing = [value for value in selected if value not in snp_df.index]
    backgrounds = [value for value in selected if value in snp_df.index]
    if not backgrounds:
        raise ValueError("None of the selected backgrounds occur in the SNP matrix")
    batch_mode = args.background_file is not None
    signature = _signature(args)
    shard = f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}"
    status_path = os.path.join(args.output_dir, f"batch_status_{shard}.tsv")
    error_path = os.path.join(args.output_dir, f"batch_errors_{shard}.tsv")
    for value in missing:
        _append_record(error_path, {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "background": value, "status": "missing",
            "message": "Background locus is absent from the input matrix",
        })
    pending = []
    resumed = 0
    for background in backgrounds:
        output_dir = os.path.join(args.output_dir, f"background_{background}") if batch_mode else args.output_dir
        if not args.no_resume and _matching_marker(os.path.join(output_dir, "_SUCCESS.json"), signature):
            resumed += 1
        else:
            pending.append(background)
    print(f"Selected {len(backgrounds):,}; pending={len(pending):,}; resumed={resumed:,}")
    if not pending:
        return 0
    full_signature = _signature(args, full_only=True)
    full_dir = os.path.join(args.output_dir, "_full", full_signature[:16]) if batch_mode else args.output_dir
    os.makedirs(full_dir, exist_ok=True)
    full = _load_full_cache(full_dir, full_signature) if batch_mode and not args.no_resume else None
    if full is None:
        full = run_vp_pipeline(
            snp_m=snp_df, dir_save=full_dir, prefix=args.prefix,
            threshold=args.threshold, k=args.full_k, **_vp_common(args),
        )
        if batch_mode:
            _save_full_cache(full, full_dir, full_signature)
    else:
        print(f"Reusing full-data VP cache: {full_dir}")
    completed = 0
    failed = len(missing)
    for background in pending:
        output_dir = os.path.join(args.output_dir, f"background_{background}") if batch_mode else args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        try:
            counts = _run_background(args, snp_df, full, background, output_dir)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_json_atomic(os.path.join(output_dir, "_SUCCESS.json"), {
                "background": int(background), "status": "complete",
                "analysis_signature": signature, "completed_at": now,
                "allele_1_samples": counts["1"], "allele_0_samples": counts["0"],
            })
            _append_record(status_path, {
                "time": now, "background": background, "status": "complete",
                "allele_1": counts["1"], "allele_0": counts["0"],
            })
            completed += 1
        except Exception as exc:
            failed += 1
            message = f"{type(exc).__name__}: {exc}"
            _append_record(error_path, {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "background": background, "status": "failed", "message": message,
            })
            with open(os.path.join(output_dir, "_ERROR.txt"), "w", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
            logging.error("Background %s failed: %s", background, message)
            if not batch_mode:
                raise
    print(f"Batch finished: completed={completed}, resumed={resumed}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
