#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact SNP-pattern collapsing and EpiDis result restoration.

Typical use::

    import epidis_filter as ef

    snp_reduced, pattern_map = ef.collapse_duplicate_patterns(
        snp_bin,
        weight_ser=weight_ser,
        map_path="vp10k_pattern_map.parquet",
    )
    # Run EpiDis on snp_reduced, then restore the original locus identities.
    restored = ef.restore_pattern_pairs(
        df_pairs,
        pattern_map,
        threshold=0.3,
        output_path="vp10k_Epi_pairs_restored.parquet",
    )

The collapse is exact: rows are packed into bits and only identical 0/1
genotype vectors are grouped.  It is not an approximate LD-pruning step.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd


PathLike = Union[str, Path]

__all__ = [
    "collapse_duplicate_patterns",
    "restore_pattern_pairs",
]


def _as_weight_series(
    weight_ser: Union[pd.Series, pd.DataFrame],
) -> pd.Series:
    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError("weight_ser DataFrame must have exactly one column")
        weight_ser = weight_ser.iloc[:, 0]
    if not isinstance(weight_ser, pd.Series):
        raise TypeError("weight_ser must be a pandas Series or one-column DataFrame")
    return weight_ser


def _write_parquet(df: pd.DataFrame, path: PathLike) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)


def collapse_duplicate_patterns(
    snp_bin: pd.DataFrame,
    *,
    weight_ser: Optional[Union[pd.Series, pd.DataFrame]] = None,
    map_path: Optional[PathLike] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse exactly identical binary SNP patterns.

    Parameters
    ----------
    snp_bin
        Binary loci-by-samples DataFrame. Values must be 0 or 1.
    weight_ser
        Optional sample weights aligned by sample name. When supplied, the
        EpiDis value shared by pairs inside each identical-pattern group is
        calculated and stored as ``within_pattern_v`` in the map.
    map_path
        Optional Parquet destination for the pattern map.
    verbose
        Print a short collapse summary.

    Returns
    -------
    snp_reduced, pattern_map
        ``snp_reduced`` contains the first locus from every exact pattern.
        ``pattern_map`` maps every original locus to its representative.
    """
    if not isinstance(snp_bin, pd.DataFrame):
        raise TypeError("snp_bin must be a pandas DataFrame")
    if snp_bin.empty:
        raise ValueError("snp_bin cannot be empty")
    if not snp_bin.index.is_unique:
        raise ValueError("snp_bin index contains duplicate locus identifiers")
    if not snp_bin.columns.is_unique:
        raise ValueError("snp_bin columns contain duplicate sample identifiers")

    values = snp_bin.to_numpy(copy=False)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("snp_bin must contain numeric 0/1 values")
    if np.isnan(values).any():
        raise ValueError("snp_bin contains missing values")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("snp_bin must contain only 0 and 1")

    x = np.ascontiguousarray(values, dtype=np.uint8)
    packed = np.packbits(x, axis=1)
    _, first_positions_raw, pattern_group = np.unique(
        packed,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    loci = snp_bin.index.to_numpy(copy=False)
    representatives = loci[first_positions_raw]
    group_sizes = np.bincount(pattern_group)

    pattern_map = pd.DataFrame(
        {
            "locus": loci,
            "row_position": np.arange(len(loci), dtype=np.int64),
            "pattern_group": pattern_group.astype(np.int64, copy=False),
            "representative": representatives[pattern_group],
            "pattern_size": group_sizes[pattern_group],
        }
    )

    if weight_ser is not None:
        weights_source = _as_weight_series(weight_ser)
        missing_samples = snp_bin.columns.difference(weights_source.index).tolist()
        if missing_samples:
            raise ValueError(
                f"weight_ser is missing {len(missing_samples)} samples required by "
                f"snp_bin; first missing samples: {missing_samples[:10]}"
            )
        aligned_weights = weights_source.reindex(snp_bin.columns)
        weights = aligned_weights.to_numpy(dtype=np.float64)
        nonfinite_mask = ~np.isfinite(weights)
        if nonfinite_mask.any():
            bad_samples = aligned_weights.index[nonfinite_mask].tolist()
            raise ValueError(
                f"weight_ser contains non-finite weights for {len(bad_samples)} "
                f"samples; first affected samples: {bad_samples[:10]}"
            )
        if np.any(weights < 0):
            raise ValueError("sample weights cannot be negative")
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            raise ValueError("sample weights must have a positive sum")

        representative_x = x[first_positions_raw].astype(np.float64, copy=False)
        p1 = representative_x @ weights / weight_sum
        p0 = 1.0 - p1
        entropy = np.zeros_like(p1)
        valid = (p1 > 0.0) & (p1 < 1.0)
        entropy[valid] = -(
            p1[valid] * np.log2(p1[valid])
            + p0[valid] * np.log2(p0[valid])
        )
        within_pattern_v = np.sqrt(entropy)
        pattern_map["within_pattern_v"] = within_pattern_v[pattern_group]

    # np.unique orders packed values; restore the original locus order.
    first_positions = np.sort(first_positions_raw)
    snp_reduced = snp_bin.iloc[first_positions].copy()

    if map_path is not None:
        _write_parquet(pattern_map, map_path)

    if verbose:
        duplicate_count = len(snp_bin) - len(snp_reduced)
        print(f"Original loci: {len(snp_bin):,}")
        print(f"Unique genotype patterns: {len(snp_reduced):,}")
        print(f"Collapsed duplicate-pattern loci: {duplicate_count:,}")
        if map_path is not None:
            print(f"Pattern map saved: {Path(map_path)}")

    return snp_reduced, pattern_map


def restore_pattern_pairs(
    df_pairs: pd.DataFrame,
    pattern_map: Union[pd.DataFrame, PathLike],
    *,
    threshold: float = 0.0,
    include_within_pattern: bool = True,
    input_stage: str = "auto",
    preserve_pair_columns: bool = True,
    output_path: Optional[PathLike] = None,
    max_output_rows: Optional[int] = 50_000_000,
    chunk_output_rows: int = 1_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Restore representative EpiDis pairs to the original loci.

    Representative pairs are expanded as the Cartesian product of their two
    pattern groups.  When ``include_within_pattern`` is true, pairs belonging
    to the same exact pattern are also restored using ``within_pattern_v``.

    ``input_stage`` may be ``"pre_gwes"``, ``"post_gwes"``, or ``"auto"``.
    In ``"auto"`` mode an input containing ``geno_dis`` or ``LD`` is treated
    as post-GWES output.  For post-GWES input, those position-dependent values
    describe the representative pair, not every restored physical pair, so
    they are renamed to ``representative_geno_dis`` and
    ``representative_LD``. Other input columns are retained when
    ``preserve_pair_columns`` is true.

    Within-pattern pairs were not tested by GWES when GWES was run before
    restoration.  Therefore ``include_within_pattern=False`` is normally the
    appropriate choice for post-GWES restoration. If they are explicitly
    requested, they are selected only by ``within_pattern_v >= threshold`` and
    their unavailable pair metadata is stored as missing values.

    ``max_output_rows`` is a preflight memory guard. Set it to ``None`` only
    when the expanded result is known to fit in memory.
    """
    if not isinstance(df_pairs, pd.DataFrame):
        raise TypeError("df_pairs must be a pandas DataFrame")
    if int(chunk_output_rows) < 1:
        raise ValueError("chunk_output_rows must be positive")
    required_pair_columns = {"locus1", "locus2", "v"}
    missing_pair_columns = required_pair_columns.difference(df_pairs.columns)
    if missing_pair_columns:
        raise ValueError(f"df_pairs is missing columns: {sorted(missing_pair_columns)}")

    valid_input_stages = {"auto", "pre_gwes", "post_gwes"}
    if input_stage not in valid_input_stages:
        raise ValueError(
            f"input_stage must be one of {sorted(valid_input_stages)}, "
            f"got {input_stage!r}"
        )
    if input_stage == "auto":
        detected_stage = (
            "post_gwes"
            if {"geno_dis", "LD"}.intersection(df_pairs.columns)
            else "pre_gwes"
        )
    else:
        detected_stage = input_stage

    if detected_stage == "post_gwes" and include_within_pattern:
        warnings.warn(
            "include_within_pattern=True with post-GWES input adds pairs that "
            "were not evaluated by run_gwes_vp; they are filtered only by "
            "within_pattern_v >= threshold. Use False for a strictly GWES-"
            "selected restoration.",
            UserWarning,
            stacklevel=2,
        )

    if isinstance(pattern_map, (str, Path)):
        pattern_map = pd.read_parquet(pattern_map)
    elif not isinstance(pattern_map, pd.DataFrame):
        raise TypeError("pattern_map must be a DataFrame or Parquet path")

    required_map_columns = {
        "locus",
        "row_position",
        "pattern_group",
        "representative",
        "pattern_size",
    }
    missing_map_columns = required_map_columns.difference(pattern_map.columns)
    if missing_map_columns:
        raise ValueError(
            f"pattern_map is missing columns: {sorted(missing_map_columns)}"
        )
    if pattern_map["locus"].duplicated().any():
        raise ValueError("pattern_map contains duplicate loci")

    representative_info = pattern_map.drop_duplicates("representative")
    representative_sizes = representative_info.set_index("representative")[
        "pattern_size"
    ]
    size1 = df_pairs["locus1"].map(representative_sizes)
    size2 = df_pairs["locus2"].map(representative_sizes)
    if size1.isna().any() or size2.isna().any():
        raise ValueError("df_pairs contains loci that are not representatives in pattern_map")

    estimated_between = int(
        (size1.astype(np.int64) * size2.astype(np.int64)).sum()
    )
    estimated_within = 0
    if include_within_pattern:
        if "within_pattern_v" not in pattern_map.columns:
            raise ValueError(
                "pattern_map has no within_pattern_v; pass weight_ser when "
                "calling collapse_duplicate_patterns"
            )
        eligible = representative_info["within_pattern_v"] >= float(threshold)
        sizes = representative_info.loc[eligible, "pattern_size"].to_numpy(
            dtype=np.int64
        )
        estimated_within = int(np.sum(sizes * (sizes - 1) // 2))

    estimated_total = estimated_between + estimated_within
    if verbose:
        print(f"Estimated between-pattern pairs: {estimated_between:,}")
        print(f"Estimated within-pattern pairs: {estimated_within:,}")
        print(f"Estimated restored rows: {estimated_total:,}")
    if max_output_rows is not None and estimated_total > max_output_rows:
        raise MemoryError(
            f"Restoration would create about {estimated_total:,} rows, exceeding "
            f"max_output_rows={max_output_rows:,}. Restore a filtered result, "
            "increase the limit, or implement chunked output."
        )

    left_map = pattern_map[["locus", "row_position", "representative"]].rename(
        columns={
            "locus": "_original_locus1",
            "row_position": "_position1",
            "representative": "locus1",
        }
    )
    right_map = pattern_map[["locus", "row_position", "representative"]].rename(
        columns={
            "locus": "_original_locus2",
            "row_position": "_position2",
            "representative": "locus2",
        }
    )
    if preserve_pair_columns:
        pair_columns = list(df_pairs.columns)
    else:
        pair_columns = ["locus1", "locus2", "v"]

    pair_input = df_pairs[pair_columns].copy()
    position_column_renames = {}
    if detected_stage == "post_gwes":
        position_column_renames = {
            column: f"representative_{column}"
            for column in ("geno_dis", "LD")
            if column in pair_input.columns
        }
        pair_input = pair_input.rename(columns=position_column_renames)

    payload_columns = [
        column
        for column in pair_input.columns
        if column not in {"locus1", "locus2"}
    ]
    expansion_sizes = (
        size1.astype(np.int64) * size2.astype(np.int64)
    ).to_numpy(copy=False)
    pair_chunks = []
    chunk_start = 0
    chunk_rows = 0
    for row_index, expanded_rows in enumerate(expansion_sizes):
        expanded_rows = int(expanded_rows)
        if (
            row_index > chunk_start
            and chunk_rows + expanded_rows > int(chunk_output_rows)
        ):
            pair_chunks.append(pair_input.iloc[chunk_start:row_index])
            chunk_start = row_index
            chunk_rows = 0
        chunk_rows += expanded_rows
    if chunk_start < len(pair_input):
        pair_chunks.append(pair_input.iloc[chunk_start:])

    between_parts = []
    for pair_chunk in pair_chunks:
        expanded = (
            pair_chunk
            .merge(left_map, on="locus1", how="left", validate="many_to_many")
            .merge(right_map, on="locus2", how="left", validate="many_to_many")
        )
        left_first = expanded["_position1"].to_numpy() < expanded[
            "_position2"
        ].to_numpy()
        restored_chunk = pd.DataFrame({
            "locus1": np.where(
                left_first,
                expanded["_original_locus1"],
                expanded["_original_locus2"],
            ),
            "locus2": np.where(
                left_first,
                expanded["_original_locus2"],
                expanded["_original_locus1"],
            ),
        })
        for column in payload_columns:
            restored_chunk[column] = expanded[column].to_numpy(copy=False)
        if detected_stage == "post_gwes":
            restored_chunk["restoration_source"] = "between_pattern"
        between_parts.append(restored_chunk)
    restored_between = pd.concat(between_parts, ignore_index=True)
    parts = [restored_between]

    if include_within_pattern:
        within_rows = []
        for _, group in pattern_map.groupby("pattern_group", sort=False):
            if len(group) < 2:
                continue
            value = float(group["within_pattern_v"].iloc[0])
            if value < threshold:
                continue
            loci = group.sort_values("row_position")["locus"].tolist()
            within_rows.extend(
                (locus1, locus2, value)
                for locus1, locus2 in combinations(loci, 2)
            )
        if within_rows:
            restored_within = pd.DataFrame(
                within_rows, columns=["locus1", "locus2", "v"]
            )
            for column in payload_columns:
                if column != "v":
                    restored_within[column] = pd.NA
            if detected_stage == "post_gwes":
                restored_within["restoration_source"] = "within_pattern"
            restored_within = restored_within.reindex(
                columns=restored_between.columns
            )
            parts.append(restored_within)

    restored = pd.concat(parts, ignore_index=True)
    restored = restored.drop_duplicates(["locus1", "locus2"], keep="first")

    if output_path is not None:
        _write_parquet(restored, output_path)
        if verbose:
            print(f"Restored pairs saved: {Path(output_path)}")
    if verbose:
        print(f"Input stage: {detected_stage}")
        if position_column_renames:
            renamed = ", ".join(
                f"{old}->{new}" for old, new in position_column_renames.items()
            )
            print(f"Representative-only position columns renamed: {renamed}")
        print(f"Final restored rows: {len(restored):,}")
    return restored
