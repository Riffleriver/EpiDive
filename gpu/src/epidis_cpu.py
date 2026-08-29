# epidis_cpu_normal.py
# -*- coding: utf-8 -*-
"""
可直接 import 的 **纯 CPU + Numba** 版本（normal 模式）。
基于你提供的 `R_EpiDis_normal_tttt.py` 的**核心公式**：`compute_e_for_row` 与 `process_block_numba`，
并封装成可调用函数，默认返回 pandas.DataFrame；可选写出到文件。

提供的高层接口：
- load_inputs(snp_path, weight_path): 读取 SNP 矩阵与权重，返回 (data_f32, weight_f32, index_arr)
- run_epidis_normal_cpu(data_f32, weight_f32, index_arr, ...): 用 Numba 按上三角分片并行计算，返回 DataFrame
- run_from_files(snp_path, weight_path, ...): 便捷入口（读文件 + 计算）

特性：
- 与你原脚本数值保持一致：完全复用你给的 Numba 核心实现
- 支持可选 memmap（减少内存峰值，适合超大矩阵）
- joblib 进程并行；可调 slices_per_job 控制切片粒度

注意：
- 输入矩阵应为数值 0/1（或 0/1 浮点）编码；若是核苷酸字符，需要先自行转换。
- 建议将底层 BLAS/OMP 线程数设为 1，避免与 n_jobs 竞争：
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"""
from __future__ import annotations
from typing import Optional, List, Tuple

import os
import math
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from numba import njit, prange

# ---------------------------
# logging
# ---------------------------

def get_logger(name: str = "EpiDisCPU", verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ---------------------------
# I/O
# ---------------------------
def _is_binary_row(row: pd.Series) -> bool:
    if not np.issubdtype(row.dtype, np.number):
        return False
    vals = pd.unique(row.values)
    try:
        fvals = pd.Series(vals).astype(float).tolist()
    except Exception:
        return False
    return set(fvals).issubset({0.0, 1.0})

def _transform_row_to_binary(row: pd.Series) -> np.ndarray:
    vc = row.value_counts()
    major = vc.idxmax()
    return (row == major).astype(np.float16).to_numpy(dtype=np.float16)


def _convert_single_character_matrix(df_full: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Vectorize major-allele conversion for one-character SNP data.

    Ties preserve the legacy ``value_counts().idxmax()`` behavior by selecting
    the allele whose first occurrence in the row is earliest. ``None`` asks
    the caller to use the generic Pandas fallback for unsupported input.
    """
    values = df_full.to_numpy(copy=False)
    probe = values[: min(len(df_full), 1024)].reshape(-1)
    for value in probe:
        if pd.isna(value) or not isinstance(value, str) or len(value) != 1:
            return None

    try:
        codes = np.asarray(values, dtype="S1").view(np.uint8)
        codes = codes.reshape(values.shape)
    except (TypeError, ValueError):
        return None

    return _major_indicator_from_codes(codes, df_full.index, df_full.columns)


def _major_indicator_from_codes(codes, index, columns) -> pd.DataFrame:
    """Return the legacy per-row major-allele indicator from uint8 codes."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    row_count, column_count = codes.shape
    best_count = np.full(row_count, -1, dtype=np.int64)
    best_first = np.full(row_count, column_count, dtype=np.int64)
    major_code = np.empty(row_count, dtype=np.uint8)

    for allele_code in np.unique(codes):
        allele_mask = codes == allele_code
        allele_count = np.count_nonzero(allele_mask, axis=1)
        first_position = np.argmax(allele_mask, axis=1).astype(np.int64)
        first_position[allele_count == 0] = column_count
        replace = (allele_count > best_count) | (
            (allele_count == best_count) & (first_position < best_first)
        )
        best_count[replace] = allele_count[replace]
        best_first[replace] = first_position[replace]
        major_code[replace] = allele_code

    binary = (codes == major_code[:, None]).astype(np.float16, copy=False)
    return pd.DataFrame(binary, index=index, columns=columns)

def convert_snp_to_binary(
    df_full: pd.DataFrame,
    *,
    n_jobs: int = -1,
    convert_chunksize: int = 20_000,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    将 SNP 矩阵 DataFrame 转为 0/1（float16）。如果本来就是0/1数值矩阵，则直接返回快路径。
    行=位点，列=样本；保留原索引和列名。
    """
    lg = logger or get_logger()
    if df_full.attrs.get("epidive_allele_uint8", False):
        lg.info("Using cached uint8 allele encoding for subgroup conversion")
        return _major_indicator_from_codes(
            df_full.to_numpy(dtype=np.uint8, copy=False),
            df_full.index,
            df_full.columns,
        )

    first_row = df_full.iloc[0]
    if _is_binary_row(first_row):
        lg.info("Detected numeric-binary matrix → fast path (no conversion)")
        data_bin = df_full.values.astype(np.float16, copy=False)
        return pd.DataFrame(data_bin, index=df_full.index, columns=df_full.columns)

    lg.info(
        "Detected nucleotide/categorical matrix → converting rows to 0/1 (major-allele indicator)"
    )
    vectorized = _convert_single_character_matrix(df_full)
    if vectorized is not None:
        lg.info("Using vectorized single-character allele conversion")
        return vectorized

    lg.info("Using generic parallel Pandas conversion fallback")
    n_rows = len(df_full)

    if n_rows <= convert_chunksize:
        block_df = df_full.apply(
            lambda r: _transform_row_to_binary(r), axis=1, result_type="expand"
        )
        block_df.columns = df_full.columns
        block_df.index = df_full.index
        return block_df.astype(np.float16, copy=False)

    bounds = list(range(0, n_rows, convert_chunksize))
    chunks = [df_full.iloc[i : i + convert_chunksize] for i in bounds]

    def _process_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
        block_df = df_chunk.apply(
            lambda r: _transform_row_to_binary(r), axis=1, result_type="expand"
        )
        block_df.columns = df_chunk.columns
        block_df.index = df_chunk.index
        return block_df

    lg.info("Splitting into %d chunks (chunksize=%d)", len(chunks), convert_chunksize)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_chunk)(ck) for ck in tqdm(chunks, desc="Converting chunks")
    )

    out_df = pd.concat(results, axis=0)
    return out_df.astype(np.float16, copy=False)


# ========= Updated: load_inputs 只读文件+交给 convert_snp_to_binary =========
def load_inputs(
    snp_path: str,
    weight_path: str,
    *,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = -1,
    convert_chunksize: int = 20_000,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    读取 SNP 矩阵和权重；返回:
      - data_df: 行=位点(索引), 列=样本 的 0/1 DataFrame (float16/float32)
      - w_ser : 与列同序的权重 Series (float32，索引=样本名)
    """
    if n_jobs <= 0:
        n_jobs = cpu_count()
    lg = logger or get_logger()
    lg.info("Loading weight: %s", weight_path)
    df_w = pd.read_csv(weight_path, index_col=0)
    if "wet" in df_w.columns:
        w_ser = df_w["wet"].astype(np.float32)
    else:
        first_col = df_w.columns[0]
        w_ser = df_w[first_col].astype(np.float32)
        lg.warning("Column 'wet' not found, using first column '%s'", first_col)

    lg.info("Loading SNP: %s", snp_path)
    ext = os.path.splitext(snp_path)[1].lower()
    if ext == ".parquet":
        df_full = pd.read_parquet(snp_path)
    else:
        df_full = pd.read_csv(snp_path, sep="\t", index_col=0)

    data_df = convert_snp_to_binary(
        df_full, n_jobs=n_jobs, convert_chunksize=convert_chunksize, logger=lg
    )
    try:
        w_ser = w_ser.reindex(data_df.columns).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Weight index must cover all SNP matrix columns. Detail: {e}")

    return data_df, w_ser


# ---------------------------
# ---------------------------

@njit(parallel=True, fastmath=True, cache=True)
def compute_e_for_row(s1, submatrix, dp, dq):
    m, n = submatrix.shape
    e = np.empty(m, dtype=np.float32)
    size_p = dp / (dp + dq)
    size_q = 1 - size_p
    for j in prange(m):
        row = submatrix[j]
        data_p = 0.0
        data_q = 0.0
        for k in range(n):
            val = row[k]
            data_p += val * s1[k]
            data_q += val * (1 - s1[k])
        p1 = max(min(data_p / dp, 1 - 1e-9), 1e-9)
        p0 = 1 - p1
        q1 = max(min(data_q / dq, 1 - 1e-9), 1e-9)
        q0 = 1 - q1
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
        js = (st1_0 * np.log2(r1_0) + st1_1 * np.log2(r1_1)) * size_p \
           + (st2_0 * np.log2(r2_0) + st2_1 * np.log2(r2_1)) * size_q
        e[j] = np.sqrt(js) if js > 0 else 0.0
    return e

@njit(fastmath=True)
def process_block_numba(arr, wmat, wgt, start, end, thod, index_arr):
    """在区间 [start, end) 上计算 EpiDis（与原脚本一致）。
    arr:   (T,N)  原始 SNP 0/1
    wmat:  (T,N)  arr * wgt（逐列加权后的矩阵）
    wgt:   (N,)   权重向量（此函数中仅用于 dp/dq）
    """
    results = []
    for i in range(start, end):
        s1 = arr[i]
        dp = 0.0
        dq = 0.0
        for k in range(s1.shape[0]):
            dp += wgt[k] * s1[k]
            dq += wgt[k] * (1 - s1[k])
        if dp <= 0 or dq <= 0:
            continue
        sub = wmat[i+1:]
        e_vals = compute_e_for_row(s1, sub, dp, dq)
        for j in range(e_vals.shape[0]):
            if e_vals[j] >= thod:
                results.append((index_arr[i], index_arr[i+1+j], e_vals[j]))
    return results


# ---------------------------
# 分片与并行
# ---------------------------

# def make_triangular_blocks(n: int, n_jobs: int, slices_per_job: int = 8) -> List[Tuple[int, int]]:
#     total_slices = max(1, n_jobs) * max(1, slices_per_job)
#     bounds = [int(n * (1.0 - math.sqrt(1.0 - k / total_slices))) for k in range(total_slices + 1)]
#     bounds = sorted(set(bounds))
#     return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

def make_triangular_blocks(n, n_jobs, slices_per_job=16, min_len=32):
    total = max(1, n_jobs) * max(1, slices_per_job)
    bounds = [int(n * (1.0 - math.sqrt(1.0 - k / total))) for k in range(total + 1)]
    merged, last = [], bounds[0]
    for b in bounds[1:]:
        if b - last < min_len:
            continue
        merged.append((last, b)); last = b
    if last < n: merged.append((last, n))
    return merged

# ---------------------------
# 可选 memmap（大矩阵更稳）
# ---------------------------
_global_mem = {"arr": None, "wmat": None, "w": None, "idx": None}

def _build_memmaps(arr: np.ndarray, w_mat: np.ndarray, memmap_dir: str) -> tuple[str, str]:
    os.makedirs(memmap_dir, exist_ok=True)
    f_arr = os.path.join(memmap_dir, "data_arr.memmap")
    f_w   = os.path.join(memmap_dir, "data_w.memmap")
    mm1 = np.memmap(f_arr, dtype='float32', mode='w+', shape=arr.shape)
    mm1[:] = arr; mm1.flush()
    mm2 = np.memmap(f_w, dtype='float32', mode='w+', shape=w_mat.shape)
    mm2[:] = w_mat; mm2.flush()
    return f_arr, f_w


def _worker_memmap(start: int, end: int, thod: float,
                   file_arr: str, file_w: str, shape_arr: tuple, shape_w: tuple,
                   w_vec: np.ndarray, idx_arr: np.ndarray) -> np.ndarray:
    if _global_mem["arr"] is None:
        _global_mem["arr"]  = np.memmap(file_arr, dtype='float32', mode='r', shape=shape_arr)
        _global_mem["wmat"] = np.memmap(file_w,  dtype='float32', mode='r', shape=shape_w)
        _global_mem["w"]    = w_vec.astype(np.float32, copy=False)
        _global_mem["idx"]  = idx_arr.astype(np.int64, copy=False)
    blk = process_block_numba(_global_mem["arr"], _global_mem["wmat"], _global_mem["w"],
                              start, end, thod, _global_mem["idx"])
    if blk:
        return np.array(blk, dtype=np.float32)
    else:
        return np.empty((0, 3), dtype=np.float32)


# ---------------------------
# 主计算接口
# ---------------------------
# ========= Updated: run_epidis_normal_cpu 复用 convert_snp_to_binary =========
def run_epidis_normal_cpu(
    data_df: pd.DataFrame,          # 注意：这里现在明确要求 DataFrame（行=位点，列=样本）
    weight_ser: Union[pd.Series, pd.DataFrame],
    *,
    threshold: float = 0.2,
    n_jobs: int = 1,
    slices_per_job: int = 8,
    use_memmap: bool = False,
    memmap_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    out_format: str = "txt",        # "txt" | "parquet"
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    lg = logger or get_logger()
    if n_jobs<=0:
        n_jobs = cpu_count()

    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError("weight_ser DataFrame must have exactly one column.")
        weight_ser = weight_ser.iloc[:, 0]
    weight_ser = weight_ser.reindex(data_df.columns).astype(np.float32)

    if not _is_binary_row(data_df.iloc[0, :]):
        lg.info("Input matrix is not numeric-binary → converting to 0/1 inside run_epidis_normal_cpu")
        data_df = convert_snp_to_binary(
            data_df, n_jobs=max(1, n_jobs), convert_chunksize=20_000, logger=lg
        )

    # X = data_df.to_numpy(dtype=np.float32, copy=False)
    # w = weight_ser.to_numpy(dtype=np.float32, copy=False)
    # index_arr = data_df.index.to_numpy(dtype=np.int64, copy=False)
    # w_mat = (X * w.reshape(1, -1)).astype(np.float32, copy=False)

    X = np.ascontiguousarray(data_df.to_numpy(dtype=np.float32, copy=False))
    w = np.ascontiguousarray(weight_ser.to_numpy(dtype=np.float32, copy=False))
    w_mat = np.ascontiguousarray(X * w.reshape(1, -1), dtype=np.float32)
    index_arr = np.ascontiguousarray(data_df.index.to_numpy(dtype=np.int64, copy=False))

    T, N = X.shape
    if use_memmap and (T * N < 2e8):
        lg.info("Matrix is moderate; disable memmap to avoid I/O overhead.")
        use_memmap = False

    lg.info("EpiDis CPU normal | T=%d, N=%d, n_jobs=%d, slices/job=%d, threshold=%.3f",
            T, N, n_jobs, slices_per_job, threshold)

    blocks = make_triangular_blocks(T, max(1, n_jobs), slices_per_job=slices_per_job)

    start_t = time.time()
    results_arrays: List[np.ndarray] = []

    if use_memmap:
        if not memmap_dir:
            raise ValueError("memmap_dir is required when use_memmap=True")
        file_arr, file_w = _build_memmaps(X, w_mat, memmap_dir)

        def _worker_memmap_local(s, e):
            return _worker_memmap(s, e, float(threshold), file_arr, file_w, X.shape, w_mat.shape, w, index_arr)

        with Parallel(n_jobs=n_jobs, backend='loky', batch_size='auto') as pool:
            results_arrays = pool(delayed(_worker_memmap_local)(s, e) for (s, e) in blocks)
    else:
        def _worker_local(s, e):
            blk = process_block_numba(X, w_mat, w, s, e, float(threshold), index_arr)
            return np.array(blk, dtype=np.float32) if blk else np.empty((0, 3), dtype=np.float32)

        if n_jobs == 1:
            results_arrays = [_worker_local(s, e) for (s, e) in blocks]
        else:
            with Parallel(n_jobs=n_jobs, backend='loky', batch_size='auto') as pool:
                results_arrays = pool(delayed(_worker_local)(s, e) for (s, e) in blocks)

    dt = time.time() - start_t
    lg.info("EpiDis done in %.3f s", dt)

    if results_arrays:
        all_pairs = np.vstack([r for r in results_arrays if r.size > 0])
    else:
        all_pairs = np.empty((0, 3), dtype=np.float32)
    df_out = pd.DataFrame(all_pairs, columns=["locus1", "locus2", "v"])
    if not df_out.empty:
        df_out["locus1"] = df_out["locus1"].astype(np.int64)
        df_out["locus2"] = df_out["locus2"].astype(np.int64)

    if prefix:
        if out_format not in {"txt", "parquet"}:
            raise ValueError("out_format must be 'txt' or 'parquet'")
        if out_format == "txt":
            out = f"{prefix}_Epi_pairs.txt"
            df_out.to_csv(out, sep="\t", index=False, header=False)
        else:
            out = f"{prefix}_Epi_pairs.parquet"
            df_out.to_parquet(out, index=False)
        lg.info("Results saved to %s (%d rows)", out, len(df_out))
    else:
        lg.info("Results kept in memory: %d rows", len(df_out))

    return df_out

# ---------------------------
# 便捷入口：直接从文件运行
# ---------------------------
def run_from_files(
    snp_path: str,
    weight_path: str,
    *,
    threshold: float = 0.2,
    n_jobs: int = 1,
    slices_per_job: int = 8,
    use_memmap: bool = True,
    memmap_dir: str = ".",
    prefix: str | None = None,
    out_format: str = "txt",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    data_df, weight_ser = load_inputs(snp_path, weight_path, logger=logger)
    return run_epidis_normal_cpu(
        data_df, weight_ser,
        threshold=threshold,
        n_jobs=n_jobs,
        slices_per_job=slices_per_job,
        use_memmap=use_memmap,
        memmap_dir=memmap_dir,
        prefix=prefix,
        out_format=out_format,
        logger=logger,
    )
