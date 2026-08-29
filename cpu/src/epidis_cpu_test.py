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
    first_row = df_full.iloc[0]
    if _is_binary_row(first_row):
        lg.info("Detected numeric-binary matrix → fast path (no conversion)")
        data_bin = df_full.values.astype(np.float16, copy=False)
        return pd.DataFrame(data_bin, index=df_full.index, columns=df_full.columns)

    lg.info(
        "Detected nucleotide/categorical matrix → converting rows to 0/1 (major-allele indicator)"
    )
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

# === 新增: 批量 pair 计算 (与现有公式一致) ===
@njit(fastmath=True, parallel=True, cache=True)
def compute_pairs_batch(pairs_ij: np.ndarray, X: np.ndarray, Wmat: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    对给定的 pairs_ij (shape=(M,2), 每行是 [i,j]) 计算每对 (i,j) 的 E 值。
    使用与 process_block_numba 同一公式:
      - dp = sum_k W[k] * X[i,k]
      - dq = sum_k W[k] * (1 - X[i,k])
      - data_p = sum_k (Wmat[j,k] * X[i,k])          # Wmat[j,k] = W[k] * X[j,k]
      - data_q = sum_k (Wmat[j,k] * (1 - X[i,k]))
      - 然后按现有的 p1/q1/clipping/JS/sqrt 计算
    """
    m = pairs_ij.shape[0]
    out = np.zeros(m, dtype=np.float32)
    N = X.shape[1]
    for t in prange(m):
        i = int(pairs_ij[t, 0])
        j = int(pairs_ij[t, 1])
        # dp, dq
        dp = 0.0
        dq = 0.0
        for k in range(N):
            xik = X[i, k]
            wk  = W[k]
            dp += wk * xik
            dq += wk * (1.0 - xik)
        if dp <= 0.0 or dq <= 0.0:
            out[t] = 0.0
            continue

        # data_p, data_q
        data_p = 0.0
        data_q = 0.0
        for k in range(N):
            val = Wmat[j, k]         # = W[k] * X[j,k]
            xik = X[i, k]
            data_p += val * xik
            data_q += val * (1.0 - xik)

        # 同 clamping
        p1 = data_p / dp
        if p1 < 1e-9: p1 = 1e-9
        if p1 > 1 - 1e-9: p1 = 1 - 1e-9
        p0 = 1.0 - p1

        q1 = data_q / dq
        if q1 < 1e-9: q1 = 1e-9
        if q1 > 1 - 1e-9: q1 = 1 - 1e-9
        q0 = 1.0 - q1

        size_p = dp / (dp + dq)
        size_q = 1.0 - size_p

        st1_0 = p1 + 1e-27; st1_1 = p0 + 1e-27
        st2_0 = q1 + 1e-27; st2_1 = q0 + 1e-27
        st3_0 = st1_0 * size_p + st2_0 * size_q
        st3_1 = st1_1 * size_p + st2_1 * size_q
        r1_0 = st1_0 / st3_0
        r1_1 = st1_1 / st3_1
        r2_0 = st2_0 / st3_0
        r2_1 = st2_1 / st3_1
        js = (st1_0 * np.log2(r1_0) + st1_1 * np.log2(r1_1)) * size_p \
           + (st2_0 * np.log2(r2_0) + st2_1 * np.log2(r2_1)) * size_q
        out[t] = np.sqrt(js) if js > 0.0 else 0.0
    return out

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

def _pairs_to_row_indices(
    pairs: "pd.DataFrame | np.ndarray | List[Tuple]]",
    index_arr: np.ndarray,
    *,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    将用户给的 pairs（locus 值）转换为 data_df 行位置 (i,j)：
      - 若 pairs 是 DataFrame，优先使用列名 'locus1','locus2'（不区分大小写）
      - 否则视为二维数组/列表
      - locus 值优先直接在 index_arr 中查找；若是 'chr*_N' 风格，尝试提取 N 再查找
    返回 shape=(M,2) 的 np.int64 数组；会自动丢弃无法匹配的行，并打印日志。
    """
    lg = logger or get_logger()
    # 取 pairs 数组
    if isinstance(pairs, pd.DataFrame):
        colmap = {c.lower(): c for c in pairs.columns}
        if not {"locus1", "locus2"}.issubset(set(colmap)):
            raise ValueError("pairs DataFrame 需要包含列 'locus1','locus2'")
        arr = pairs[[colmap["locus1"], colmap["locus2"]]].to_numpy()
    else:
        arr = np.asarray(pairs)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("pairs 必须是 (M,2) 的数组/列表，或包含 ['locus1','locus2'] 的 DataFrame")

    # 建立 {locus_value -> 行位置} 的映射
    pos_map = {}
    for ridx, loc in enumerate(index_arr):
        pos_map[loc] = ridx

    def _to_idx_one(x):
        # 先直接匹配
        if x in pos_map:
            return pos_map[x]
        # 若是字符串 'chr..._N'，提取 N 再查
        if isinstance(x, str):
            if "_" in x:
                tail = x.split("_")[-1]
                try:
                    v = int(tail)
                    if v in pos_map:
                        return pos_map[v]
                except Exception:
                    pass
            # 纯数字字符串
            try:
                v = int(x)
                if v in pos_map:
                    return pos_map[v]
            except Exception:
                return -1
        # 数值型转 int 再查
        if isinstance(x, (np.integer, int, float, np.floating)):
            try:
                v = int(x)
                if v in pos_map:
                    return pos_map[v]
            except Exception:
                return -1
        return -1

    out = np.empty_like(arr, dtype=np.int64)
    valid = np.ones(arr.shape[0], dtype=bool)
    for i in range(arr.shape[0]):
        ii = _to_idx_one(arr[i,0])
        jj = _to_idx_one(arr[i,1])
        out[i,0] = ii; out[i,1] = jj
        if ii < 0 or jj < 0:
            valid[i] = False

    dropped = int((~valid).sum())
    if dropped > 0:
        lg.warning("pairs 中有 %d 条无法与矩阵索引对齐，已丢弃。", dropped)
    return out[valid]
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

def run_epidis_for_pairs(
    data_df: pd.DataFrame,
    weight_ser: "pd.Series | pd.DataFrame",
    pairs: "pd.DataFrame | np.ndarray | List[Tuple[int,int]]",
    *,
    n_jobs: int = 1,
    threshold: float | None = None,   # 若给阈值则过滤
    logger: Optional[logging.Logger] = None,
    prefix: Optional[str] = None,
    out_format: str = "txt",
) -> pd.DataFrame:
    """
    只计算给定 pair 列表的 E 值（不做全上三角扫描）。

    Parameters
    ----------
    data_df : (T,N) 0/1 DataFrame，行=位点，列=样本（若非 0/1 会自动转）
    weight_ser : Series 或 单列 DataFrame，索引为样本名
    pairs : DataFrame(['locus1','locus2']) 或 二维数组/列表 (M,2)
            允许 locus 使用整数或 'chr*_N' 字符串，函数会尽力对齐到行索引。
    n_jobs : int
        目前 Numba 内部并行，外层不再使用 joblib 分发（设多少都不影响结果，仅为接口一致保留）。
    threshold : float | None
        若给定，则返回时只保留 v >= threshold 的 pair。
    prefix : str | None
        若提供，则落盘到 {prefix}_Epi_pairs_selected.(txt|parquet)

    Returns
    -------
    DataFrame with columns ['locus1','locus2','v']
    """
    lg = logger or get_logger()
    # 权重整理
    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError("weight_ser DataFrame must have exactly one column.")
        weight_ser = weight_ser.iloc[:, 0]
    weight_ser = weight_ser.reindex(data_df.columns).astype(np.float32)

    # 矩阵确保 0/1
    if not _is_binary_row(data_df.iloc[0, :]):
        lg.info("Input matrix is not numeric-binary → converting to 0/1 inside run_epidis_for_pairs")
        data_df = convert_snp_to_binary(data_df, n_jobs=1, convert_chunksize=20_000, logger=lg)

    # 准备底层数组
    X     = np.ascontiguousarray(data_df.to_numpy(dtype=np.float32, copy=False))
    W     = np.ascontiguousarray(weight_ser.to_numpy(dtype=np.float32, copy=False))
    Wmat  = np.ascontiguousarray(X * W.reshape(1, -1), dtype=np.float32)
    index_arr = np.ascontiguousarray(data_df.index.to_numpy(dtype=np.int64, copy=False))

    # pairs -> 行位置
    pairs_idx = _pairs_to_row_indices(pairs, index_arr, logger=lg)
    if pairs_idx.size == 0:
        lg.warning("没有任何可用的 pairs 与矩阵索引对齐，返回空结果。")
        return pd.DataFrame(columns=["locus1","locus2","v"])

    # 计算
    lg.info("Computing selected pairs: %d", pairs_idx.shape[0])
    v_vals = compute_pairs_batch(pairs_idx.astype(np.int64), X, Wmat, W)

    # 输出 DataFrame，locus 用“原索引值”（而不是行号）
    loc1 = index_arr[pairs_idx[:,0]]
    loc2 = index_arr[pairs_idx[:,1]]
    df_out = pd.DataFrame({"locus1": loc1, "locus2": loc2, "v": v_vals})

    # 过滤
    if threshold is not None:
        df_out = df_out[df_out["v"] >= float(threshold)].reset_index(drop=True)

    # 落盘
    if prefix:
        if out_format not in {"txt","parquet"}:
            raise ValueError("out_format must be 'txt' or 'parquet'")
        if out_format == "txt":
            out = f"{prefix}_Epi_pairs_selected.txt"
            df_out.to_csv(out, sep="\t", index=False, header=False)
        else:
            out = f"{prefix}_Epi_pairs_selected.parquet"
            df_out.to_parquet(out, index=False)
        lg.info("Selected-pairs results saved: %s (%d rows)", out, len(df_out))
    else:
        lg.info("Selected-pairs results in memory: %d rows", len(df_out))

    return df_out


def calculate_complete_pair_scores(
    data_df: pd.DataFrame,
    weight_ser: "pd.Series | pd.DataFrame",
    pairs: "pd.DataFrame | np.ndarray | List[Tuple[int, int]]",
    *,
    label: str = "Dataset",
    n_jobs: int = 1,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Calculate every requested pair without silently dropping pair keys."""
    lg = logger or get_logger()
    pair_columns = ["locus1", "locus2"]

    if not isinstance(data_df, pd.DataFrame):
        raise TypeError("data_df must be a pandas DataFrame")
    if data_df.empty:
        raise ValueError("data_df cannot be empty")
    if not data_df.index.is_unique:
        raise ValueError("data_df index must contain unique locus identifiers")

    if isinstance(pairs, pd.DataFrame):
        missing_columns = [c for c in pair_columns if c not in pairs.columns]
        if missing_columns:
            raise ValueError(
                f"pairs is missing required columns: {missing_columns}"
            )
        requested = pairs.loc[:, pair_columns].copy()
    else:
        pair_array = np.asarray(pairs)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2)")
        requested = pd.DataFrame(pair_array, columns=pair_columns)

    if requested.empty:
        return pd.DataFrame(columns=["locus1", "locus2", "v"])

    locus1 = requested["locus1"].to_numpy()
    locus2 = requested["locus2"].to_numpy()
    requested["locus1"] = np.minimum(locus1, locus2)
    requested["locus2"] = np.maximum(locus1, locus2)
    requested = requested.drop_duplicates(pair_columns).reset_index(drop=True)

    if isinstance(weight_ser, pd.DataFrame):
        if weight_ser.shape[1] != 1:
            raise ValueError(
                "weight_ser DataFrame must have exactly one column"
            )
        weight_ser = weight_ser.iloc[:, 0]

    weights = weight_ser.reindex(data_df.columns).astype(np.float32)
    if weights.isna().any() or not np.isfinite(weights.to_numpy()).all():
        raise ValueError(
            f"{label}: sample weights are missing or non-finite"
        )

    if not _is_binary_row(data_df.iloc[0, :]):
        lg.info(
            "%s: converting the SNP matrix to numeric binary values",
            label,
        )
        data_binary = convert_snp_to_binary(
            data_df,
            n_jobs=max(1, n_jobs),
            convert_chunksize=20_000,
            logger=lg,
        )
    else:
        data_binary = data_df

    data_index = pd.Index(data_binary.index)
    row1 = data_index.get_indexer(requested["locus1"])
    row2 = data_index.get_indexer(requested["locus2"])
    valid = (row1 >= 0) & (row2 >= 0)

    if not valid.all():
        missing_pairs = requested.loc[~valid, pair_columns]
        missing_loci = pd.Index(
            np.concatenate(
                [
                    missing_pairs["locus1"].to_numpy(),
                    missing_pairs["locus2"].to_numpy(),
                ]
            )
        )
        missing_loci = missing_loci[~missing_loci.isin(data_index)].unique()
        raise ValueError(
            f"{label}: {len(missing_pairs)} pairs reference loci absent "
            f"from data_df; missing_loci={len(missing_loci)}, "
            f"locus_examples={missing_loci[:10].tolist()}, "
            f"pair_examples={missing_pairs.head(10).to_dict('records')}"
        )

    X = np.ascontiguousarray(
        data_binary.to_numpy(dtype=np.float32, copy=False)
    )
    W = np.ascontiguousarray(weights.to_numpy(dtype=np.float32, copy=False))
    Wmat = np.ascontiguousarray(X * W.reshape(1, -1), dtype=np.float32)
    pairs_idx = np.ascontiguousarray(
        np.column_stack([row1, row2]),
        dtype=np.int64,
    )

    lg.info("%s: computing all %d requested pairs", label, len(requested))
    values = compute_pairs_batch(pairs_idx, X, Wmat, W)

    if len(values) != len(requested):
        raise RuntimeError(
            f"{label}: calculator returned {len(values)} values for "
            f"{len(requested)} requested pairs"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            f"{label}: calculator returned "
            f"{int((~np.isfinite(values)).sum())} non-finite values"
        )

    result = requested.copy()
    result["v"] = values
    lg.info(
        "%s: returned all %d requested pairs (%d zero scores)",
        label,
        len(result),
        int((result["v"] == 0).sum()),
    )
    return result
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
