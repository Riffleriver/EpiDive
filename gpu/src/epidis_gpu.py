#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triangular Block-Processing for EpiDis with GPU Acceleration (CuPy, float32)

Version: 1.1.3
Author: rivers_imac
Created on: 2025-04-30
"""
import os
def _prepend_path(var: str, path: str):
    if not path:
        return
    cur = os.environ.get(var, "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if path not in parts:
        os.environ[var] = (path + (os.pathsep + cur if cur else ""))

def configure_cuda_env_from_conda():
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    os.environ["CUDA_PATH"] = prefix
    os.environ["CUDA_HOME"] = prefix
    _prepend_path("LD_LIBRARY_PATH", os.path.join(prefix, "lib"))
    _prepend_path("LD_LIBRARY_PATH", os.path.join(prefix, "lib64"))
    inc = os.path.join(prefix, "include")
    if os.path.exists(os.path.join(inc, "cuda_fp16.h")):
        os.environ.setdefault("CUPY_NVRTC_INCLUDE_DIRS", inc)
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join("/tmp", os.environ.get("USER", "user"), "cupy_cache"))

configure_cuda_env_from_conda()

import sys
import argparse
import logging
import shutil
import tempfile
import numpy as np
import pandas as pd
import cupy as cp
from multiprocessing import get_context
from joblib import Parallel, delayed
import time

# Globals set by init_worker
DATA_FP32      = None
DATA_WEIGHTED  = None
WEIGHT_FP32    = None
INDEX_ARR      = None
T2             = None
THRESHOLD      = None
BLOCK_SIZE     = None
STREAM_DIR     = None
STREAM_FMT     = None
STREAM_BATCH_ROWS = None


def init_logger(level=logging.INFO):
    logger = logging.getLogger("EpiDisGPU")
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)
    return logger


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU EpiDis computation with CuPy (float32)."
    )
    p.add_argument("--snp",      required=True, help="Path to SNP matrix TSV/Parquet")
    p.add_argument("--w",        required=True, help="Path to weight CSV (must contain 'wet')")
    p.add_argument("--outlier",  required=True, type=float, help="EpiDis threshold")
    p.add_argument("--prefix",   required=True, help="Output file prefix")
    p.add_argument("--t1",       type=int, default=0,   help="Start SNP index (default 0)")
    p.add_argument("--t2",       type=int, default=-1,  help="End SNP index (default all)")
    p.add_argument("--format",   choices=["txt","parquet"], default="txt", help="Output format")
    p.add_argument("--block_size", type=int, default=10000, help="Block size for tiling")
    p.add_argument("--convert_chunksize", type=int, default=20000, help="Rows per chunk for SNP→binary conversion (default 20000)")
    p.add_argument("--maf",      type=float, default=None,
                   help="Keep loci with minor-allele frequency >= maf. Omit to disable.")
    return p.parse_args()


def load_data(snp_path, weight_path, chunksize, logger, maf: float | None = None):
    logger.info("Loading SNP matrix from %s", snp_path)
    _, ext = os.path.splitext(snp_path)

    def is_binary_row(row):
        if not np.issubdtype(row.dtype, np.number):
            return False
        vals = np.unique(row.values)
        return set(vals.tolist()).issubset({0, 1})

    def _transform_row(row):
        vc = row.value_counts()
        major = vc.idxmax()
        return (row == major).astype(np.float32)

    def _process_chunk(df_chunk):
        df_bin = df_chunk.apply(_transform_row, axis=1)
        return df_bin.values, df_bin.index.to_list()

    if ext.lower() == ".parquet":
        df_full = pd.read_parquet(snp_path)
        first_row = df_full.iloc[0]
        binary_input = is_binary_row(first_row)
        if binary_input:
            logger.info("Detected numeric binary Parquet → skip conversion")
            data_bin  = df_full.values.astype(np.float32)
            index_arr = df_full.index.to_numpy(dtype=object)
        else:
            logger.info("Detected nucleotide Parquet → convert in parallel")
            n = len(df_full)
            bounds = range(0, n, chunksize)
            chunks = [df_full.iloc[i:i+chunksize] for i in bounds]
            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(_process_chunk)(chunk) for chunk in chunks
            )
            blocks, idx_lists = zip(*results)
            data_bin  = np.vstack(blocks)
            index_arr = np.concatenate([np.array(ix, dtype=object) for ix in idx_lists])

    else:
        reader = pd.read_csv(snp_path, sep="\t", index_col=0, chunksize=chunksize)
        first_chunk = next(reader)
        first_row   = first_chunk.iloc[0]
        binary_input = is_binary_row(first_row)
        if binary_input:
            logger.info("Detected numeric binary TSV → skip conversion")
            blocks, idx_lists = [first_chunk.values.astype(np.float32)], [first_chunk.index.to_list()]
            for chunk in reader:
                blocks.append(chunk.values.astype(np.float32))
                idx_lists.append(chunk.index.to_list())
            data_bin  = np.vstack(blocks)
            index_arr = np.concatenate([np.array(ix, dtype=object) for ix in idx_lists])
        else:
            logger.info("Detected nucleotide TSV → convert in parallel")
            all_chunks = [first_chunk] + list(reader)
            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(_process_chunk)(chunk) for chunk in all_chunks
            )
            blocks, idx_lists = zip(*results)
            data_bin  = np.vstack(blocks)
            index_arr = np.concatenate([np.array(ix, dtype=object) for ix in idx_lists])

    logger.info("Final binary SNP matrix shape: %d loci × %d samples", *data_bin.shape)

    # —— 读取权重 —— #
    logger.info("Loading weights from %s", weight_path)
    df_w = pd.read_csv(weight_path, index_col=0)
    if 'wet' not in df_w.columns:
        logger.error("Weight file must contain 'wet' column.")
        sys.exit(1)
    # df_w = df_w.reindex(data_bin.columns).astype(np.float32)
    weight_arr = df_w['wet'].values.astype(np.float32)
    if weight_arr.size != data_bin.shape[1]:
        raise ValueError(
            f"Weight length {weight_arr.size} does not match sample count {data_bin.shape[1]}"
        )
    if not np.isfinite(weight_arr).all():
        raise ValueError("Weights must contain only finite values")
    if np.any(weight_arr < 0) or float(weight_arr.sum(dtype=np.float64)) <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum")
    logger.info("Weight vector length: %d", weight_arr.size)

    # ====== 新增：行过滤（按每行1的比例 p1） ======
    # data_bin 已是 0/1，通常无 NaN；这里仍用 nansum 以防万一
    if maf is not None:
        if not (0.0 <= float(maf) <= 0.5):
            raise ValueError("maf must be between 0 and 0.5")
        n_rows, n_cols = data_bin.shape
        not_nan = ~np.isnan(data_bin)
        n_eff = not_nan.sum(axis=1)                       # 每行有效样本数
        sum1  = np.nansum(data_bin, axis=1)               # 每行1的数量
        # p1 = 每行1的比例；对 n_eff=0 的行，p1 设为 0 以便后续统一处理
        p1 = np.divide(sum1, n_eff, out=np.zeros_like(sum1, dtype=np.float32), where=(n_eff > 0))

        # Allele 1 is the major allele after conversion, so MAF=1-p1.
        thr = 1.0 - float(maf)
        keep = (n_eff > 0) & (p1 <= thr)

        kept = int(keep.sum())
        logger.info(
            "Applied row filter: keep MAF >= %.4f (p1 <= %.4f) → kept %d/%d rows",
            float(maf), thr, kept, int(n_rows)
        )

        # 应用到矩阵与索引
        data_bin  = data_bin[keep, :]
        index_arr = index_arr[keep]
    # ====== 新增结束 ======
    return data_bin, weight_arr, index_arr


def write_in_chunks(df, filename, chunksize=100000, sep="\t", logger=None):
    mode = 'w'; header = False
    for start in range(0, len(df), chunksize):
        chunk = df.iloc[start:start+chunksize]
        chunk.to_csv(filename, sep=sep, index=False, header=header, mode=mode)
        mode = 'a'
        if logger and start == 0:
            logger.info("Writing results to %s", filename)


def _merge_stream_parts(part_paths, out_file, fmt, logger):
    """Merge worker output incrementally, without collecting all pairs in RAM."""
    part_paths = sorted(part_paths)
    if fmt == "txt":
        with open(out_file, "wb") as dst:
            for part_path in part_paths:
                with open(part_path, "rb") as src:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    elif part_paths:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Incremental Parquet output requires pyarrow. Install pyarrow or use fmt='txt'."
            ) from exc
        writer = None
        try:
            for part_path in part_paths:
                table = pq.read_table(part_path)
                if writer is None:
                    writer = pq.ParquetWriter(out_file, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    else:
        empty = pd.DataFrame({
            "locus1": pd.Series(dtype="object"),
            "locus2": pd.Series(dtype="object"),
            "v": pd.Series(dtype="float32"),
        })
        if fmt == "txt":
            empty.to_csv(out_file, sep="\t", index=False, header=False)
        else:
            empty.to_parquet(out_file, index=False)
    logger.info("Results saved to %s", out_file)


def init_worker(data_fp32, data_weighted, weight_fp32, index_arr, t2, threshold,
                block_size, stream_dir=None, stream_fmt=None,
                stream_batch_rows=1_000_000):
    global DATA_FP32, DATA_WEIGHTED, WEIGHT_FP32, INDEX_ARR, T2, THRESHOLD, BLOCK_SIZE
    global STREAM_DIR, STREAM_FMT, STREAM_BATCH_ROWS
    DATA_FP32      = data_fp32
    DATA_WEIGHTED  = data_weighted
    WEIGHT_FP32    = weight_fp32
    INDEX_ARR      = index_arr
    T2             = t2
    THRESHOLD      = threshold
    BLOCK_SIZE     = block_size
    STREAM_DIR     = stream_dir
    STREAM_FMT     = stream_fmt
    STREAM_BATCH_ROWS = int(stream_batch_rows)


def _result_frame(rows):
    """Build one output batch without forcing locus labels to integers."""
    return pd.DataFrame(rows, columns=["locus1", "locus2", "v"])


def _result_frame_arrays(global_i, global_j, values):
    """Build a result batch without creating one Python tuple per pair."""
    return pd.DataFrame({
        "locus1": INDEX_ARR[global_i],
        "locus2": INDEX_ARR[global_j],
        "v": np.asarray(values, dtype=np.float32),
    })


def _balanced_outer_blocks(outer, t2, block_size, ngpu):
    """Assign triangular tiles by estimated work, not block count."""
    assignments = [[] for _ in range(ngpu)]
    loads = [0] * ngpu
    weighted = [
        (int(np.ceil((t2 - bi) / block_size)), bi)
        for bi in outer
    ]
    for tile_count, bi in sorted(weighted, reverse=True):
        gpu_id = min(range(ngpu), key=lambda idx: (loads[idx], idx))
        assignments[gpu_id].append(bi)
        loads[gpu_id] += tile_count
    for blocks in assignments:
        blocks.sort()
    return assignments, loads


def _epidis_elementwise_kernel():
    """Fuse the elementwise EpiDis formula into one CUDA kernel."""
    return cp.ElementwiseKernel(
        "float32 joint, float32 marginal, float32 safe_dp, float32 safe_dq, "
        "float32 size_p, float32 size_q, bool valid_ref, float32 prob_eps, "
        "float32 eps",
        "float32 e",
        r'''
        if (!valid_ref) {
            e = -1.0f;
        } else {
            float p1 = fminf(fmaxf(joint / safe_dp, prob_eps), 1.0f - prob_eps);
            float q1 = fminf(fmaxf((marginal - joint) / safe_dq, prob_eps),
                             1.0f - prob_eps);
            float p0 = 1.0f - p1;
            float q0 = 1.0f - q1;
            float st10 = p1 + eps;
            float st11 = p0 + eps;
            float st20 = q1 + eps;
            float st21 = q0 + eps;
            float mix0 = st10 * size_p + st20 * size_q;
            float mix1 = st11 * size_p + st21 * size_q;
            float r10 = fmaxf(st10 / mix0, eps);
            float r11 = fmaxf(st11 / mix1, eps);
            float r20 = fmaxf(st20 / mix0, eps);
            float r21 = fmaxf(st21 / mix1, eps);
            float js = (st10 * log2f(r10) + st11 * log2f(r11)) * size_p
                     + (st20 * log2f(r20) + st21 * log2f(r21)) * size_q;
            e = sqrtf(fmaxf(js, 0.0f));
        }
        ''',
        "epidis_fused_float32",
    )


def gpu_worker(task):
    gpu_id, bi_list = task
    result_frames = []
    buffered_frames = []
    buffered_rows = 0
    part_paths = []
    part_number = 0
    # Bind this process to one GPU
    with cp.cuda.Device(gpu_id):
        data_gpu     = cp.asarray(DATA_FP32)
        weighted_gpu = cp.asarray(DATA_WEIGHTED)
        weight_gpu   = cp.asarray(WEIGHT_FP32)
        # This value depends only on locus j, so calculate it once per GPU.
        marginal_gpu = weighted_gpu.sum(axis=1)
        epidis_kernel = _epidis_elementwise_kernel()
        # Keep the divergence stabilizer consistent with the CPU definition.
        eps = cp.float32(1e-27)
        # float32 cannot represent 1-1e-9; this is its practical safe boundary.
        prob_eps = cp.float32(1e-7)

        for bi in bi_list:
            end_i = min(bi + BLOCK_SIZE, T2)
            block_i = data_gpu[bi:end_i]
            inv_i   = 1 - block_i

            dp_vec = block_i.dot(weight_gpu)
            dq_vec = inv_i.dot(weight_gpu)
            valid_ref = (dp_vec > 0) & (dq_vec > 0)
            safe_dp = cp.where(valid_ref, dp_vec, cp.float32(1.0))
            safe_dq = cp.where(valid_ref, dq_vec, cp.float32(1.0))
            size_p = safe_dp / (safe_dp + safe_dq)
            size_q = 1 - size_p

            for bj in range(bi, T2, BLOCK_SIZE):
                end_j  = min(bj + BLOCK_SIZE, T2)
                block_j= weighted_gpu[bj:end_j]

                data_p = block_j.dot(block_i.T)
                marginal_j = marginal_gpu[bj:end_j]
                e_mat = epidis_kernel(
                    data_p,
                    marginal_j[:, None],
                    safe_dp[None, :],
                    safe_dq[None, :],
                    size_p[None, :],
                    size_q[None, :],
                    valid_ref[None, :],
                    prob_eps,
                    eps,
                )
                idx_j, idx_i = cp.where(e_mat >= THRESHOLD)

                if bj == bi:
                    mask = idx_j > idx_i
                    idx_i, idx_j = idx_i[mask], idx_j[mask]

                if idx_i.size:
                    gi = cp.asnumpy(idx_i + bi).astype(np.intp, copy=False)
                    gj = cp.asnumpy(idx_j + bj).astype(np.intp, copy=False)
                    vals = cp.asnumpy(e_mat[idx_j, idx_i])
                    frame = _result_frame_arrays(gi, gj, vals)
                    if STREAM_DIR is None:
                        result_frames.append(frame)
                    else:
                        buffered_frames.append(frame)
                        buffered_rows += len(frame)
                        if buffered_rows >= STREAM_BATCH_ROWS:
                            suffix = "txt" if STREAM_FMT == "txt" else "parquet"
                            part_path = os.path.join(
                                STREAM_DIR,
                                f"gpu{gpu_id}_part{part_number:08d}.{suffix}",
                            )
                            combined = pd.concat(buffered_frames, ignore_index=True)
                            if STREAM_FMT == "txt":
                                combined.to_csv(
                                    part_path, sep="\t", index=False, header=False
                                )
                            else:
                                combined.to_parquet(part_path, index=False)
                            part_paths.append(part_path)
                            part_number += 1
                            buffered_frames.clear()
                            buffered_rows = 0
        if STREAM_DIR is not None and buffered_frames:
            suffix = "txt" if STREAM_FMT == "txt" else "parquet"
            part_path = os.path.join(
                STREAM_DIR, f"gpu{gpu_id}_part{part_number:08d}.{suffix}"
            )
            combined = pd.concat(buffered_frames, ignore_index=True)
            if STREAM_FMT == "txt":
                combined.to_csv(part_path, sep="\t", index=False, header=False)
            else:
                combined.to_parquet(part_path, index=False)
            part_paths.append(part_path)
    if STREAM_DIR is not None:
        return part_paths
    if result_frames:
        return pd.concat(result_frames, ignore_index=True)
    return pd.DataFrame(columns=["locus1", "locus2", "v"])


# =============================
# NEW: DataFrame-based function
# =============================
def run_epidis_gpu_df(
    *,
    snp_df: pd.DataFrame,
    weight_df: pd.Series | pd.DataFrame,
    outlier: float,
    prefix: str | None = None,
    t1: int = 0,
    t2: int = -1,
    fmt: str = "txt",
    block_size: int = 5000,
    convert_chunksize: int = 20000,
    convert_n_jobs: int = -1,
    maf: float | None = None,
    logger: logging.Logger | None = None,
    return_df: bool = True,
    verbose: bool = True,
    stream_batch_rows: int = 1_000_000,
):
    """
    与脚本主体同逻辑，但输入改为 DataFrame。
    - snp_df: 行=位点、列=样本；若为非0/1类型，会自动做“众数→1 其他→0”的行级转换（并行）。
    - weight_df: 支持 Series，或包含 'wet' 列/单列的 DataFrame；按样本名与 snp_df.columns 对齐。
    - return_df=True 返回 DataFrame；否则按 fmt+prefix 落盘并返回路径。
    """
    lg = logger or init_logger(
        logging.INFO if verbose else logging.WARNING
    )

    if not isinstance(snp_df, pd.DataFrame) or snp_df.empty:
        raise ValueError("snp_df must be a non-empty pandas DataFrame")
    if fmt not in {"txt", "parquet"}:
        raise ValueError("fmt must be 'txt' or 'parquet'")
    if block_size < 1 or convert_chunksize < 1 or stream_batch_rows < 1:
        raise ValueError(
            "block_size, convert_chunksize and stream_batch_rows must be positive"
        )
    if not np.isfinite(float(outlier)) or float(outlier) < 0:
        raise ValueError("outlier must be a finite non-negative number")
    if maf is not None and not (0.0 <= float(maf) <= 0.5):
        raise ValueError("maf must be between 0 and 0.5")
    if not return_df and prefix is None:
        raise ValueError("When return_df=False, 'prefix' must be provided to save results.")

    # GPU check
    ngpu = cp.cuda.runtime.getDeviceCount()
    if ngpu < 1:
        lg.error("No GPU detected.")
        raise RuntimeError("No GPU detected")
    lg.info("Detected %d GPU(s)", ngpu)

    # 与 load_data 中一致的二值化流程（最小改动：按位置，不做对齐）
    def _is_binary_row(row: pd.Series) -> bool:
        if not np.issubdtype(row.dtype, np.number):
            return False
        vals = np.unique(row.values)
        return set(vals.tolist()).issubset({0, 1})

    def _transform_row(row: pd.Series) -> pd.Series:
        vc = row.value_counts()
        major = vc.idxmax()
        return (row == major).astype(np.float32)

    def _process_chunk(df_chunk: pd.DataFrame):
        df_bin = df_chunk.apply(_transform_row, axis=1)
        return df_bin.values, df_bin.index.to_list()

    # SNP 二值化（与 load_data 同步）
    first_row = snp_df.iloc[0]
    if _is_binary_row(first_row):
        lg.info("Detected numeric binary SNP DataFrame → skip conversion")
        data_bin  = snp_df.values.astype(np.float32, copy=False)
        index_arr = snp_df.index.to_numpy(dtype=object)
    else:
        lg.info("Detected nucleotide SNP DataFrame → convert in parallel")
        n = len(snp_df)
        if n <= convert_chunksize:
            blk, idx = _process_chunk(snp_df)
            data_bin  = np.asarray(blk, dtype=np.float32)
            index_arr = np.array(idx, dtype=object)
        else:
            bounds = range(0, n, convert_chunksize)
            chunks = [snp_df.iloc[i:i+convert_chunksize] for i in bounds]
            results = Parallel(n_jobs=convert_n_jobs, backend="loky")(
                delayed(_process_chunk)(chunk) for chunk in chunks
            )
            blocks, idx_lists = zip(*results)
            data_bin  = np.vstack(blocks)
            index_arr = np.concatenate([np.array(ix, dtype=object) for ix in idx_lists])

    lg.info("Final binary SNP matrix shape: %d loci × %d samples", *data_bin.shape)

    # —— 行过滤（与文件入口一致，仅当 maf 提供时） —— #
    if maf is not None:
        n_rows, n_cols = data_bin.shape
        not_nan = ~np.isnan(data_bin)
        n_eff = not_nan.sum(axis=1)
        sum1  = np.nansum(data_bin, axis=1)
        p1 = np.divide(sum1, n_eff, out=np.zeros_like(sum1, dtype=np.float32), where=(n_eff > 0))

        thr = 1.0 - float(maf)
        # Binary coding makes allele 1 the major allele, hence MAF=1-p1.
        # Keep the conventional inclusive boundary MAF >= requested maf.
        keep = (n_eff > 0) & (p1 <= thr)

        kept = int(keep.sum())
        lg.info("Applied row filter: keep MAF >= %.4f (p1 <= %.4f) → kept %d/%d rows",
                float(maf), thr, kept, int(n_rows))

        data_bin  = data_bin[keep, :]
        index_arr = index_arr[keep]
    # —— 行过滤结束 —— #

    # 权重：支持 Series/单列 DataFrame，并严格按样本名对齐。
    if isinstance(weight_df, pd.DataFrame):
        if "wet" in weight_df.columns:
            weight_ser = weight_df["wet"]
        elif weight_df.shape[1] == 1:
            weight_ser = weight_df.iloc[:, 0]
        else:
            raise ValueError("weight_df must contain 'wet' or have exactly one column")
    elif isinstance(weight_df, pd.Series):
        weight_ser = weight_df
    else:
        raise TypeError("weight_df must be a pandas Series or DataFrame")
    weight_ser = weight_ser.reindex(snp_df.columns)
    if weight_ser.isna().any():
        missing = weight_ser.index[weight_ser.isna()].tolist()[:5]
        raise ValueError(f"Weights are missing for SNP samples; examples: {missing}")
    weight_arr = weight_ser.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(weight_arr).all():
        raise ValueError("Weights must contain only finite values")
    if np.any(weight_arr < 0) or float(weight_arr.sum(dtype=np.float64)) <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum")
    lg.info("Weight vector length: %d", weight_arr.size)

    # 准备矩阵
    data_fp32     = data_bin.astype(np.float32, copy=False)
    weight_fp32   = weight_arr.astype(np.float32, copy=False)
    data_weighted = (data_fp32 * weight_fp32[None, :]).astype(np.float32, copy=False)

    # 索引范围
    N  = data_fp32.shape[0]
    t1 = max(int(t1), 0)
    t2 = int(N if t2 <= 0 else min(t2, N))
    B  = int(block_size)
    if t1 >= t2:
        raise ValueError(f"Invalid index range [{t1}:{t2})")
    thr = float(outlier)
    lg.info("Processing indices [%d:%d) block_size=%d threshold=%.8g", t1, t2, B, thr)

    # 任务拆分
    outer = list(range(t1, t2, B))
    per_gpu, gpu_loads = _balanced_outer_blocks(outer, t2, B, ngpu)
    tasks = [(i, per_gpu[i]) for i in range(ngpu)]
    lg.info("Estimated triangular tile loads per GPU: %s", gpu_loads)

    stream_dir = None
    out_file = None
    if not return_df:
        out_file = (
            f"{prefix}_Epi_pairs_w_{t1}_{t2}.{fmt}"
            if t2 < N else f"{prefix}_Epi_pairs_w.{fmt}"
        )
        stream_dir = tempfile.mkdtemp(prefix="epidis_gpu_parts_")

    # 多进程（spawn）
    ctx = get_context('spawn')
    start_time = time.time()
    try:
        pool = ctx.Pool(
            processes=ngpu,
            initializer=init_worker,
            initargs=(data_fp32, data_weighted, weight_fp32, index_arr, t2, thr, B,
                      stream_dir, fmt if stream_dir else None, stream_batch_rows)
        )
        try:
            all_parts = pool.map(gpu_worker, tasks)
        finally:
            pool.close()
            pool.join()
    except Exception:
        if stream_dir is not None:
            shutil.rmtree(stream_dir, ignore_errors=True)
        raise

    runtime_sec = time.time() - start_time
    lg.info(f"EpiDis GPU Processing completed! Runtime: {runtime_sec:.3f} seconds ({runtime_sec*1000:.0f} ms)")
    lg.info("GPU processing completed!")

    if return_df:
        frames = [frame for frame in all_parts if not frame.empty]
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=["locus1", "locus2", "v"])

    part_paths = [path for paths in all_parts for path in paths]
    try:
        _merge_stream_parts(part_paths, out_file, fmt, lg)
    finally:
        shutil.rmtree(stream_dir, ignore_errors=True)

    lg.info("Done.")
    return out_file


def main():
    args   = parse_args()
    logger = init_logger()

    # Check GPUs
    ngpu = cp.cuda.runtime.getDeviceCount()
    if ngpu < 1:
        logger.error("No GPU detected. Exiting.")
        sys.exit(1)
    logger.info("Detected %d GPU(s)", ngpu)

    # Load & prepare (保持原行为：从文件读取)
    data_cpu, weight_arr, index_arr = load_data(args.snp, args.w, args.convert_chunksize, logger, maf=args.maf)
    data_fp32     = data_cpu.astype(np.float32)
    weight_fp32   = weight_arr.astype(np.float32)
    data_weighted = (data_fp32 * weight_fp32[None, :]).astype(np.float32)
    del data_cpu, weight_arr

    # Index ranges
    N  = data_fp32.shape[0]
    t1 = max(args.t1, 0)
    t2 = args.t2 if args.t2 > 0 else N
    t2 = min(t2, N)
    B  = args.block_size
    thr = float(args.outlier)
    logger.info("Processing indices [%d:%d) block_size=%d threshold=%.8g",
                t1, t2, B, thr)

    # Split blocks
    outer = list(range(t1, t2, B))
    per_gpu, gpu_loads = _balanced_outer_blocks(outer, t2, B, ngpu)
    tasks = [(i, per_gpu[i]) for i in range(ngpu)]
    logger.info("Estimated triangular tile loads per GPU: %s", gpu_loads)

    # Spawn pool
    ctx = get_context('spawn')   # <-- use spawn instead of fork
    start_time = time.time()
    pool = ctx.Pool(
        processes=ngpu,
        initializer=init_worker,
        initargs=(data_fp32, data_weighted, weight_fp32, index_arr, t2, thr, B)
    )
    # Execute
    all_parts = pool.map(gpu_worker, tasks)
    pool.close(); pool.join()
    end_time = time.time()
    runtime_sec = end_time - start_time
    runtime_ms = runtime_sec * 1000
    logger.info(f"EpiDis GPU Processing completed! Runtime: {runtime_sec:.3f} seconds ({runtime_ms:.0f} ms)")
    logger.info("GPU processing completed!")
    # Collect & write
    frames = [frame for frame in all_parts if not frame.empty]
    if frames:
        df_out = pd.concat(frames, ignore_index=True)
    else:
        df_out = pd.DataFrame(columns=['locus1', 'locus2', 'v'])

    if args.format == 'txt':
        out_file = (
            f"{args.prefix}_Epi_pairs_w_{t1}_{t2}.txt"
            if args.t2 > 0
            else f"{args.prefix}_Epi_pairs_w.txt"
        )
        write_in_chunks(df_out, out_file, logger=logger)
    else:
        out_file = (
            f"{args.prefix}_Epi_pairs_w_{t1}_{t2}.parquet"
            if args.t2 > 0
            else f"{args.prefix}_Epi_pairs_w.parquet"
        )
        df_out.to_parquet(out_file, index=False)
        logger.info("Results saved to %s", out_file)

    logger.info("Done.")


if __name__ == '__main__':
    main()
