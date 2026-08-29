# sample_reweight_func.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Literal, Dict, Any, List
from contextlib import nullcontext

import os
import gc
import logging
import time
import numpy as np
import pandas as pd
from joblib import cpu_count
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

ModelT = Literal["HCC", "SCC", "NW"]
ReweightBackendT = Literal["auto", "cpu", "gpu"]

# ---------------------------
# logger
# ---------------------------
def get_logger(name: str = "R_sample_reweight", verbose: bool = True) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.DEBUG if verbose else logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h)
    return lg

# ---------------------------
# 仅 NW 模式需要：列名获取
# ---------------------------
def _columns_from_tsv_header(path: str) -> List[str]:
    df0 = pd.read_table(path, sep="\t", index_col=0, nrows=0)
    return df0.columns.tolist()

def _columns_from_parquet_metadata(path: str, logger: Optional[logging.Logger] = None) -> Optional[List[str]]:
    lg = logger or get_logger()
    try:
        import pyarrow.parquet as pq
    except Exception:
        lg.info("pyarrow 不可用，NW 模式 Parquet 将回退到 pandas.read_parquet")
        return None
    try:
        pf = pq.ParquetFile(path)
        return list(pf.schema.names)
    except Exception as e:
        lg.warning("读取 Parquet 元数据失败：%s", e)
        return None

# ---------------------------
# SNP 读取函数
# ---------------------------
def load_snp_matrix(path: str, nw_mode: bool = False, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    从文件读取 SNP 矩阵。
    - 普通模式：返回完整 DataFrame（行=位点，列=样本）
    - NW 模式：只读列名（不加载矩阵数据），返回空 DataFrame 但含列名
    同时：若输入是 .tsv/.txt，自动保存为同名 .parquet 文件。
    """
    lg = logger or get_logger()
    ext = os.path.splitext(path)[1].lower()

    if nw_mode:
        lg.info("NW 模式：仅取列名，不加载矩阵数据。")
        if ext == ".parquet":
            cols = _columns_from_parquet_metadata(path, logger=lg)
            if cols is None:
                df = pd.read_parquet(path)
                cols = df.columns.tolist()
            return pd.DataFrame(index=[], columns=cols)
        else:
            # 只读表头
            df0 = pd.read_table(path, sep="\t", index_col=0, nrows=0)
            cols = df0.columns.tolist()
            # 生成一个空 DataFrame
            df_empty = pd.DataFrame(index=[], columns=cols)
            # 自动保存 parquet（只含列名，无数据）
            parquet_path = os.path.splitext(path)[0] + ".parquet"
            try:
                df_empty.to_parquet(parquet_path)
                lg.info("NW 模式下也已自动生成 parquet: %s", parquet_path)
            except Exception as e:
                lg.warning("NW 模式保存 parquet 失败: %s", e)
            return df_empty

    # 非 NW：加载完整矩阵
    lg.info("Loading SNP matrix from: %s", path)
    if ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_table(path, index_col=0)
        # 自动保存为 parquet
        parquet_path = os.path.splitext(path)[0] + ".parquet"
        try:
            df.to_parquet(parquet_path)
            lg.info("已自动保存为 parquet: %s", parquet_path)
        except Exception as e:
            lg.warning("保存为 parquet 失败: %s", e)
    return df

# ---------------------------
# 工具函数（和之前一致）
# ---------------------------
ALLELE_ENCODING_ATTR = "epidive_allele_uint8"


def encode_snp_alleles_uint8(snp_df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical SNP alleles once while preserving their identity."""
    if not isinstance(snp_df, pd.DataFrame) or snp_df.empty:
        raise ValueError("snp_df must be a non-empty DataFrame")
    if snp_df.attrs.get(ALLELE_ENCODING_ATTR, False):
        return snp_df

    values = snp_df.to_numpy(copy=False)
    if np.issubdtype(values.dtype, np.number):
        return snp_df

    mapping = {ord("A"): 0, ord("T"): 1, ord("C"): 2, ord("G"): 3, ord("-"): 4}
    lookup = np.full(256, 255, dtype=np.uint8)
    for char_code, numeric_code in mapping.items():
        lookup[char_code] = numeric_code
    allele_values = np.asarray(values, dtype="S1")
    byte_values = allele_values.view(np.uint8).reshape(allele_values.shape)
    encoded = lookup[byte_values]
    if np.any(encoded == 255):
        unknown = np.unique(allele_values[encoded == 255]).astype(str).tolist()
        raise ValueError(f"SNP matrix contains unsupported alleles: {unknown[:10]}")

    result = pd.DataFrame(
        np.ascontiguousarray(encoded, dtype=np.uint8),
        index=snp_df.index,
        columns=snp_df.columns,
    )
    result.attrs[ALLELE_ENCODING_ATTR] = True
    result.attrs["epidive_allele_mapping"] = {
        "A": 0,
        "T": 1,
        "C": 2,
        "G": 3,
        "-": 4,
    }
    return result


def _convert_to_numeric(snp_df: pd.DataFrame) -> np.ndarray:
    """Fast single-character allele encoding using a byte lookup table."""
    if snp_df.attrs.get(ALLELE_ENCODING_ATTR, False):
        encoded = snp_df.to_numpy(dtype=np.uint8, copy=False)
        return np.ascontiguousarray(encoded, dtype=np.uint8)

    numeric_values = snp_df.to_numpy(copy=False)
    if np.issubdtype(numeric_values.dtype, np.number):
        if not np.isfinite(numeric_values).all():
            raise ValueError("Numeric SNP matrix contains non-finite values")
        return np.ascontiguousarray(numeric_values, dtype=np.uint8)

    mapping = {ord('A'): 0, ord('T'): 1, ord('C'): 2, ord('G'): 3, ord('-'): 4}
    lookup = np.full(256, 255, dtype=np.uint8)
    for char_code, numeric_code in mapping.items():
        lookup[char_code] = numeric_code

    values = np.asarray(snp_df.to_numpy(copy=False), dtype="S1")
    byte_values = values.view(np.uint8).reshape(values.shape)
    encoded = lookup[byte_values]
    if np.any(encoded == 255):
        unknown = np.unique(values[encoded == 255]).astype(str).tolist()
        raise ValueError(f"SNP matrix contains unsupported alleles: {unknown[:10]}")
    return np.ascontiguousarray(encoded, dtype=np.uint8)

def _compute_distance_matrix(
    snp_numeric: np.ndarray,
    n_jobs: int = -1,
    block_size: int = 100,
    logger=None,
    show_progress: bool = True,
    backend: ReweightBackendT = "auto",
    gpu_id: int = 0,
    gpu_block_size: Optional[int] = None,
) -> np.ndarray:
    """Compute exact pairwise identical-allele counts with tiled SGEMM.

    For each allele state, ``B.T @ B`` gives the number of loci at which two
    samples carry that same allele. The GPU path keeps only one locus block and
    the sample-by-sample accumulator on the device. ``backend='auto'`` uses a
    working CuPy device when available and otherwise falls back to CPU BLAS.
    """
    lg = logger or get_logger()
    if snp_numeric.ndim != 2:
        raise ValueError("snp_numeric must be a 2D loci-by-samples matrix")
    n_loci, n_samples = snp_numeric.shape
    if n_loci == 0 or n_samples == 0:
        raise ValueError("SNP matrix cannot be empty")
    backend = str(backend).lower()
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'")
    if isinstance(gpu_id, bool) or int(gpu_id) != gpu_id or int(gpu_id) < 0:
        raise ValueError("gpu_id must be a non-negative integer")
    gpu_id = int(gpu_id)
    if gpu_block_size is not None and int(gpu_block_size) < 1:
        raise ValueError("gpu_block_size must be a positive integer")

    if n_jobs <= 0:
        n_jobs = cpu_count()
    n_jobs = max(1, int(n_jobs))
    # The former default (100) represented submitted sample columns. For SGEMM
    # it is too small, so legacy small values select a cache-friendly default.
    locus_block_size = 5_000 if int(block_size) < 1_000 else int(block_size)
    allele_states = np.unique(snp_numeric).tolist()

    def _progress(iterator, size: int, description: str):
        if not show_progress:
            return iterator
        return tqdm(
            iterator,
            total=(n_loci + size - 1) // size,
            desc=description,
            unit="blocks",
            dynamic_ncols=True,
        )

    def _log_completion(backend_name: str, matrix_seconds: float) -> None:
        estimated_gflop = (
            2.0 * n_loci * n_samples * n_samples * len(allele_states) / 1.0e9
        )
        lg.info(
            "Matrix similarity backend=%s complete: runtime=%.3f s, "
            "estimated throughput=%.2f GFLOP/s",
            backend_name,
            matrix_seconds,
            estimated_gflop / max(matrix_seconds, 1.0e-12),
        )

    def _cpu_impl() -> np.ndarray:
        counts = np.zeros((n_samples, n_samples), dtype=np.float32)
        lg.info(
            "Matrix similarity backend=CPU: %d loci x %d samples, "
            "block=%d, BLAS threads=%d",
            n_loci, n_samples, locus_block_size, n_jobs,
        )
        matrix_start = time.perf_counter()
        limiter = (
            threadpool_limits(limits=n_jobs, user_api="blas")
            if threadpool_limits is not None else nullcontext()
        )
        with limiter:
            iterator = _progress(
                range(0, n_loci, locus_block_size),
                locus_block_size,
                "Computing similarity matrix (CPU)",
            )
            for start in iterator:
                stop = min(start + locus_block_size, n_loci)
                locus_block = snp_numeric[start:stop]
                for allele in allele_states:
                    one_hot = np.asarray(
                        locus_block == allele, dtype=np.float32, order="C"
                    )
                    counts += one_hot.T @ one_hot

        _log_completion("CPU", time.perf_counter() - matrix_start)
        return counts.astype(np.float64)

    def _gpu_impl() -> np.ndarray:
        try:
            import cupy as cp
        except Exception as exc:
            raise RuntimeError("CuPy is not importable") from exc

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("CuPy found no CUDA devices")
        if gpu_id >= device_count:
            raise RuntimeError(
                f"gpu_id {gpu_id} is invalid; {device_count} CUDA device(s) visible"
            )

        with cp.cuda.Device(gpu_id):
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            accumulator_bytes = n_samples * n_samples * np.dtype(np.float32).itemsize
            reserve_bytes = max(256 * 1024**2, accumulator_bytes * 4)
            usable_bytes = int(free_bytes) - reserve_bytes
            bytes_per_locus = max(1, n_samples * 8)
            if gpu_block_size is None:
                max_memory_block = usable_bytes // bytes_per_locus
                if max_memory_block < 1:
                    raise MemoryError(
                        "insufficient free GPU memory for one similarity block"
                    )
                device_block_size = min(n_loci, 100_000, max_memory_block)
            else:
                device_block_size = min(n_loci, int(gpu_block_size))

            lg.info(
                "Matrix similarity backend=GPU: %d loci x %d samples, "
                "device=%d, block=%d, free=%.2f/%.2f GiB",
                n_loci,
                n_samples,
                gpu_id,
                device_block_size,
                free_bytes / 1024**3,
                total_bytes / 1024**3,
            )
            matrix_start = time.perf_counter()
            counts_gpu = cp.zeros((n_samples, n_samples), dtype=cp.float32)
            iterator = _progress(
                range(0, n_loci, device_block_size),
                device_block_size,
                "Computing similarity matrix (GPU)",
            )
            for start in iterator:
                stop = min(start + device_block_size, n_loci)
                locus_gpu = cp.asarray(
                    snp_numeric[start:stop], dtype=cp.uint8, order="C"
                )
                for allele in allele_states:
                    one_hot = (locus_gpu == allele).astype(cp.float32)
                    counts_gpu += one_hot.T @ one_hot
                    del one_hot
                del locus_gpu

            cp.cuda.get_current_stream().synchronize()
            counts = cp.asnumpy(counts_gpu).astype(np.float64)
            del counts_gpu
            matrix_seconds = time.perf_counter() - matrix_start

        _log_completion("GPU", matrix_seconds)
        return counts

    if backend == "cpu":
        return _cpu_impl()
    try:
        return _gpu_impl()
    except Exception as exc:
        if backend == "gpu":
            raise RuntimeError(f"GPU similarity computation failed: {exc}") from exc
        lg.warning(
            "GPU similarity path unavailable (%s: %s); falling back to CPU BLAS.",
            type(exc).__name__, exc,
        )
        return _cpu_impl()


def _process_SCC(snp_df: pd.DataFrame, snp_pro: np.ndarray, high_threshold: float, logger) -> tuple[np.ndarray, pd.DataFrame]:
    lg = logger or get_logger()
    def _scc_core(data: np.ndarray, threshold: float) -> np.ndarray:
        # Exclude self-connections only; distinct samples with exact similarity
        # 1.0 should remain connected in the SCC graph.
        adjacency = data > threshold
        np.fill_diagonal(adjacency, False)
        graph = csr_matrix(adjacency)
        _, labels = connected_components(
            csgraph=graph, directed=False, return_labels=True
        )
        return labels
    labels = _scc_core(snp_pro, high_threshold)
    # Keep the historical group_info schema: one descending label index and
    # no data columns (the original groupby used its only column as the key).
    grouped = pd.DataFrame(
        index=pd.Index(np.unique(labels)[::-1], name="labels")
    )
    return labels, grouped

def _process_HCC(snp_df: pd.DataFrame, dist_df: pd.DataFrame, threshold: float, prefix: Optional[str], logger) -> pd.DataFrame:
    Z = linkage(dist_df, method="complete")
    if prefix:  # 保存树图
        plt.figure(figsize=(10, 30))
        dn = dendrogram(Z, orientation="right", labels=dist_df.index, color_threshold=threshold)
        plt.axvline(x=threshold, color="red", linestyle="--")
        plt.savefig(prefix + ".EpiDive_SampleReweighting_HCC_tree.png", dpi=170)
        plt.close()
        leaf_order = [dist_df.index[int(idx)] for idx in dn["leaves"]]
    else:
        dn = dendrogram(Z, no_plot=True)
        leaf_order = [dist_df.index[int(idx)] for idx in dn["leaves"]]
    clusters = fcluster(Z, threshold, criterion="distance")
    return pd.DataFrame(clusters, index=dist_df.index, columns=["cluster"]).loc[leaf_order]

# ---------------------------
# 主计算函数
# ---------------------------
def run_sample_reweight(
    snp_df: pd.DataFrame,
    *,
    model: ModelT = "HCC",
    prefix: Optional[str] = None,
    hcc_threshold: float = 0.75,
    scc_threshold: float = 0.98,
    n_jobs: int = -1,
    block_size: int = 100,
    backend: ReweightBackendT = "auto",
    gpu_id: int = 0,
    gpu_block_size: Optional[int] = None,
    save_tree_when_hcc: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    根据 SNP DataFrame 计算样本权重。
    - model="NW" : 不做加权，所有样本权重=1
    - model="HCC": 层次聚类，使用 hcc_threshold 作为距离阈值
    - model="SCC": 稀疏连通，使用 scc_threshold 作为相似度阈值
    - backend="auto": 优先使用分块 CuPy GPU，不可用时回退 CPU BLAS
    """
    lg = logger or get_logger()
    model_u = model.upper()

    if model_u == "NW":
        snp_wet = pd.DataFrame(index=snp_df.columns, columns=["wet"], data=1.0)
        return {
            "snp_wet": snp_wet,
            "dist_df": None,
            "clusters_or_labels": None,
            "group_info": None,
        }

    lg.info("Converting SNP matrix to numeric ...")
    snp_numeric = _convert_to_numeric(snp_df)
    gc.collect()

    lg.info("Computing SNP distance matrix ...")
    snp_dist = _compute_distance_matrix(
        snp_numeric,
        n_jobs=n_jobs,
        block_size=block_size,
        logger=lg,
        backend=backend,
        gpu_id=gpu_id,
        gpu_block_size=gpu_block_size,
    )
    snp_dist = snp_dist / len(snp_df.index)
    dist_df = pd.DataFrame(snp_dist, index=snp_df.columns, columns=snp_df.columns)

    if model_u == "SCC":
        labels, group_info = _process_SCC(
            snp_df, snp_dist, high_threshold=scc_threshold, logger=lg
        )
        group_sizes = np.bincount(labels)
        snp_wet = pd.DataFrame(
            {"wet": 1.0 / group_sizes[labels]}, index=snp_df.columns
        )
        return {
            "snp_wet": snp_wet,
            "dist_df": dist_df,
            "clusters_or_labels": labels,
            "group_info": group_info,
        }

    elif model_u == "HCC":
        plot_prefix = (prefix if save_tree_when_hcc else None)
        clusters = _process_HCC(
            snp_df, dist_df, hcc_threshold, plot_prefix, logger=lg
        )
        clusters["wet"] = clusters.groupby("cluster")["cluster"].transform("count").rdiv(1.0)
        snp_wet = pd.DataFrame(clusters["wet"].loc[dist_df.index])
        return {
            "snp_wet": snp_wet,
            "dist_df": dist_df,
            "clusters_or_labels": clusters,
            "group_info": None,
        }

    else:
        raise ValueError(f"Unknown model: {model}")
