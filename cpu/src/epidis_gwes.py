#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Complex Release Version] R_EpiDis_gwes_fun.py

Version: 1.2.0
Author: rivers_imac
Created on Wed May  1 20:45:37 2024

Usage (CLI):
    python3 R_EpiDis_gwes_fun.py --epidis <input_file> --LD <LD_dis> --k <outlier_k> --circle <circle_dna> --prefix <output_prefix>

Import (API):
    from R_EpiDis_gwes_fun import run_gwes_from_df, run_gwes_from_file
"""

import sys
import argparse
import gc
import os
import logging
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
pd.options.mode.chained_assignment = None

# ---------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------
def _init_logger() -> logging.Logger:
    logger = logging.getLogger("R_EpiDis_gwes_fun")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

_logger = _init_logger()

# ---------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="GWES analysis with EpiDis functionality."
    )
    parser.add_argument("--epidis", required=True,
                        help="Input file for loci pairs (tab-delimited). Columns: locus1,locus2,v")
    parser.add_argument("--LD", type=int, required=True,
                        help="Physical LD threshold in bp (≤0 defaults to 1).")
    parser.add_argument("--k", type=float, required=True,
                        help="Outlier parameter: if >=1 use IQR rule (k=1/2/3), else use fixed threshold.")
    parser.add_argument("--circle", type=int, required=True,
                        help="Circular DNA length (≤0: non-circular).")
    parser.add_argument("--prefix", required=True,
                        help="Output file prefix.")
    return parser.parse_args()

# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
def _load_data(path: str) -> pd.DataFrame:
    """
    与原脚本一致：若存在同名 parquet 缓存则优先读，否则按 TSV 读入再缓存为 parquet。
    列名固定为 ['locus1','locus2','v']，dtype 与原文相同。
    """
    cache = path + ".parquet"
    if os.path.exists(cache):
        _logger.info("Loading data from cache %s", cache)
        df = pd.read_parquet(cache)
    else:
        _logger.info("Reading CSV %s", path)
        df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['locus1', 'locus2', 'v'],
            dtype={'locus1': np.int32, 'locus2': np.int32, 'v': np.float32},
            engine='c',
            memory_map=True
        )
        #_logger.info("Caching to Parquet %s", cache)
        #df.to_parquet(cache, index=False)
    _logger.info("Loaded %d pairs", len(df))
    return df

# ---------------------------------------------------------------------
# core helpers（保持与原脚本一致的计算）
# ---------------------------------------------------------------------
def _compute_distances(df: pd.DataFrame, circle_dna: int) -> np.ndarray:
    """向量化基因组距离；若 circle_dna>0 则按环状基因组最短距离修正。"""
    d = np.abs(df['locus1'].values - df['locus2'].values)
    if circle_dna > 0:
        half = circle_dna / 2
        mask = d > half
        d[mask] = circle_dna - d[mask]
        _logger.info("Applied circular adjustment for length %d", circle_dna)
    return d.astype(int)

def _detect_outliers(vals: np.ndarray) -> List[float]:
    """IQR 上界阈值：返回 [k=1, k=2, k=3] 三个阈值（与原脚本一致）。"""
    x = np.asarray(vals, dtype=float)
    if x.size == 0:
        return [0.0, 0.0, 0.0]
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    return [q3 + mul * 1.5 * iqr for mul in (1, 2, 3)]

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
    _logger.info("Calculated outlier threshold (k=%d): %.2f", k, outlier_k)
    return outlier_k

def find_outliers(df_sp1: pd.DataFrame):
    df_sp = df_sp1.copy()
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
        return float(peak + float(margin))
    return float(find_outliers_k(df_sp1, 1))
# --------------------------------- 异常值检测 ---------------------------------

def _plot_GWES_PU(df_sp2: pd.DataFrame, LD_dis: int, outliers: List[float], save_prefix: str, flag_chr: str = 'chr1', outlier_mode: str = 'iqr'
                  ) -> Tuple[float, float, float]:
    """与原脚本的绘图函数一致，仅返回三条阈值以便上层使用。"""
    plot_l = 28 if flag_chr == 'chr1' else 16

    df_sp = df_sp2[df_sp2['geno_dis'] > 0]
    df_sp_PLD = df_sp[df_sp['geno_dis'] <= LD_dis]
    df_sp_LD  = df_sp[df_sp['geno_dis'] > LD_dis]

    # 容错：如果只给了一个阈值，重复三次用于绘图
    if len(outliers) > 1:
        outlier, ext_outlier, ext_plus_outlier = outliers[:3]
    else:
        outlier = ext_outlier = ext_plus_outlier = outliers[0]

    df_sp_ex = df_sp_LD[df_sp_LD['v'] > outlier]

    sns.set_style("white")
    plt.figure(figsize=(plot_l, 7))
    plt.plot(df_sp['geno_dis'], df_sp['v'], '.', alpha=0.5, markersize=0.7, color='linen', label='Common pairs')
    plt.plot(df_sp_PLD['geno_dis'], df_sp_PLD['v'], '.', alpha=0.5, markersize=0.7, color='turquoise', label='PLD pairs')
    plt.plot(df_sp_ex['geno_dis'], df_sp_ex['v'], '.', alpha=0.5, markersize=0.7, color='deepskyblue', label='Strong pairs')

    if outlier_mode == 'kde':
        plt.axhline(y=outlier, c="r", ls="--")
    else:
        if outlier<1:
            plt.axhline(y=outlier, c="r", ls="--")
            if len(outliers) > 1:
                if ext_outlier <1:
                    plt.axhline(y=ext_outlier, c="r", ls="--")
                    if ext_plus_outlier <1:
                        plt.axhline(y=ext_plus_outlier, c="r", ls="--")

    plt.text(LD_dis, plt.ylim()[1], f'P_ld: {LD_dis}', va='bottom', ha='left', color='red')
    plt.axvline(x=LD_dis, c="r", ls="--")

    txt_x = plt.xlim()[1]
    if outlier_mode == 'kde':
        plt.text(txt_x, outlier, f'KDE threshold: {np.around(outlier, 2)}', va='bottom', ha='right', color='red')
    else:
        if len(outliers) > 1:
            if outlier<1:
                plt.text(txt_x, outlier,         f'Outlier(k=1): {outlier:.2f}',         va='bottom', ha='right', color='red')
                if ext_outlier <1:
                    plt.text(txt_x, ext_outlier,     f'Outlier(k=2): {ext_outlier:.2f}',     va='bottom', ha='right', color='red')
                    if ext_plus_outlier <1:
                        plt.text(txt_x, ext_plus_outlier,f'Outlier(k=3): {ext_plus_outlier:.2f}',va='bottom', ha='right', color='red')
        else:
            plt.text(txt_x, outlier, f'Outlier: {outlier:.2f}', va='bottom', ha='right', color='red')

    plt.xlabel('Genome Distance', fontsize=16)
    plt.ylabel('EpiDis Value', fontsize=16)
    plt.title(f'EpiDis_GWES {flag_chr}', fontsize=16)
    sns.despine()
    leg = plt.legend(loc='upper right', fontsize=12, markerscale=17)
    leg.get_frame().set_alpha(1)
    out_file = save_prefix + ".GWES_Epi_LD_gene.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    _logger.info("Saved GWES plot to %s", out_file)
    return outlier, ext_outlier, ext_plus_outlier

# ---------------------------------------------------------------------
# Public APIs（可 import 使用）
# ---------------------------------------------------------------------
def run_gwes_from_df(
    df_pairs: pd.DataFrame,
    *,
    LD: int,
    k: float = 0,
    circle: int,
    prefix: str,
    draw_plot: bool = True,
    logger: Optional[logging.Logger] = None,
    flag_chr: str = 'chr1',
    outlier_mode: str = 'iqr',          # 'iqr' | 'kde'
    kde_margin: float = 0.1,            # KDE: 阈值 = peak + margin
    kde_bw: float | str = 'scott',  # KDE 带宽：'scott'/'silverman'/float
) -> Dict[str, object]:
    """
    直接在 DataFrame 上运行（最小改动复刻原 main() 流程）。

    参数
    ----
    df_pairs : DataFrame，列为 ['locus1','locus2','v']（与原脚本一致）
    LD      : 物理 LD 阈值（bp；≤0 将视为 1）
    k       : 若 k >= 1，使用 IQR(outlier k=1/2/3)；若 k < 1，按固定阈值使用 k
    circle  : 环状基因组长度；≤0 表示线性基因组
    prefix  : 输出前缀（会写出 <prefix>.pairs.LD.outlier_<int(k)>_Uni.csv 与 GWES 图）
    draw_plot : 是否画图（默认 True）
    flag_chr : 染色体标记，默认 'chr1'
    """
    lg = logger or _logger

    # 1) 复制并规范 dtype
    df = df_pairs.copy()
    df['locus1'] = df['locus1'].astype(np.int32)
    df['locus2'] = df['locus2'].astype(np.int32)
    df['v']      = df['v'].astype(np.float32)

    # 2) 距离 + LD 标记
    LD_dis = LD if LD > 0 else 1
    df['geno_dis'] = _compute_distances(df, circle)
    df['LD'] = np.where(df['geno_dis'] <= LD_dis, -1, 0)

    # 3) 仅在 LD 之外的对上做阈值
    df_ld = df[df['geno_dis'] > LD_dis]
    lg.info("Pairs beyond LD>%d: %d", LD_dis, len(df_ld))

    if outlier_mode == 'kde':
        thresh = find_outliers_kde_upper(df_ld, margin=kde_margin, bw=kde_bw)
        outs = [thresh, thresh, thresh]
    else:
        if k >= 1:
            # IQR 阈值（与原脚本一致：输出三个阈值）
            outs = _detect_outliers(df_ld['v'].values)
            for i, val in enumerate(outs, start=1):
                lg.info("Outlier(k=%d): %.2f", i, val)
            k_idx = max(1, min(3, int(k)))
            thresh = outs[k_idx - 1]
        else:
            # 固定阈值
            if k < -1:
                k = 0
            outs = [k]
            thresh = k
            lg.info("Fixed outlier (k<1): %.2f", k)

    # 4) 过滤并保存
    lg.info("Filtering pairs with v >= %.2f", thresh)
    df_final = df_ld[df_ld['v'] >= float(thresh)].copy()

    if outlier_mode == 'kde':
        thr_tag = f"{float(thresh):.2f}".replace('.', '_')
        out_csv = f"{prefix}.pairs.LD.outlier_KDE_thr{thr_tag}_Uni.csv"
    else:
        out_csv = f"{prefix}.pairs.LD.outlier_{int(k) if k>=1 else float(thresh)}_Uni.csv"
    df_final.to_csv(out_csv, index=False)
    lg.info("Saved filtered pairs to %s", out_csv)

    # 5) 绘图（与原脚本一致风格）
    if draw_plot:
        _plot_GWES_PU(df, LD_dis, outs, prefix, flag_chr=flag_chr, outlier_mode=outlier_mode)

    gc.collect()

    return {
        "df_final": df_final,
        "thresholds": {"k1": (outs[0] if len(outs) > 0 else None),
                       "k2": (outs[1] if len(outs) > 1 else None),
                       "k3": (outs[2] if len(outs) > 2 else None)},
        "df_all": df
    }

def run_gwes_from_file(
    epidis_path: str,
    *,
    LD: int,
    k: float,
    circle: int,
    prefix: str,
    draw_plot: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """
    文件入口：完全保留你原来的读取与缓存逻辑，再调用 run_gwes_from_df。
    """
    lg = logger or _logger
    df = _load_data(epidis_path)
    return run_gwes_from_df(
        df,
        LD=LD,
        k=k,
        circle=circle,
        prefix=prefix,
        draw_plot=draw_plot,
        logger=lg,
    )

# ---------------------------------------------------------------------
# CLI（保持原用法）
# ---------------------------------------------------------------------
def main():
    args = _parse_args()

    res = run_gwes_from_file(
        args.epidis,
        LD=args.LD,
        k=args.k,
        circle=args.circle,
        prefix=args.prefix,
        draw_plot=True,
        logger=_logger,
    )
    _logger.info("Processing complete. Filtered pairs: %d", len(res["df_final"]))

if __name__ == "__main__":
    main()