<p align="center">
  <img src="docs/assets/epidive-logo.png" alt="EpiDive logo" width="240">
</p>

# EpiDive

EpiDive 是面向 SNP 矩阵的 VP 与高阶上位性分析流程，提供 CPU 和 GPU 两套实现，适用于单个 background、批量 background 以及多服务器分片计算。

> **当前组件版本：** CPU `1.0.5`；GPU `1.0.14`。两套组件独立维护版本号。

[English](README.md) · [安装](docs/INSTALLATION.md) · [使用](docs/USAGE.md) · [输入输出](docs/INPUT_OUTPUT.md) · [HPC/Slurm](docs/HPC.md)

## 主要能力

- CPU 支持 Intel MKL 与 OpenBLAS。
- GPU 支持 CuPy、CUDA 11/12，并在部分阶段提供经验证的 CPU 回退路径。
- 同时支持 MAF 和最小次要等位基因计数过滤。
- 支持 SCC、HCC、NW 三种样本重加权模式。
- 使用流式 Parquet 输出和紧凑的 `uint8` 等位基因缓存降低内存压力。
- 批量 background、确定性多服务器分片、断点续算和全数据 VP 缓存。
- 提供成功标记、错误日志、分片清单、混合 ESS 筛选和精简最终输出。

## 快速安装

CPU（MKL）：

```bash
cd cpu
conda env create -f environment-cpu-mkl.yml
conda activate epidive-cpu
python -m pip install --no-deps .
python tests/smoke_test.py
epidive-vp-ho-cpu --help
```

GPU（CUDA 12）：

```bash
cd gpu
bash install.sh --cuda 12 --blas mkl
conda activate epidive-gpu
epidive-vp-ho-gpu --help
```

## 单个 background 示例

```bash
epidive-vp-ho-cpu \
  --input /data/snp_matrix.parquet \
  --output-dir /data/results/background_4938638 \
  --background 4938638 \
  --pair-jobs 32 \
  --prefix vp10k
```

输入矩阵必须为“位点 × 样本”；第一列为位点编号，其余列为样本。支持 Parquet 或制表符分隔文本。background 位点必须存在于矩阵行索引中。

## 生产运行建议

- 先用 `--background-limit 2` 做小规模验证。
- 生产续算保持默认设置，不要使用 `--no-resume`。
- 多服务器使用相同 background 列表和 `--shard-count`，每台服务器设置不同的 `--shard-index`。
- 每台服务器使用独立输出目录，全部完成后再汇总。
- BLAS 线程数应与调度器申请的 CPU 数一致。

完整说明见 [安装文档](docs/INSTALLATION.md)、[使用文档](docs/USAGE.md)、[输入输出约定](docs/INPUT_OUTPUT.md) 和 [HPC 指南](docs/HPC.md)。

## 引用与许可证

EpiDive 使用 [MIT 许可证](LICENSE)发布。正式引用信息尚未提供；补充前，请在论文中引用仓库地址、Release 标签以及实际使用的 CPU/GPU 组件版本。
