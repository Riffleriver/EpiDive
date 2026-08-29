# EpiDive VP/High-order GPU server package 1.0.14

Version 1.0.14 adds `--min-mac` as an independent per-locus filter. The
default is 5, and loci must pass both MAF and MAC filters. MAC is included in
cache/resume signatures. Use `--min-mac 0` only for legacy MAF-only behavior.

Version 1.0.13 adds a memory-safe GPU backend for SCC/HCC sample-similarity
calculation while retaining all 1.0.12 preprocessing optimizations:

- Use tiled CuPy SGEMM for the sample-similarity matrix when a CUDA GPU is
  available; keep only one locus block and the accumulator on the device.
- Select a safe GPU block size from current free memory, capped at 100,000 loci.
- Fall back to the validated CPU BLAS implementation when CuPy/CUDA is not
  available; explicit `backend="gpu"` calls fail instead of silently falling back.
- Log the selected CPU/GPU backend, device, block size, runtime, and estimated
  throughput.
- Validate CPU/GPU result equivalence during setup when a GPU is visible.

- Encode categorical alleles once as a compact uint8 matrix.
- Recompute subgroup major alleles with vectorized NumPy operations while
  preserving the legacy first-occurrence tie rule.
- Avoid the Pandas row-wise/joblib conversion path for standard SNP input.
- Install an explicit MKL BLAS backend by default for SCC sample reweighting;
  OpenBLAS remains available through `--blas openblas`.
- Validate the installed BLAS backend and vectorized conversion during setup.
- Report SCC similarity runtime and estimated throughput.

The EpiDis, MAF, SCC, GWES, high-order comparison, pattern restoration, ESS,
resume, and final-only result semantics remain unchanged from 1.0.11.

本包包含 `EpiDive_vp_ho_gpu.py`、其调用的全部本地 Python 模块，以及 GWES 使用的 3 个基因参考文件。代码默认从已安装的 `epidive_data` 包中定位参考文件，不依赖原 Mac 绝对路径。第三方科学计算库和 CuPy 会在服务器安装时从网络下载；用户的 SNP 输入矩阵不包含在包内。

## 1. 服务器要求

- Linux x86_64（推荐）
- NVIDIA GPU，且 `nvidia-smi` 可正常运行
- Conda/Miniforge/Miniconda
- 能访问 conda-forge 和 PyPI
- 默认使用 Python 3.10

先用 `nvidia-smi` 查看驱动支持的 CUDA 版本。CUDA 12 用默认命令；老服务器可选 CUDA 11。

## 2. 一键安装

```bash
tar -xzf EpiDive_vp_ho_gpu_server_package-1.0.14.tar.gz
cd EpiDive_vp_ho_gpu_server_package-1.0.14
bash install.sh --cuda 12 --blas mkl
conda activate epidive-gpu
```

自定义环境名：

```bash
bash install.sh --name my-epidive --cuda 12 --blas mkl
conda activate my-epidive
```

CUDA 11 服务器：

```bash
bash install.sh --cuda 11
```

删除旧环境并全新安装（也适用于上一次安装中途失败）：

```bash
bash install.sh --reinstall --cuda 12
```

若环境已经创建，也可直接安装：

```bash
python -m pip install --no-deps ./dist/epidive_vp_ho_gpu-1.0.14-py3-none-any.whl
python tests/smoke_test.py
EPIDIVE_EXPECTED_BLAS=mkl python tests/blas_backend_test.py
python tests/vectorized_conversion_regression_test.py
python tests/reweight_backend_test.py
```

## 3. 运行分析

安装后不再依赖原脚本中的 Mac 绝对路径：

```bash
epidive-vp-ho-gpu \
  --input /data/10k.NR150S.SNP.GENE.snp \
  --output-dir /data/results/epidive \
  --background 7685000 \
  --prefix vp10k
```

查看所有参数：

```bash
epidive-vp-ho-gpu --help
```

默认分析参数现与 `EpiDive_vp_ho.py` 一致：全数据 `k=1`，背景 1 子组 `k=1`，背景 0 子组 `k=2`，默认不应用 GWES 过滤。需要应用时增加 `--apply-gwes-filter`。

默认使用精简输出，仅显示警告和错误。需要查看完整阶段进度、INFO 日志和汇总时增加 `--verbose`。

命令行主函数成功时返回退出码 0，不会再把包含大型 DataFrame 的结果字典打印到终端；每个高阶子组保存完成后会及时释放返回结果。底层 `run_vp_pipeline` 和 `run_highorder_comparison_pipeline` 仍可供 Python 代码调用并返回字典。

常用参数包括 `--threshold auto`、`--subgroup-threshold auto`、`--min-mac 5`、`--gpu-block-size 5000`、`--selected-pair-gpu 0`、`--no-restore-final-patterns` 和 `--verbose`。

默认生成 VP 和高阶 GWES 图片。若只需要计算结果并希望减少绘图时间和内存占用，运行时增加 `--no-plot`；需要明确开启时使用 `--plot`。

### 批量 background

1.0.11 增加 `--ess-mode absolute|hybrid` 和 `--final-only`。Hybrid ESS 同时使用固定ESS下限及原始样本比例；final-only在每个background完成后删除已知中间文件，仅保留最终筛选结果和状态标记。1.0.10的批处理、缓存、断点续跑和locus整数化逻辑全部保留。

1.0.10 支持从 TSV/CSV 列表依次运行完全相同的全基因组 pairwise 分层分析。SNP 矩阵只读取一次，全样本 VP 只计算一次并写入可复用缓存；每个 background 和 allele 都有完成标记，任务中断后使用同一命令即可续算。该版本还会在高阶合并前把缓存和新结果中的 locus join key 统一为整数，兼容 Parquet 读取出的数字字符串。

```bash
epidive-vp-ho-gpu \
  --input /data/matrix.parquet \
  --output-dir /data/results/background_batch \
  --background-file /data/10k.NR1550S.paired.representative_loci.tsv \
  --background-column representative_snp \
  --apply-gwes-filter \
  --no-plot \
  --prefix vp10k_batch
```

先测试两个 background 可增加 `--background-limit 2`。默认自动续算；只有需要全部重算时才使用 `--no-resume`。

多服务器并行时，所有服务器使用相同列表和 `--shard-count`，但分别设置不同的零起始 `--shard-index`。例如 8 台服务器对应索引 0 到 7：

```bash
epidive-vp-ho-gpu ... --shard-count 8 --shard-index 0
epidive-vp-ho-gpu ... --shard-count 8 --shard-index 1
# ...
epidive-vp-ho-gpu ... --shard-count 8 --shard-index 7
```

批量 background 分析应让每个进程只看到自己申请的 GPU，避免多个进程争抢同一张卡。对“2 节点、4 GPU、每用户最多 80 CPU”的服务器，建议同时运行 4 个任务，每个任务申请 1 GPU 和 20 物理核。不同服务器应使用独立输出目录，计算完成后再汇总。

包内的 `run_background_batch.slurm` 是 20 CPU/1 GPU 的批量模板。确认路径、array 数量和并发上限后提交：

```bash
sbatch run_background_batch.slurm
```

查看队列和日志：

```bash
squeue -j JOB_ID
tail -f epidive_bg_JOB_ID.out
```

1.0.7 对结果等价的热点进行了优化：一次预计算 GPU 边际量、融合 EpiDis 逐元素 kernel、数组化收集 pair、批量写 Parquet、按上三角 tile 负载分配多 GPU，并减少高阶比较中的 Pandas/Polars 往返及分块恢复 pattern。

## 4. 说明

- 输入应为“位点 × 样本”的 SNP 矩阵，格式需能由 `epidis_reweight.load_snp_matrix` 读取。
- wheel 内置 `gene.ids.txt`、参考基因 CSV 和 rep65 CSV；无需在服务器上另外配置这三个路径。
- 安装脚本通过 conda-forge 安装 pandas、NumPy、CuPy 等二进制包，不会调用服务器 GCC 编译 pandas。
- `--background` 指定的位点必须存在于输入矩阵的行索引中。
- 程序会计算全样本 VP，并按背景位点的 1/0 状态分别运行两个高阶子组。
- 多 GPU 代表模式计算会使用所有可见 CUDA GPU；`--selected-pair-gpu` 选择补分阶段使用的单张 GPU。可用 `CUDA_VISIBLE_DEVICES` 限制设备。
- 标准 SNP 输入会先缓存为 uint8，每个子群仍独立重新确定 major allele，不改变统计语义。
- Intel 服务器默认使用 MKL；非 Intel 服务器可通过 `bash install.sh --blas openblas` 安装。
- 大规模 SCC 建议从 `--reweight-jobs 20` 开始测试，不要盲目使用 72 个 BLAS 线程。
- 生产续算不要使用 `--no-resume`；该选项会故意绕过全数据缓存。
- 安装包不捆绑 CUDA 驱动。CuPy wheel 自带所需 CUDA 运行库，但 NVIDIA 驱动必须足够新。
- BLAS 线程数应与 `--reweight-jobs` 及 Slurm `--cpus-per-task` 保持一致。
