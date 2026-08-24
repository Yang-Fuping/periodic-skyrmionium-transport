# Periodic Skyrmionium Transport

[**English**](#english) | [简体中文](#简体中文)

## English

Reproducible NumPy/SciPy calculations for the manuscript
*Zero-Chern Minigap and Locally Compensated Hall Response in Periodic
Skyrmionium Arrays*.

> **Development release.** Version `0.1.0` contains the tested source code and
> reproducibility workflow. The immutable production dataset and its DOI will
> be linked before the archival `v1.0.0` release.

## Scope

The code studies non-interacting spinful electrons on a square lattice with
nearest-neighbour hopping and local exchange coupling to a frozen classical
magnetic texture,

$$
H=-t\sum_{\langle i,j\rangle}c_i^\dagger c_j
-J\sum_i c_i^\dagger(\mathbf m_i\cdot\boldsymbol\sigma)c_i
+\sum_i U_i c_i^\dagger c_i.
$$

It provides:

- normalized uniform, $Q=0$ skyrmionium, and $Q=\pm1$ skyrmion textures;
- lattice solid-angle topological charge and local charge density;
- Bloch-supercell bands, full-zone gaps, DOS, and FHS Chern numbers;
- two-terminal recursive Green-function and full-matrix reference solvers;
- four-terminal NEGF and Landauer--Büttiker voltage probes;
- spin-resolved scattering observables;
- Anderson-disorder statistics and finite-temperature Fermi convolution;
- an independently constructed Kwant backend for selected two- and
  four-terminal cross-validation;
- scripts used for the manuscript's numerical checks and figures.

The production calculations use NumPy/SciPy. A separate Conda environment
retains Kwant 1.5 as an independent validation backend. Across the selected
single-texture and array cases, the largest relative Kwant/NEGF difference is
$7.61\times10^{-6}$ and the full 24-test suite passes.

## Repository layout

```text
skyrmion_transport/   Core numerical library
scripts/              Production, analysis, and plotting scripts
tests/                Nineteen core plus five Kwant regression tests
legacy/               Frozen single-skyrmionium baseline and reference image
data/                 Instructions for the separately archived Zenodo dataset
docs/                  Reproducibility map and release notes
```

Generated arrays are written below `results/` and generated paper figures below
`generated_figures/`; both locations are ignored by Git.

## Installation

Python 3.10--3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or macOS, activate the environment with
`source .venv/bin/activate`.

## Validation

The test suite uses only small systems and normally finishes within seconds:

```powershell
python -m pytest tests/test_core.py -q -p no:cacheprovider
```

The expected result is `19 passed`. The tests cover texture
normalization and charge, analytic folded bands, a zero uniform Chern number,
stable strip complex bands, texture/profile controls, recursive/full-inverse
transmission agreement, lead channel counts,
Landauer--Büttiker gauge invariance and current conservation, $Q\to-Q$ Hall
symmetry, sparse/dense agreement, disorder identity at $W_d=0$, and thermal
convolution.

For the complete Kwant cross-validation suite:

```powershell
conda env create -f environment-kwant.yml
conda run -n kwant-validate python -m pytest -q -p no:cacheprovider
conda run -n kwant-validate python scripts/run_kwant_validation.py --include-array-hall
```

The expected test result is `24 passed`. See
[docs/KWANT_VALIDATION.md](docs/KWANT_VALIDATION.md) for the covered devices,
thresholds, and numerical comparison.

## Minimal examples

Generate the frozen single-texture benchmark:

```powershell
python scripts/run_stage0.py
```

The expected zero-energy values are approximately
`T(0) = 33.3240877` and `N(0) = 34`.

Compute the baseline occupied-subspace Chern convergence:

```powershell
python scripts/run_chern_convergence.py `
  --kind skyrmionium_q_zero --A 18 --R 8 --J 5 `
  --n-occ 325 --nk 11 21 31
```

Compute the baseline length scaling:

```powershell
python scripts/run_length_scaling.py `
  --kind skyrmionium_q_zero --A 18 --R 8 --J 5 --Ny 2 `
  --Nx 1 2 4 8 --energy 1.065 1.0997714941836594 1.15
```

The production calculations can be computationally expensive. In particular,
fine Chern meshes, four-terminal scans, and 100-realization disorder ensembles
should not be started as quick smoke tests.

## Paper dataset and figures

The frozen production arrays are archived separately because research data
should have an immutable dataset DOI rather than being duplicated through Git
history. Follow [data/README.md](data/README.md), then arrange the extracted data
as:

```text
data/results/stage0/
data/results/gap_scan/
data/results/chern/
...
```

The data root and output directory can be overridden:

```powershell
$env:SKYRMIONIUM_RESULTS = "D:\path\to\results"
$env:SKYRMIONIUM_FIGURES = "D:\path\to\figures"
python scripts/generate_paper_figures.py
```

Without those variables, the defaults are `data/results/` and
`generated_figures/`.

## Interpretation limits

- A high-symmetry-path opening is not called a minigap unless the full magnetic
  Brillouin-zone indirect separation remains positive.
- Numerical broadening $\eta$ is not temperature or dephasing.
- The frozen textures do not establish thermodynamic magnetic stability.
- The baseline model excludes spin--orbit coupling, interactions, phonons, and
  self-consistent magnetic dynamics.
- The finite-device Hall result is not extrapolated to a converged two-dimensional
  bulk residual.

## Data, citation, and license

- Dataset DOI: **to be added before `v1.0.0`**.
- Software DOI: generated by the GitHub--Zenodo archive of `v1.0.0`.
- Source repository: [Yang-Fuping/periodic-skyrmionium-transport](https://github.com/Yang-Fuping/periodic-skyrmionium-transport).
- Citation metadata: [CITATION.cff](CITATION.cff).
- License: [MIT](LICENSE).

---

## 简体中文

论文 *Zero-Chern Minigap and Locally Compensated Hall Response in Periodic
Skyrmionium Arrays* 的 NumPy/SciPy 可复现计算代码。

> **开发版本。** `0.1.0` 已包含经过测试的源代码和可复现流程。冻结生产数据及其 DOI
> 将在建立可归档的公开 `v1.0.0` Release 前补充。

### 研究范围

程序研究方形晶格上与冻结经典磁纹理局域交换耦合的非相互作用自旋电子：

$$
H=-t\sum_{\langle i,j\rangle}c_i^\dagger c_j
-J\sum_i c_i^\dagger(\mathbf m_i\cdot\boldsymbol\sigma)c_i
+\sum_i U_i c_i^\dagger c_i.
$$

仓库包含：

- 归一化的均匀铁磁、 $Q=0$ Skyrmionium 和 $Q=\pm1$ Skyrmion 纹理；
- 格点固体角拓扑荷及局域拓扑荷密度；
- Bloch 超胞能带、全布里渊区带隙、DOS 和 FHS Chern 数；
- 两端递归格林函数及完整矩阵逆参考实现；
- 四端 NEGF 和 Landauer--Büttiker 电压探针；
- 自旋分辨散射观测量；
- Anderson 无序统计与有限温度 Fermi 窗卷积；
- 用于选定两端与四端案例独立交叉验证的 Kwant 后端；
- 论文数值检查及主图生成程序。

生产计算仍采用 NumPy/SciPy；独立的 Conda 环境保留 Kwant 1.5 验证后端。
在选定单体与阵列案例中，Kwant/NEGF 最大相对差异为
$7.61\times10^{-6}$，完整 24 项测试全部通过。

### 仓库结构

```text
skyrmion_transport/   核心数值库
scripts/              生产计算、分析和绘图程序
tests/                19 项核心测试及 5 项 Kwant 回归测试
legacy/               冻结的单体 Skyrmionium 基准程序和参考图
data/                 独立 Zenodo 数据集的下载及目录说明
docs/                  可复现性索引和版本说明
```

新计算结果写入 `results/`，论文图写入 `generated_figures/`；两者均被 Git 忽略。

### 安装

建议使用 Python 3.10--3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux 或 macOS 使用 `source .venv/bin/activate` 激活环境。

### 数值验证

测试只使用小系统，通常数秒内完成：

```powershell
python -m pytest tests/test_core.py -q -p no:cacheprovider
```

预期结果为 `19 passed`。测试覆盖纹理归一化与拓扑荷、解析折叠能带、
均匀系统零 Chern 数、稳定条带复能带、纹理/径向轮廓对照、递归算法与完整矩阵逆透射对照、引线通道数、
Landauer--Büttiker 规范不变性和电流守恒、 $Q\to-Q$ Hall 反号关系、
稀疏/稠密算法一致性、 $W_d=0$ 无序恒等检查及温度卷积。

完整 Kwant 交叉验证使用：

```powershell
conda env create -f environment-kwant.yml
conda run -n kwant-validate python -m pytest -q -p no:cacheprovider
conda run -n kwant-validate python scripts/run_kwant_validation.py --include-array-hall
```

预期结果为 `24 passed`。覆盖案例、阈值和数值对照见
[docs/KWANT_VALIDATION.md](docs/KWANT_VALIDATION.md)。

### 最小示例

生成冻结的单体基准：

```powershell
python scripts/run_stage0.py
```

零能量预期得到约 `T(0) = 33.3240877` 和 `N(0) = 34`。

计算基线占据子空间 Chern 数收敛：

```powershell
python scripts/run_chern_convergence.py `
  --kind skyrmionium_q_zero --A 18 --R 8 --J 5 `
  --n-occ 325 --nk 11 21 31
```

计算基线长度标度：

```powershell
python scripts/run_length_scaling.py `
  --kind skyrmionium_q_zero --A 18 --R 8 --J 5 --Ny 2 `
  --Nx 1 2 4 8 --energy 1.065 1.0997714941836594 1.15
```

生产计算可能耗时较长。精细 Chern 网格、四端扫描和 100 个无序样本不应当作为快速
冒烟测试运行。

### 论文数据和主图

生产数组作为独立 Zenodo Dataset 归档，避免让科研数据在 Git 历史中重复，同时为冻结
数值证据提供不可变的独立 DOI。按照 [data/README.md](data/README.md) 下载数据，并解压为：

```text
data/results/stage0/
data/results/gap_scan/
data/results/chern/
...
```

也可以覆盖数据和图片输出目录：

```powershell
$env:SKYRMIONIUM_RESULTS = "D:\path\to\results"
$env:SKYRMIONIUM_FIGURES = "D:\path\to\figures"
python scripts/generate_paper_figures.py
```

未设置环境变量时，默认读取 `data/results/` 并输出到 `generated_figures/`。

### 解释边界

- 只有在完整磁布里渊区上间接带隙保持正值，才把高对称路径缺口称为迷你带隙；
- 数值展宽 $\eta$ 不是温度，也不是退相干；
- 冻结纹理计算不能证明阵列的热力学磁稳定性；
- 基线模型不包含自旋轨道耦合、电子相互作用、声子或自洽磁动力学；
- 有限器件 Hall 结果不外推为已收敛的二维体残余 Hall 响应。

### 数据、引用和许可证

- 数据集 DOI：在公开 `v1.0.0` 前补充；
- 代码 DOI：由 GitHub--Zenodo 归档 `v1.0.0` 后生成；
- 源代码仓库：[Yang-Fuping/periodic-skyrmionium-transport](https://github.com/Yang-Fuping/periodic-skyrmionium-transport)；
- 引用元数据：[CITATION.cff](CITATION.cff)；
- 许可证：[MIT](LICENSE)。
