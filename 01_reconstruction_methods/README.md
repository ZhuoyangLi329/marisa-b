# 第一部分：重构方法

本部分保存后续教学、示范和维护所需的核心代码。迁移依据是实际调用关系，
三种基础方法共用 catalog、随机样本和统计量测量代码；组合方法不复制数值内核。

## 阅读顺序与命名

| 分类 | 必要文件 | 用途 |
|---|---|---|
| 标准 gra-recon | `standard_gra/run_halo_b000_standard_gra.py` | 标准方法示范入口 |
| 标准位移与公共辅助 | `common/quijote_dm_2d_paperbin_runner_tools.py` 的 `run_jaxrecon` | 真正调用 jaxrecon 的位移逻辑；随机点、周期 wrap 与校验 |
| Marisa gra-recon | `marisa_gra/png_recon_tools.py` | 原样复制的 PNG-aware 位移核心和 alpha 表格 |
| Marisa 连接器 | `marisa_gra/marisa_recon_runner_tools.py` | 直接调用本目录 Marisa 核心，保持 shifted-random convention |
| STO | `sto_recon/run_halo_sto_variants.py` | 现役完整权重求解器，支持 N+Env、mass-only 与组合分量 |
| STO + gra-recon | `combined/run_sto_standard.py`、`combined/run_sto_marisa.py` | 使用 STO 加权 catalog 接入对应 gra-recon |
| 共用测量示例 | `common/halo_b000_pipeline.py` | 一次 realization 的 pre/recon/post B000 流程 |
| 运行兼容性 | `common/stovar_jaxpower_compat.py` | 现有 lsstypes 摘要 API 适配 |
| 示例参数 | `examples/*.jsonl` 和两份 bias JSON | 每种方法只保留一条 fiducial 示例与 bias 依据 |

`run_*` 是执行入口，`*_tools.py` 是函数库，`submit_*.slurm` 是提交模板。
方法目录内保留稳定名称，样本、红移、阈值、fNL 等条件放在配置和输出 metadata。

## 科学口径

标准和 Marisa 都属于 gravitational reconstruction。共同保留 real-space
`growth_rate=0`、data/random 都使用 `disp`、post 场使用 displaced data 减 shifted randoms。
Marisa 采用 `b_eff(k)=b1+2*delta_c*(b1-p)*fNL*alpha(k)`，这里只适用于 local PNG。

示例沿用原 halo manifest：z=0.5、Mmin=1e13、R=10、重构 cellsize=8 Mpc/h、
B000 mesh=256、15 个 centered bins、完整 Sugiyama 上三角 120 维。
重构网格和 B000 测量网格是独立配置，不应混为一谈。
标准/普通 Marisa 的 b1=1.9326526728794324；STO 组合示例的
b1=2.197952896776342；Marisa 示例沿用历史 p_rec=1，不能与 p_halo=1.12 混用。

STO 基线示例是 **matter-assisted N+Env**：env_source=dm、weight_source=dm、
lambda_th=1、number weighting。它利用模拟 DM，不能描述为直接适用观测的 halo-only 方法。
现役求解器同时支持 halo-split 自监督、monly、m2env、m4env、m4rho3env；
这些是已实现的配置能力，不意味着所有变体已经生产验证。
STO 只改 halo 权重、保留位置；输出 `post_sto_halo.npz` 是组合方法的接口。

## 环境与示例

```bash
cd /pscratch/sd/l/lzy/reconstuction-png-v2/01_reconstruction_methods
source common/nersc_env.sh
python standard_gra/run_halo_b000_standard_gra.py --help
python marisa_gra/run_halo_b000_marisa_gra.py --help
python sto_recon/run_halo_sto_variants.py --manifest examples/sto_recon.jsonl --task-index 0 --check-only
```

gra-recon 的 `--check-only` 会构造 B000 binner，仍需要实际数值环境；不是纯文本检查。
直接运行 gra 入口默认使用一条本地 fiducial 配置。
生产条件以 manifest 为准；旧 runner 的兼容默认值不代表当前推荐配置。

组合方法应先运行 STO source，再调用 `combined/run_sto_standard.py` 或
`combined/run_sto_marisa.py`。加权场的 bias 必须单独校准；本示例的 STO bias 仅适用
已给出的 N+Env 基线，改变 STO 配置后不能照搬。

shared GPU 提交示例（仅一条，不是整批生产请求）：

```bash
mkdir -p logs
sbatch --array=0 --chdir="$PWD" --output="$PWD/logs/standard_%A_%a.out" \
  --error="$PWD/logs/standard_%A_%a.err" standard_gra/submit_standard_gra.slurm
```

数值依赖复用 NERSC 现有 cosmodesi 环境中的 numpy/h5py/mpi4py/pmesh/jax/jaxrecon/
jaxpower/clustering_statistics/cosmoprimo；不安装新包、不复制整个环境。
Marisa 科学核心已经本地化，不再要求旧 marisa-recon checkout。
Quijote 原始数据共享读取，不复制大数据。

## 来源和验证范围

参见 `MIGRATION_MANIFEST.json` 的来源文件、SHA256 与改动清单。
此轮迁移不提交模拟生产、不重复 Fisher/MCMC，也不宣称新的科学结果。
具体验证结论见 `VALIDATION.md`。
