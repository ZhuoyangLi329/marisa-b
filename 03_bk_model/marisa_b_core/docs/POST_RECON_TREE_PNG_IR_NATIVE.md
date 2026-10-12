# Post-reconstruction local-PNG tree IR：NERSC native 集成

本入口实现所交付方案的完整 **tree-level LO IR** 内容，不是全阶 resummation：
实空间、local PNG、严格单 wiggle `O(Pw)`、有限 PNG 至 `fNL²`、Gaussian 线性长位移。
不含 one-loop/NLO matching、`Pw²`、PNG 高阶位移累积量、RSD 或 survey mask。
代码和数值测试通过不等于 post catalogue / cosmology 拟合已验收。

## 方法和参数路径

| 方法标识 | 位移分母 | 必须使用的 tracer 参数 |
|---|---|---|
| `standard_gra` | 固定 `b_rec`，`eta=0` | 普通 halo 的 `b1,bphi,...` |
| `marisa_gra` | `b_rec + eta*fNL*bphi_rec/M`，通常 `eta=1` | 普通 halo，`p_halo` 与 `p_rec` 分开 |
| `sto_marisa` | 同 adaptive 分母 | STO 加权 tracer 的全部 bias 和加权位移噪声 |
| `sto_standard` | 固定分母，`eta=0` | STO 加权 tracer；同一个数值入口 |

本实现的路径是 **`fNL_rec = eta*fNL`**。如果目录用固定非零 `fNL_rec` 生成、
拟合只改变物理 `fNL`，必须另做固定分母函数的展开，不能把该目录直接标成 `eta=1`。
STO-only 不移动位置，沿用加权 tracer 的 pre-recon 模型，不应默认施加 post 位移阻尼。

`bphi=2*delta_c*(b1-p_halo)` 和 `bphi_rec=2*delta_c*(b_rec-p_rec)` 不可互换。
只有 `b_rec=b1, bphi_rec=bphi, eta=1` 且未加 shift noise 时，PNG 位移响应才完全相消。
当前 parent-tracer current-p 示例不满足此匹配条件。

## 实现与旧入口的关系

- `src/marisa_b/tree_png_ir.cpp`：复用原子项，增加逐腿源项自动微分和三个 adaptive 基底。
- `src/marisa_b/post_recon_tree_png_ir.{h,cpp}`：位移方差积分、CIC 方向积分、阻尼 Taylor 展开和指数混合。
- `python/marisa_b/post_recon_tree_png_ir_native.py`：批量 C ABI、结构化结果、保持原分母的节点投影。
- `python/marisa_b/post_recon_tree_png_ir.py`：用户交付的独立 Python oracle，原样导入并记录哈希。
- `scripts/production/generate_post_tree_png_ir.py`：显式配置的分块模板生成器，输出输入/动态库哈希。

旧 `NativeTreeIR` 的 24 项 ABI、固定阶和 pre IR 默认行为保留。
旧标量 post IR 入口仍拒绝活动 CIC；完整 post IR 必须显式使用 `NativePostTreeIR`。
旧 profile/MCMC 脚本不会悄悄启用新模型，也不会覆盖已有结果。

## 计算合同

1. 在同一 redshift/归一化下输入正的 `P,M,Pnw`，用 `Mnw=M*sqrt(Pnw/P)` 保持 `Pphi` 不变。
2. 指定有限且正的 `q_min`；本任务示例采用 `2*pi/1000 h/Mpc`，`kIR=.15 h/Mpc`，`rbao=110 Mpc/h`。
3. 重构平滑窗包含 Gaussian 和逐笛卡尔分量的 CIC sinc。CIC 活动时使用逐外腿方向的二维角积分。
4. 在每个实际方向的三角形节点上完成 IR，再按原测量的 shell measure 投影。
   不能对边长代表点、平均 Sigma、平均 D 或固定平面代表三角形直接套统一阻尼。
5. shell 权重必须已经除以完整 pair 数；排除零腿/闭合零模后不重新归一化。
   Haar 方向近似不能未经对照就冒充精确 FFT 格点方向。
6. shift noise 可显式输入数组/函数，默认不计。加权 tracer 的噪声不能未经测量就设成 `1/nbar`。
7. 每次改变影响位移的 bias、重构、谱或 IR 设置，必须重算；本入口不使用隐式跨参数缓存。

令 `Sigma_i² = sigma0_i + f*sigma1_i + f²*sigma2_i`，则

```
D0_i = exp(-ki²*sigma0_i)
d1_i = -ki²*sigma1_i
d2_i = ki⁴*sigma1_i²/2 - ki²*sigma2_i
G_IR = G_nw + sum_i D0_i*G_wi
L_IR = L_nw + sum_i D0_i*(L_wi + d1_i*G_wi)
Q_IR = Q_nw + sum_i D0_i*(Q_wi + d1_i*L_wi + d2_i*G_wi)
```

其中 `L_wi,Q_wi` 包含 adaptive 源项；二阶基底已经是系数，不能再除以二。
**std 即使分母固定，halo 位移的分子仍含 `fNL*bphi/M`，因此指数混合一般非零。**
`source_operator_terms` 仅含源项阻尼，不能当成完整 G/L/Q；使用 `deterministic` 获得完整结果。
stochastic 源项与指数诱导的三种 mixing 单独输出。
只有显式选择同一 alpha3 跨 Gaussian/linear/quadratic 阶时，才调用 `stochastic_if_common_alpha3()`。

## 构建与调用

在 NERSC 工作副本根目录运行（依赖目录保持只读，无需安装包）：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REACT_DIR=/pscratch/sd/l/lzy/reconstruction-png/codes/marisa_b/external/ACTio-ReACTio bash scripts/build_tree_png_ir.sh
PYTHONPATH=python /global/homes/l/lzy/anaconda3/bin/python -m pytest -q \
  tests/python/test_tree_png_ir_resummation.py \
  tests/python/test_post_recon_tree_png_ir.py \
  tests/python/test_post_recon_tree_png_ir_native.py
```

显式批量 API（`p,nw,m` 是 `N x 3`，向量是 `N x 3 x 3`；q 谱是共享的一维数组）：

```python
import numpy as np
from marisa_b.post_recon_tree_png_ir_native import NativePostTreeIR
from marisa_b.post_recon_tree_png_ir import AdaptiveReconstructionConfig, PostReconIrSettings

engine = NativePostTreeIR("build/tree_png_ir/libmarisa_post_tree_png_ir.so")
settings = PostReconIrSettings(r_bao=110, k_ir=.15, q_min=2*np.pi/1000)
adaptive = AdaptiveReconstructionConfig(bphi_recon=bphi_rec, fnl_recon_ratio=1)
batch = engine.evaluate(vectors, p, nw, m, bias, reconstruction,
                        q, qpower, qnw, qm, settings, adaptive)
G, L, Q = batch.deterministic.T
B = G + fnl*L + fnl**2*Q
result = batch.result(0)  # 包含 source-only、mixing、Sigma 和 D 的可审计单节点结果
```

完整可运行的 current-p/真实谱/精确首个 shell 示例位于本任务
`03_bk_model/halo/code/audit_post_tree_png_ir.py`；它生成三种方法各自的 NPZ 输入、JSON 配置和模板。
模板入口可对这些文件再运行；新输出名必须不存在：

```bash
python scripts/production/generate_post_tree_png_ir.py \
  --input ../halo/outputs/local_png_post_tree_ir_20260930/audit/marisa_gra_exact_shell_nodes.npz \
  --config ../halo/outputs/local_png_post_tree_ir_20260930/audit/marisa_gra_config.json \
  --output ../halo/outputs/local_png_post_tree_ir_20260930/audit/marisa_gra_shell_recheck.npz
```

该示例的 `p_halo` 是已有 pre / STO-only 校准的 parent-tracer 映射，
`b2=.3,gamma2=-.15` 是明确声明的演示 nuisance 值；不能解读为重新测得 post bias。

## 验收状态与未完成的物理判断

本次 native/oracle 单元测试与现有 native 基底交叉检查在 NERSC 上执行。
真实谱审计比较 `24x48 → 32x64 → 48x96` IR 角积分、257/513/1025/2049 径向网格、
qmin/kIR/rbao 敏感性、fNL=±100 方差 Taylor 截断，以及实际 FFT 首个 shell 的 306 个非零闭合节点。
节点权重和是 `306/324`，而非 1。分块模板结果需与直接投影一致。
`scaled_error=max(abs(a-b)/max(1,abs(b)))`；不是观测协方差加权误差。

输出日志与审计见 `03_bk_model/halo/outputs/local_png_post_tree_ir_20260930/`。
完整 `make release-test` 的 Git 边界检查不能在当前无 `.git` 的工作副本通过；
全仓库 Python suite 还涉及未迁入的 nonlocal/production 文件。不能把这些门禁报告为通过。

还需要所有拟合 shell 的协方差加权数值收敛、可信 post PNG 目录、专属 post bias/噪声校准，
以及实际 B-only 注入恢复/kmax 扫描。**本次不声称 IR 已提升 fNL 的可靠 kmax。**
既有 `SCIENTIFIC_STATUS.md` 的 provisional / halo post PNG catalogue 限制保持不变。

详细推导与参考 oracle 合同见
[交付方案](POST_RECON_TREE_PNG_IR_RESUMMATION_PLAN.md) 和
[oracle API](POST_RECON_TREE_PNG_IR_API.md)。
