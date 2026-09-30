# 四种重构方法的理论映射与 current-p 示例

本文是阅读 MARISA-B 的方法索引，聚焦 **real-space、local-PNG、tree-level**。
MARISA-B 是双谱理论框架；下文的 **Marisa** 则是一个 PNG-aware 重构方法，二者不是同一个概念。
这里补充定义、参数约定和可运行的算术示例，不发布新的 post-reconstruction 拟合结果，也不更改生产模型的科学验收状态。

## 1. 先区分 tracer、位移算法和 likelihood

| 用户名称 | 稳定方法标识 | 被建模的 tracer | 位移/理论分支 | 不能混用的输入 |
|---|---|---|---|---|
| pre，对照 | `pre` | 普通 halo | 无重构位移，pre kernel | 普通 halo 的 bias、噪声、数据与 covariance |
| std | `standard_gra` | 普通 halo | 固定 `b_rec` 的 post kernel | std 实际使用的 window、`b_rec` 和 post covariance |
| marisa | `marisa_gra` | 普通 halo | PNG-aware/adaptive `b_rec(k)` | 实际 `p_rec`、`fNL_rec`、window 和 Marisa covariance |
| sto | `sto_recon` | STO 加权 halo | 不改位置；当前近似采用加权 tracer 的 pre 型 kernel | 加权场的 bias、PNG 响应、噪声和 covariance |
| sto+marisa | `sto_marisa` | STO 加权 halo | 在加权场上使用 adaptive post kernel | 加权 tracer 参数和该组合的位移配置、covariance |

外部任务中还有 `sto_standard`，即 STO 加权场接固定 `b_rec` 重构；本示例不额外增加该分支。
旧分析里的 `standard`/`marisa` 与此表的 `standard_gra`/`marisa_gra` 对应。

**STO 不等于代码里的 stochastic terms。** 前者是构造加权 tracer 的方法；后者是双谱模型中的随机/噪声贡献。
外部 STO 数据文件即使叫 `post_B000`，也不代表其理论必须打开 gravitational reconstruction kernel。

### 普通 tracer 与 STO 加权 tracer

STO 场可表示为

\[
\delta_w(\mathbf x)=\frac{\sum_a w_a\delta_D(\mathbf x-\mathbf x_a)}{\bar n_w}-1,
\qquad \bar n_w=\left\langle\sum_a w_a\delta_D(\mathbf x-\mathbf x_a)\right\rangle.
\]

STO 改权重而不移动 \(\mathbf x_a\)。当前任务基线是 **matter-assisted N+Env**：
`env_source=dm`、`weight_source=dm`、`lambda_th=1`、number weighting。
它利用模拟 DM，不能直接称作可用于观测的 halo-only 方法。

当前采用的建模近似，是把加权场当作一个具有独立有效 bias 和噪声参数的 tracer，复用通用 halo tree 算符基底。
这不是“任意加权都精确等价于改一个 b1”的定理；环境选择、权重变化和 PNG 响应的完备性仍需验证。
尤其不能把普通 halo 的 Poisson 噪声或 covariance 直接复制给 STO。
加权离散噪声涉及权重矩，不能未经 estimator/随机项约定核对就只替换一个数密度。

## 2. std 与 Marisa 的理论差别

按本仓库的实空间约定，data 与 random 使用同一个位移，post 场是 displaced data 减 shifted randoms。
重构 kernel 的位移块为

\[
R_{\mathbf k}(\mathbf q)=-\frac{\mathbf k\cdot\mathbf q}{q^2}
\frac{W_{\rm rec}(\mathbf q)}{b_{\rm rec}(q)},
\qquad
W_{\rm rec}(\mathbf q)=e^{-q^2R^2/2}
\prod_{a=x,y,z}\operatorname{sinc}^{4}(q_a\Delta/2).
\]

std 使用固定分母 \(b_{\rm rec}(q)=b_{\rm rec}\)。Marisa 使用

\[
b_{\rm rec}(q;f_{\rm NL}^{\rm rec})=b_{\rm rec}
+f_{\rm NL}^{\rm rec}\frac{b_{\phi,\rm rec}}{\mathcal M(q)},
\qquad b_{\phi,\rm rec}=2\delta_c(b_{\rm rec}-p_{\rm rec}).
\]

这里 \(\delta_L=\mathcal M\phi\)，重构实现的 `alpha(k)` 必须与 \(1/\mathcal M(k)\) 的红移、归一化约定相符。
`R` 和网格 cell size `Delta` 是位移算法参数，不是 B000 测量网格的同义词。
示例采用 z=0.5、R=10 Mpc/h、Delta=8 Mpc/h；[早期 bias-v1 推导](halo_v1_derivation.md#reconstruction-map)中的 R=15 等数值不是本例配置。

### 必须分开的两组参数

- `p_halo` 定义被建模 tracer 的 \(b_\phi=2\delta_c(b_1-p_{\rm halo})\)。
- `p_rec` 定义生成位移时使用的 \(b_{\phi,\rm rec}\)。更新 `p_halo` 不会自动改变已生成 catalogue 的 `p_rec`。
- `fNL` 是宇宙学/理论参数；`fNL_rec` 是位移算法的设定。`fNL_rec=fNL` 是显式 matched/oracle 路径，不能当成固定 catalogue likelihood 的隐含默认值。
- 普通 halo 的理论固定值 `b1=1.93` 与位移使用的 `b_rec=1.9326526728794324` 也不是同一个字段。

对固定重构 catalogue 扫描宇宙学 `fNL` 时，应保留该 catalogue 的 `fNL_rec` 等 metadata；
若沿着 `fNL_rec=fNL` 变化，讨论的是另一条重构/响应路径，不能混为同一个观测向量。

### local-PNG tree 的组合方式

固定重构分母的确定性部分记为

\[
B_{\rm fixed}^{\rm det}=G+fL+f^2Q,\qquad f=f_{\rm NL}.
\]

native 导出的三个 adaptive raw basis 是 `gaussian_linear`、`gaussian_quadratic`、`png_linear_cross`。
对应归一化为

\[
G_1=b_1^4\frac{b_{\phi,\rm rec}}{b_{\rm rec}}\,T_{G1},\quad
G_2=b_1^4\left(\frac{b_{\phi,\rm rec}}{b_{\rm rec}}\right)^2T_{G2},\quad
Q_\times=b_1^3b_\phi\frac{b_{\phi,\rm rec}}{b_{\rm rec}}\,T_\times.
\]

保留总阶数不超过二阶时，示意组合为

\[
B_{\rm adaptive}^{\rm det}=G+fL+f^2Q+rG_1+r^2G_2+frQ_\times,
\qquad r=f_{\rm NL}^{\rm rec}.
\]

`G2` 已是平方项的系数，**不能再乘一次 1/2**。这是原点附近的截断展开，不是对任意大 `fNL_rec` 精确计算有理分母。
上述三个 basis **不包含完整 adaptive stochastic closure，也不包含完整 adaptive IR damping**；随机项必须按独立约定补入。
当 `r=f` 时，线性系数变为 `L+G1`，二次系数变为 `Q+G2+Q_cross`；当 `r=0` 时回到固定分母分支。

## 3. 代码阅读入口

| 要回答的问题 | 仓库文件 / 符号 |
|---|---|
| 固定/尺度依赖分母和重构配置如何表达？ | [`src/halo_v1/halo_v1.h`](../src/halo_v1/halo_v1.h)：`ReconstructionConfig`；对应 `.cpp` 的 reconstruction kernel |
| pre/std 的 local-PNG 原子项在哪里？ | [`src/marisa_b/marisa_b_native.h`](../src/marisa_b/marisa_b_native.h) 和 [实现](../src/marisa_b/marisa_b_native.cpp)：`compute_pre_recon_halo_bias_v1_local_png_tree_dfNL`、`compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors` |
| Marisa 的有限二阶 adaptive 项在哪里？ | 同一 native 文件中的 `AdaptiveBrecFiniteTreeBasis`、`compute_post_recon_halo_local_png_brec_finite_tree_bases_vectors` |
| 如何导出可投影的原子基底？ | [`src/eft_v2/post_r1_png_template_driver.cpp`](../src/eft_v2/post_r1_png_template_driver.cpp)：`tree-fixed`、`adaptive-brec-tree` 两个 sector |
| 是否验证过系数而非只看总 B？ | [`tests/eft_v2/test_post_r1_png_native.cpp`](../tests/eft_v2/test_post_r1_png_native.cpp)：独立有限差分及二次/交叉项测试 |
| Python 生产适配器如何组装模板？ | [`scripts/production/run_post_recon_halo_finite_png_v1.py`](../scripts/production/run_post_recon_halo_finite_png_v1.py)：`PngTemplateSet.components`；其原有默认值不是本例 current-p 配置的自动消费者 |
| STO/组合方法如何对应这些接口？ | 本文第 1 节与下方配置示例；不需要另复制一份 native kernel，但需要自己的 tracer 参数、数据和验证 |

生产脚本中的旧 `P_UNIVERSALITY` 等默认值并不会因为新加这份文档而更新。
本例也不是可直接传给所有生产 runner 的通用 inference 配置。
在真正连接 runner 时，必须逐一核对 bias override、`bphidelta` anchor、模板 metadata、estimator 和 covariance。

## 4. `current p calibration` 到底指什么？

这里冻结的是 2026-09-30 核对的任务设置，而不是一个随 `main` 自动变化的全局默认值。
两个 `p_halo` 值来自任务 runner 记录的 P0、固定输入 fNL=+100 的校准；本次没有重新拟合校准或传播其不确定度。

| tracer 校准 | 理论 b1 | p_halo | bphi，delta_c=1.686 | 适用角色 |
|---|---:|---:|---:|---|
| 普通 pre halo / 当前 IR 扫描 | 1.93 | 1.1788659191136683 | 2.532824120748710 | `current p calibration` 标签的准确含义 |
| STO-only 加权 tracer | 2.197952896776342 | 1.3394637946618602 | 2.894825252330032 | 外部 STO-only 校准，不是新做的 STO+Marisa post 校准 |

例子把普通 tracer 参数映射到 std/Marisa，把 STO tracer 参数映射到 STO+Marisa，**只用于演示共享理论的连接方式**。
这不宣称四个 post 方法都已用这些数值通过验收，也不取代外部单独的 post P0 校准。
外部已有其他 std/Marisa/STO+Marisa 校准记录；它们不能与这个明确命名的 pre IR 快照混称为同一组“current p”。

实际方法示例中的位移参数另存为：普通 halo `b_rec=1.9326526728794324`，STO 组合 `b_rec=2.197952896776342`，Marisa `p_rec=1`。
不能直接把 `p_rec` 改成上表的 `p_halo`，然后继续使用原来的 post catalogue。

### current-p 的其他 bias 约定

当前 pre IR 扫描只更新 `bphi` 的 p，保留 `bphidelta` 的独立 universal anchor p=1：

```python
b2_native = b2 - 4 * gamma2 / 3
b2_lagrangian = b2_native - 8 * (b1 - 1) / 21
bphi = 2 * delta_c * (b1 - p_halo)
bphidelta = 2 * delta_c * (b1 - 1) + 2 * (delta_c * b2_lagrangian - b1 + 1)
bphi2 = 4 * delta_c * (delta_c * b2_lagrangian - 2 * (b1 - 1))
```

因此不能将上面 `bphidelta` 的第一项无条件替换成新的 `bphi`。
这是该扫描采用的关系，不是所有 tracer 都必须遵循的普适定理。
STO 行展示沿用此关系时的参数计算，不宣称验证了加权 tracer 全部高阶 PNG bias。

## 5. 无数据、无编译依赖的可运行例子

机器可读快照：[`configs/reconstruction_current_p_example.json`](../configs/reconstruction_current_p_example.json)。
算术示例：[`scripts/examples/reconstruction_current_p.py`](../scripts/examples/reconstruction_current_p.py)，只需 Python 标准库。

```bash
# 匹配路径仅作为显式教学例子；M=1000 是一个假定输入，不是新计算的 transfer table。
python scripts/examples/reconstruction_current_p.py --fnl 100 --fnl-rec 100 --transfer-m 1000

# 宇宙学 fNL=100，但使用 fNL_rec=0 的位移：两个参数可以独立设置。
python scripts/examples/reconstruction_current_p.py --fnl 100 --fnl-rec 0 --transfer-m 1000

# 仅用于查看改变位移 p 的算术影响；不会更改 tracer p，也不会重新生成 catalogue。
python scripts/examples/reconstruction_current_p.py --p-rec 1.2

# 独立运行新测试，无需 pytest 或模拟数据；pytest 也能发现这些 unittest 测试。
python tests/python/test_reconstruction_current_p.py -v
```

第一条命令的部分预期输出：

| 方法 | tracer bphi | 位移 bphi_rec | 给定 M=1000、fNL_rec=100 时的分母 |
|---|---:|---:|---:|
| pre | 2.53282412 | 不适用 | 不适用 |
| std | 2.53282412 | 不适用 | 1.93265267，固定 |
| Marisa | 2.53282412 | 3.14490481 | 2.24714315 |
| STO-only | 2.89482525 | 不适用 | 不适用 |
| STO+Marisa | 2.89482525 | 4.03949717 | 2.60190261 |

脚本还打印 `Z1=b1+fNL*bphi/M` 和三个 adaptive raw basis 的归一化系数；它**不计算 raw basis 本身**，也不生成 B(k)、IR prediction、posterior 或 kmax 结论。
默认 `b2=gamma2=0` 只是展示用 nuisance 输入，不是最佳拟合值。
分母非正时脚本报错，不采用 bias floor 或 clipping；如果真实重构启用了 floor，必须另外匹配该非线性规则。

## 6. 统计、IR 与适用边界

当前体积例子是 V=1 (Gpc/h)^3，之前的对照体积是 42.875；改变 V 缩放的是数据 covariance，不是理论 kernel 或 p。
pre 扫描使用五个真值 -100、-50、0、50、100 的样本均值作为中心，以及 Fid500 的单 realization covariance/V 和 Hartlap 修正；不是 covariance/500。
自由 nuisance 是 b2、gamma2、Ashot_residual、Bshot_residual；没有把 P likelihood 加入该 B-only 扫描。

不同方法必须使用自身、与 estimator 和样本选择匹配的 covariance。不能把 pre 的误差、通过 kmax 或 bias 校准无条件转移到 post。
若传播 p 校准的不确定度，应另外建立联合模型或先验；那不再是这里固定 p 的例子。

本次补充**没有把外部新 tree-PNG IR 实现或扫描数据导入 GitHub**。
仓库原有的 Gaussian/EFT IR 文件不能被当成这轮完整 local-PNG tree IR 扩展。
CIC 的方向依赖位移 covariance、adaptive 分母对阻尼指数的响应，均不能由“修改 tree source 项”代替。
post 的 provisional 状态仍遵循 [`SCIENTIFIC_STATUS.md`](SCIENTIFIC_STATUS.md)。

快照 JSON 的 `provenance` 记录外部来源相对路径和 SHA256，仅用于辨识核对过的来源；
它们不是本仓库下载链接，也不是运行本例需要的文件，未打包机器路径、模拟数据、协方差或外部运行环境。

## 7. 给代码审阅者的阅读顺序

先读本文方法表和参数分离，再运行示例，随后沿第 3 节链接阅读 native 和回归测试。
请分别回答“方法定义是什么”“哪些确定性项已经实现”“噪声/IR/数据验证还有哪些限制”；
不要因为 std 和 Marisa 共享 native core，就把两种重构等同；也不要因为没有独立 `sto.cpp`，就断定通用 tracer 理论无法描述加权场。
