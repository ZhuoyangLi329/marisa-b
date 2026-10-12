# MARISA-B：tree-level finite-local-PNG halo bispectrum 的 IR resummation 设计与实施计划

**版本日期：2026-09-30**
**目标范围：仅 tree level；不使用 Gaussian/PNG one-loop，不做 tree–loop matching。**

---

## 0. 执行摘要

本计划建议在 MARISA-B 当前 real-space finite-local-PNG halo tree 模型

\[
B_h^{\rm tree}(f_{\rm NL})=G_h+f_{\rm NL}L_h+f_{\rm NL}^2Q_h
\]

中加入 IR resummation。结论分成三层：

1. **有必要**：只要分析范围包含 BAO 尺度，长波 bulk-flow 对 oscillatory transfer/power 的阻尼并不会因为模型只保留 tree level 而消失。若 Gaussian tree、primordial \(B_{111}\)、\(b_\phi\)、\(b_{\phi\delta}\)、advection 和 \(b_{\phi^2}\) 项仍使用未阻尼的 BAO，模型在 BAO 区域会系统性保留过强的振荡。
2. **可行**：MARISA-B 的 tree 模型是外部三条腿上的 \(P_L(k_i)\)、\(\mathcal M(k_i)\) 与 IR-safe bias/gravity kernels 的代数函数，因此可以在不引入任何 hard loop 的情况下，对全部 \(G,L,Q\) 及现有 stochastic raw bases 做统一的 legwise IR dressing。
3. **推荐的 fiducial 定义**：采用“非微扰地重求和长位移、但严格保留到一个 BAO wiggle insertion”的 leading-IR/leading-wiggle 结果。本文将其记为 `strict_lo`。它是当前最可控、最容易逐项验证、且不会偷偷混入 one-loop matching 的实现。

代码还提供 `factorized_closure`：直接把每个 \(P_L\) 替换成

\[
P_{\rm IR}=P_{\rm nw}+e^{-k^2\Sigma^2}P_{\rm w}
\]

并同步构造 \(\mathcal M_{\rm IR}\)。它与 `strict_lo` 在 \(O(P_{\rm w})\) 完全一致，但自动保留 \(P_{\rm w}^2\) 及更高 wiggle products。那些高阶项对一般 PNG bias operators 尚无逐算符的完整推导，因此它只作为系统学 closure，而不是 fiducial 理论定义。

### 本次交付内容

- `python/marisa_b/tree_png_ir_resummation.py`：独立、可审计的 Python oracle；
- `tests/python/test_tree_png_ir_resummation.py`：针对公式、native 字段约定与 IR 极限的测试；
- 本文档：推导、代码接口、native 集成路线与验收标准；
- 可直接应用到仓库的新文件 patch。

---

## 1. 精确定义“完整 tree-level IR resummation”

“完整”容易产生两种不同含义，必须先区分。

### 1.1 本项目需要的“完整”

本文的 fiducial 目标是：

- 对 MARISA-B 当前 tree-level finite-PNG 模型中的**全部确定性算符**进行 IR dressing；
- 同时处理 \(P_L\)、\(\mathcal M\)、\(1/\mathcal M\)、primordial \(B_{111}\)、PNG bias、advection、fixed-\(b_{\rm rec}\) reconstruction source kernel；
- 同时处理当前 native 返回的三种 stochastic raw basis；
- 对长位移的增强贡献保留指数阻尼 \(e^{-k^2\Sigma^2}\)，即对 IR displacement 非微扰；
- 在 BAO wiggle expansion 中严格保留 \(O(P_{\rm w})\)。

这就是 `strict_lo`。

### 1.2 本文不把什么称为“已证明的完整结果”

以下内容不应被悄悄混入 fiducial：

- \((1+k^2\Sigma^2)e^{-k^2\Sigma^2}\) 的 tree 修正：它属于 tree + one-loop 的 NLO matching；当前项目不使用 one-loop，因此不应出现。
- 将所有 \(P_L\) 直接换成 \(P_{\rm IR}\) 后产生的任意高次 \(P_{\rm w}\) 项：这是 practical closure；在 TSPT 中 tree bispectrum 的受控推导是 \(O(P_{\rm w})\)。
- 对 MARISA-B 每个 \(1/\mathcal M\) PNG bias operator 的完整 \(P_{\rm w}^2\)、\(P_{\rm w}^3\)、……ULPT 位移核：目前没有逐算符的现成文献公式。
- Cartesian CIC 引起的方向依赖 IR covariance，以及 adaptive \(b_{\rm rec}(k,f_{\rm NL}^{\rm rec})\) 对阻尼指数自身的响应：标量 \(\Sigma^2\) 不足以完整描述这些情况。

因此，本文的科学措辞应是：

> **完整覆盖 MARISA-B 现有 tree-level PNG halo operator basis 的 leading-IR、leading-wiggle resummation。**

---

## 2. MARISA-B 当前 tree 模型的合同

采用当前 native `halo_bias_v1` 约定：

\[
P_i\equiv P_L(k_i),\qquad
u_i\equiv \frac{1}{\mathcal M_i},\qquad
U_{ij}=u_i+u_j,
\]

\[
A_{ij}=\mu_{ij}
\left(\frac{k_i}{k_j}u_i+\frac{k_j}{k_i}u_j\right),
\]

\[
H_{ij}=b_1F_2(\mathbf k_i,\mathbf k_j)
+\frac{b_2}{2}
+b_{K^2}\left(\mu_{ij}^2-\frac13\right).
\]

这里 \(b_2\) 是 native 的 \(\delta^2/2\) 系数，\(b_{K^2}\) 乘 \(\mu^2-1/3\)。若上层使用 \((b_2^{\rm EFT},\gamma_2)\)，应继续由现有 adapter 完成

\[
b_2^{\rm native}=b_2^{\rm EFT}-\frac43\gamma_2,
\qquad b_{K^2}=\gamma_2.
\]

### 2.1 Gaussian tree

\[
G_h=2b_1^2\sum_{\rm cyc}P_iP_jH_{ij}.
\]

### 2.2 线性 PNG 系数

定义单位 \(f_{\rm NL}\) 的 primordial matter bispectrum

\[
\widehat B_0
=2\sum_{\rm cyc}P_iP_j
\frac{\mathcal M_k}{\mathcal M_i\mathcal M_j}.
\]

则

\[
\boxed{
L_h=b_1^3\widehat B_0
+\sum_{\rm cyc}P_iP_j
\left[
2b_1b_\phi U_{ij}H_{ij}
+b_1^2b_\phi A_{ij}
+b_1^2b_{\phi\delta}U_{ij}
\right].
}
\]

这包含 native 中的：

- `dBdfNL_local_primordial`；
- `dBdfNL_local_bphi_f2`；
- `dBdfNL_local_bphi_advection`；
- `dBdfNL_local_bphidelta`；
- `dBdfNL_local_bphi_b2`；
- `dBdfNL_local_bphi_bK2`；
- post 情况下的 `dBdfNL_local_bphi_reconstruction`。

### 2.3 二次 PNG 系数

\[
\boxed{
\begin{aligned}
Q_h={}&b_1^2b_\phi(u_1+u_2+u_3)\widehat B_0\\
&+\sum_{\rm cyc}P_iP_j\Big[
2b_\phi^2u_iu_jH_{ij}
+b_1b_\phi^2U_{ij}A_{ij}\\
&\hspace{26mm}
+b_1b_\phi b_{\phi\delta}U_{ij}^2
+b_1^2b_{\phi^2}u_iu_j
\Big].
\end{aligned}}
\]

这里必须继续区分：

- \(b_\phi^2\)：两个线性 PNG bias factors 的乘积；
- \(b_{\phi^2}\)：独立的 \(\phi^2\) bias coefficient。

代码中 \(Q_h\) 是 \(f_{\rm NL}^2\) 的系数，即

\[
Q_h=\left.\frac12\frac{\partial^2B_h}{\partial f_{\rm NL}^2}\right|_0.
\]

### 2.4 当前 stochastic raw bases

native 当前返回

\[
S_{\alpha_3}^{G}=b_1^2\sum_iP_i,
\]

\[
S_{\alpha_3}^{L}=b_1b_\phi\sum_i\frac{P_i}{\mathcal M_i},
\]

\[
S_{\alpha_3^{\rm PNG}}^{Q}=b_\phi^2\sum_i\frac{P_i}{\mathcal M_i^2},
\qquad S_{\alpha_4}=1.
\]

\(\alpha_3/n_{\rm bar}\)、\(\alpha_4/n_{\rm bar}^2\) 等幅度仍由数据/likelihood 层加入，不进入本 oracle。

---

## 3. IR resummation 的物理输入

### 3.1 Smooth–wiggle 分解

在同一红移处写成

\[
P_L(k)=P_{\rm nw}(k)+P_{\rm w}(k).
\]

所有 production 路径必须共享同一份 \(P_{\rm nw}\) 与 \(P_{\rm w}\)，不能让 Gaussian tree、PNG tree 和投影器分别做不同 smoothing。

### 3.2 BAO displacement variance

pre-reconstruction real-space 标量阻尼尺度定义为

\[
\boxed{
\Sigma^2(k_{\rm IR})=
\frac{1}{6\pi^2}
\int_0^{k_{\rm IR}}dq\,P_{\rm nw}(q)
\left[1-j_0(qr_{\rm BAO})+2j_2(qr_{\rm BAO})\right].
}
\]

要点：

- \(P_{\rm nw}\) 和外部 tree power 必须处于同一红移；
- \(q\) 与 \(r_{\rm BAO}\) 必须采用互为倒数的单位；
- 不应在代码中偷偷固定 \(r_{\rm BAO}\) 或 \(k_{\rm IR}\)；
- 建议把 \(k_{\rm IR}\) 变化作为 LO 理论误差诊断，例如检查 \(0.05\)、\(0.10\)、\(0.15\)、\(0.20\,h\,{\rm Mpc}^{-1}\)，而不是预先假定某一个值无误。

每一条 external leg 的 leading damping 是

\[
D_i\equiv e^{-k_i^2\Sigma^2}.
\]

### 3.3 为什么 bias kernels 本身不乘阻尼

长波 bulk flow 近似共同平移 matter 与 comoving halos。局域 bias/gravity kernels \(F_2\)、常数 \(b_2\)、tidal kernel 和 PNG operator coefficient 本身是 IR safe 的。需要重求和的是它们所乘的 oscillatory external spectra/transfer legs，而不是把整个 bispectrum 粗暴乘一个统一的 \(e^{-(k_1^2+k_2^2+k_3^2)\Sigma^2}\)。

---

## 4. 关键问题：\(P_L\) 与 \(\mathcal M\) 必须一致地 IR dress

MARISA-B 使用

\[
\delta_L(k)=\mathcal M(k)\phi(k),
\qquad
P_\phi(k)=\frac{P_L(k)}{\mathcal M^2(k)}.
\]

BAO 位于 late-time transfer 中，不应通过 IR resummation 改写 primordial spectrum。因而先定义

\[
\boxed{
\mathcal M_{\rm nw}(k)
=\mathcal M(k)\sqrt{\frac{P_{\rm nw}(k)}{P_L(k)}}
}
\]

使得

\[
\frac{P_{\rm nw}}{\mathcal M_{\rm nw}^2}
=
rac{P_L}{\mathcal M^2}=P_\phi.
\]

这一步对 PNG 尤其关键。若只阻尼 \(P_L\) 而保留原始 \(\mathcal M\)，则

- primordial \(B_{111}\) 的三条 transfer legs 不一致；
- \(b_\phi/\mathcal M\)、advection、\(b_{\phi\delta}\) 与 \(b_{\phi^2}\) 会保留错误的 BAO；
- \(P_L/\mathcal M^2\) 会伪造出 primordial BAO。

---

## 5. Fiducial `strict_lo` 的统一推导

把任意一个 tree-level atomic term 或聚合量记作

\[
X=X(P_1,P_2,P_3;\mathcal M_1,\mathcal M_2,\mathcal M_3),
\]

其中 \(X\) 可以是 \(G\)、\(L\)、\(Q\)，也可以是任一 native operator field 或 stochastic basis。

对每一条 leg 引入独立参数 \(\epsilon_i\)：

\[
P_i(\epsilon_i)=P_{{\rm nw},i}+\epsilon_iP_{{\rm w},i},
\]

\[
\mathcal M_i(\epsilon_i)
=\mathcal M_{{\rm nw},i}
\sqrt{1+\epsilon_i\frac{P_{{\rm w},i}}{P_{{\rm nw},i}}}.
\]

于是对任意 \(\epsilon_i\)，都有

\[
\frac{P_i(\epsilon_i)}{\mathcal M_i^2(\epsilon_i)}=P_{\phi,i}.
\]

严格的 leading-wiggle IR-resummed 结果是

\[
\boxed{
X_{\rm IR}^{\rm strict}
=X_{\rm nw}
+\sum_{i=1}^3D_i
\left.\frac{\partial X}{\partial\epsilon_i}\right|_{\epsilon_1=\epsilon_2=\epsilon_3=0}.
}
\]

其中

\[
X_{\rm nw}=X(P_{{\rm nw},i};\mathcal M_{{\rm nw},i}).
\]

### 5.1 这一表达式为什么适合 MARISA-B

它有以下性质：

1. **逐腿阻尼**：每一个 wiggle insertion 按它所在 external leg 的 \(D_i\) 阻尼；
2. **覆盖所有 PNG dependence**：导数自动穿过 \(P_i\)、\(1/\mathcal M_i\)、\(1/(\mathcal M_i\mathcal M_j)\)、advection 与 primordial term；
3. **不需要人工为每个 operator 猜阻尼**；
4. **不引入 hard loop**；
5. **不会产生未经证明的 \(P_{\rm w}^2\) 项**；
6. **可逐 native field 回归**。

代码用 first-order dual numbers 精确传播该导数，而不是有限差分，因此生产值不依赖差分步长。

### 5.2 Gaussian 极限

对

\[
G_{ij}=2b_1^2H_{ij}P_iP_j,
\]

上式给出

\[
\boxed{
G_{ij}^{\rm IR}=2b_1^2H_{ij}
\left[
P_{{\rm nw},i}P_{{\rm nw},j}
+D_iP_{{\rm w},i}P_{{\rm nw},j}
+D_jP_{{\rm nw},i}P_{{\rm w},j}
\right].
}
\]

这是 tree bispectrum 的标准 leading-wiggle 结构。

### 5.3 Primordial \(B_{111}\) 项

固定 \(P_\phi\) 后，local-PNG primordial matter bispectrum 可以写成

\[
\widehat B_0
=\mathcal M_1\mathcal M_2\mathcal M_3\widehat B_\phi,
\]

所以 smooth 点的一阶 wiggle 响应是

\[
\boxed{
\widehat B_{0,\rm IR}^{\rm strict}
=\widehat B_{0,\rm nw}
\left[
1+\frac12\sum_{i=1}^3
D_i\frac{P_{{\rm w},i}}{P_{{\rm nw},i}}
\right].
}
\]

这与 primordial non-Gaussian bispectrum 的 TSPT 结果一致：三条 transfer legs 分别接受 BAO damping，而不是把整个 primordial term 乘一个人为统一阻尼。

### 5.4 Stochastic bases

同一公式给出：

- \(b_1^2\sum P_i\)：每条 \(P_{{\rm w},i}\) 乘 \(D_i\)；
- \(b_1b_\phi\sum P_i/\mathcal M_i\)：\(P/\mathcal M=\sqrt{PP_\phi}\)，其 wiggle response 自动获得半幅 transfer dressing；
- \(b_\phi^2\sum P_i/\mathcal M_i^2=b_\phi^2\sum P_{\phi,i}\)：固定 primordial spectrum 后**完全 BAO-free，因此 IR resummation 不改变它**；
- 常数 \(\alpha_4\) basis 不变。

这提供了一个很强的代码单元测试。

---

## 6. `factorized_closure`：用途与限制

定义

\[
P_{{\rm IR},i}=P_{{\rm nw},i}+D_iP_{{\rm w},i},
\]

\[
\mathcal M_{{\rm IR},i}
=\mathcal M_i\sqrt{\frac{P_{{\rm IR},i}}{P_i}}
=\sqrt{\frac{P_{{\rm IR},i}}{P_{\phi,i}}}.
\]

然后直接计算

\[
\boxed{
X_{\rm IR}^{\rm fact}
=X(P_{{\rm IR},i};\mathcal M_{{\rm IR},i}).
}
\]

它的优点：

- 实现简单；
- \(\Sigma^2\to0\) 时恢复完整 fixed-order tree；
- 对 primordial term 给出紧凑的三条 dressed-transfer 形式；
- 与文献中“在 tree correlator 中把 \(P_L\) 替换成 dressed \(P\)”的 practical prescription 一致。

限制：

- 它包含 \(P_{\rm w}^2\) 与由 \(1/\mathcal M\) 平方根展开产生的更高 wiggle terms；
- 这些项并不是标准 TSPT \(O(P_{\rm w})\) 推导的一部分；
- 对 Gaussian \(P_{{\rm w},i}P_{{\rm w},j}\)，近期 ULPT 给出了更精细的 double-wiggle displacement factor；对 MARISA-B 的全部 PNG inverse-transfer operators 还需要额外推导。

因此建议：

- `strict_lo`：fiducial；
- `factorized_closure`：theory-systematics variant；
- 二者差值：估计未控制 higher-wiggle closure 对当前数据向量的影响，但不能等同于完整理论误差。

### 6.1 一个容易误解的极限

当 \(\Sigma^2=0\) 时：

- `factorized_closure` 恢复完整 fixed-order tree；
- `strict_lo` 恢复 \(X_{\rm nw}+X_{1\rm w}\)，而不是包含所有 \(P_{\rm w}^2\) 的 full tree。

这是定义差异，不是 bug。

---

## 7. Reconstruction 的处理

### 7.1 Fixed-\(b_{\rm rec}\) source kernel

MARISA-B 当前 fixed-\(b_{\rm rec}\) post tree 可写成

\[
Z_{1i}=b_1+f_{\rm NL}b_\phi u_i,
\]

\[
Z_{2,ij}^{\rm post}=Z_{2,ij}^{\rm pre}
+\frac12S_{ij}Z_{1i}Z_{1j},
\]

其中

\[
S_{ij}=R_{\mathbf k_i+\mathbf k_j}(\mathbf k_i)
+R_{\mathbf k_i+\mathbf k_j}(\mathbf k_j).
\]

本代码完整复现当前 native 的 fixed-\(b_{\rm rec}\) reconstruction tree contributions，并对其中的 \(P\) 与 \(1/\mathcal M\) 做同样的 strict IR response。

### 7.2 Post-reconstruction damping variance

在 isotropic Gaussian smoothing、CIC-off、固定 \(b_{\rm rec}\) 的 MARISA-B 单场约定下，可先采用 residual displacement

\[
A(q)=1-\frac{b_1}{b_{\rm rec}}e^{-q^2R^2/2}
\]

并计算

\[
\Sigma_{\rm rec}^2=
\frac{1}{6\pi^2}\int^{k_{\rm IR}}dq\,
P_{\rm nw}(q)A^2(q)
[1-j_0(qr_{\rm BAO})+2j_2(qr_{\rm BAO})].
\]

代码提供 `gaussian_reconstruction_residual()`，但要求调用者显式计算并传入 \(\Sigma_{\rm rec}^2\)。这样可避免 source-kernel reconstruction 参数与 IR displacement 假设被暗中耦合。

### 7.3 暂不作为 production-accepted 的情况

1. **Cartesian CIC 开启**：shift window 依赖 \(\mathbf q\) 相对于网格的方向；一个标量 \(\Sigma^2\) 不能完整表达其 displacement covariance。
2. **adaptive denominator**

   \[
   b_{\rm rec}(k;f_{\rm NL}^{\rm rec})
   =b_{\rm rec}+f_{\rm NL}^{\rm rec}
   \frac{b_{\phi,\rm rec}}{\mathcal M(k)}.
   \]

   此时不仅 source tree kernels 依赖 \(f_{\rm NL}^{\rm rec}\)，用于阻尼的 displacement variance 也会产生

   \[
   \partial\Sigma^2/\partial f_{\rm NL}^{\rm rec},
   \qquad
   \partial^2\Sigma^2/\partial(f_{\rm NL}^{\rm rec})^2.
   \]

   若只 IR-resum 当前三个 adaptive source bases，却固定指数中的 \(\Sigma^2\)，不能称为完整 adaptive IR model。

推荐先完成 pre-reconstruction；若当前分析确实使用 post，则第二步只接受 fixed-\(b_{\rm rec}\)、CIC-off。

---

## 8. 为什么 tree-only 不应出现 NLO matching factor

当 tree 与 one-loop 一起使用时，为避免 soft contribution double counting，常见匹配是

\[
P_{\rm tree}^{\rm input}
=P_{\rm nw}+(1+k^2\Sigma^2)e^{-k^2\Sigma^2}P_{\rm w},
\]

而 loop 使用

\[
P_{\rm loop}^{\rm input}
=P_{\rm nw}+e^{-k^2\Sigma^2}P_{\rm w}.
\]

当前项目明确不使用 one-loop，因此没有被 loop 重复包含的 soft expansion，也就不应加入 \(1+k^2\Sigma^2\) 补偿。本文代码故意不暴露 `tree_nlo` 选项，以防止配置层误用。

---

## 9. Python oracle 的实现结构

文件：

```text
python/marisa_b/tree_png_ir_resummation.py
```

### 9.1 公开对象

```python
HaloTreeBias
ReconstructionConfig
TreeIrConfig
TreeComponents
StochasticBases
TreeIrResult
```

### 9.2 公开函数

```python
triangle_vectors(...)
finite_png_halo_tree_operator_terms(...)
finite_png_halo_tree_components(...)
finite_png_stochastic_bases(...)
no_wiggle_transfer(...)
ir_resummed_finite_png_halo_tree(...)
bao_displacement_variance(...)
gaussian_reconstruction_residual(...)
```

### 9.3 逐 native 字段输出

`operator_terms` 包含：

```text
Btree_gaussian_pre
Btree_gaussian_reconstruction

dBdfNL_local_primordial
dBdfNL_local_bphi_f2
dBdfNL_local_bphi_advection
dBdfNL_local_bphidelta
dBdfNL_local_bphi_b2
dBdfNL_local_bphi_bK2
dBdfNL_local_bphi_reconstruction

Bhalo_tree_fNL2_bphi_B0
Bhalo_tree_fNL2_bphi_sq_advection
Bhalo_tree_fNL2_bphi_sq_F2
Bhalo_tree_fNL2_bphi_sq_b2
Bhalo_tree_fNL2_bphi_sq_bK2
Bhalo_tree_fNL2_bphi_sq_reconstruction
Bhalo_tree_fNL2_bphi_bphidelta
Bhalo_tree_fNL2_bphi2_operator

stochastic_alpha3_basis
dBdfNL_stochastic_alpha3_basis
Bhalo_tree_fNL2_stochastic_alpha3PNG_basis
stochastic_alpha4_basis
```

这使测试者能够判断差异究竟来自 primordial、advection、某个 bias operator、reconstruction，还是 stochastic，而不是只比较最终总和。

### 9.4 最小调用示例

```python
from marisa_b.tree_png_ir_resummation import (
    HaloTreeBias,
    TreeIrConfig,
    bao_displacement_variance,
    ir_resummed_finite_png_halo_tree,
)

sigma2 = bao_displacement_variance(
    q,
    p_no_wiggle_q,
    r_bao=110.0,
    k_ir=0.15,
)

bias = HaloTreeBias(
    b1=b1,
    b2=b2_native,
    bk2=bK2,
    bphi=bphi,
    bphidelta=bphidelta,
    bphi2=bphi2,
)

result = ir_resummed_finite_png_halo_tree(
    vectors=k_vectors,            # shape (3,3), closed triangle
    linear_power=p_linear_legs,   # P_L(k1,k2,k3)
    no_wiggle_power=p_nw_legs,    # P_nw(k1,k2,k3)
    transfer_m=m_legs,            # M(k1,k2,k3)
    bias=bias,
    ir=TreeIrConfig(
        sigma2=sigma2,
        scheme="strict_lo",
    ),
)

B = result.deterministic.predict(f_nl)
```

---

## 10. Native C++ 集成计划

Python oracle 通过后，再把同一代数迁入 native。不要先在 production runner 外层“乘一个 damping”。

### 10.1 推荐新增入口

```cpp
ComponentResult
compute_pre_recon_halo_bias_v1_local_png_tree_ir(
    const PowerSpectrum& P_L,
    const PowerSpectrum& P_nw,
    const PowerSpectrum& transfer_m,
    const Triangle& tri,
    const NativeConfig& config,
    const HaloBiasV1Params& bias,
    TreeIrScheme scheme);
```

以及 vector/post 对应版本。

### 10.2 推荐内部架构

把当前 tree algebra 抽成 scalar-templated evaluator：

```cpp
template <class Scalar>
TreeAtomicTerms evaluate_tree_terms(
    const std::array<Vec3,3>& k,
    const std::array<Scalar,3>& P,
    const std::array<Scalar,3>& M,
    ...);
```

再实现一个只带 `value` 与 `tangent` 的 `Dual1`：

```cpp
struct Dual1 {
    double value;
    double tangent;
};
```

每个 triangle 做：

1. 一次 smooth evaluation；
2. 三次 single-leg dual evaluation；
3. 用 \(D_i\) 合并。

这样能与 Python oracle 完全同构，并避免手工维护数十条 derivative formula。

### 10.3 为什么不建议通过全局 `PowerSpectrum` 包装器做三次替换

`strict_lo` 需要“只打开第 \(i\) 条 external leg 的 wiggle”。一个只依赖 \(k\) 的全局 spectrum wrapper 在等边或等 \(k\) legs 时无法区分 leg identity，因此不能可靠表达独立 \(\epsilon_i\)。正确位置是 triangle-level algebra evaluator。

### 10.4 `NativeConfig.ir_sigma2`

仓库已有 `ir_sigma2` 字段，可以复用数值存储，但必须补充明确语义：

- 它是当前 redshift 与当前 reconstruction 状态下的 \(\Sigma^2\)；
- tree-only 模式只使用 \(e^{-k^2\Sigma^2}\)；
- 不得同时调用旧 Gaussian IR tree 路径后再对 finite-PNG result 重复阻尼。

### 10.5 Python binding

建议返回完整 `ComponentResult`，保留所有 operator fields。上层 `FinitePngComponents` 继续组装

\[
G+f_{\rm NL}L+f_{\rm NL}^2Q.
\]

IR scheme、\(P_{\rm nw}\) identity、\(\Sigma^2\)、\(k_{\rm IR}\) 与 \(r_{\rm BAO}\) 应写入 cache/provenance key，防止旧 cache 与新模型混用。

---

## 11. Shell projection 与有限盒子顺序

正确顺序是：

1. 构造一个实际闭合的 triangle vector triplet；
2. 在三条腿上分别读取 \(P_L(k_i)\)、\(P_{\rm nw}(k_i)\)、\(\mathcal M(k_i)\)；
3. 对该实际 triangle 计算 IR-resummed atomic terms；
4. 再进入 continuum shell quadrature、FullDiscrete 或 exact-lattice averaging；
5. 保持原有 shell-pair denominator/zero-mode contract。

不能先对 fixed-order bispectrum 做 bin average，再给 bin center 乘 damping；非线性的 \(D_i\)、triangle geometry 和 post/CIC orientation 会导致两者不交换。

---

## 12. 验证与验收矩阵

测试文件：

```text
tests/python/test_tree_png_ir_resummation.py
```

### 12.1 代数回归

1. `finite_png_halo_tree_components` 与当前 native pre \(G,L,Q\) 逐项一致；
2. post-pre 差值与 native reconstruction 解析项一致；
3. atomic fields 的总和严格等于聚合 \(G,L,Q\)；
4. stochastic bases 与 native 字段定义一致；
5. \(b_\phi^2\) 与 \(b_{\phi^2}\) 始终分开。

推荐 native/Python tolerance：

- 正常幅值项：`rtol <= 1e-12`；
- 大数相消后的接近零项：采用按组成项尺度定义的绝对 tolerance，不用对零强行做 relative comparison。

### 12.2 IR 身份测试

1. \(P_{\rm nw}/\mathcal M_{\rm nw}^2=P_L/\mathcal M^2\)；
2. \(P_{\rm w}=0\) 时两种 scheme 都等于 smooth fixed tree；
3. `factorized_closure` 在 \(\Sigma^2=0\) 时恢复 full fixed tree；
4. `strict_lo` 与独立 central finite difference 的 legwise first derivative 一致；
5. primordial term 满足三条 dressed-transfer 解析式；
6. quadratic PNG stochastic basis 因为等于 \(P_\phi\) 而保持不变；
7. `factorized - strict` 在小 wiggle amplitude 下按 \(P_{\rm w}^2\) 缩放。

### 12.3 对称性与几何

至少覆盖：

- equilateral；
- isosceles；
- squeezed；
- flattened但不奇异；
- 三条 leg 的全部 permutation；
- 随机旋转（CIC-off）；
- CIC-on 时只检查与 native 相同的方向依赖，不要求旋转不变。

### 12.4 Reconstruction 极限

- smoothing radius \(R\to\infty\)：post source kernel 回到 pre；
- residual filter \(A(q)\to1\)：\(\Sigma_{\rm rec}^2\to\Sigma^2\)；
- 当 \(b_1=b_{\rm rec}\)、低 \(q\) 且 smoothing window 约为 1 时，reconstruction 应减少 \(\Sigma^2\)。

### 12.5 科学影响测试

在实际 triangle/bins/covariance 上比较：

\[
\Delta G=G_{\rm IR}-G_{\rm fixed},\quad
\Delta L=L_{\rm IR}-L_{\rm fixed},\quad
\Delta Q=Q_{\rm IR}-Q_{\rm fixed}.
\]

至少报告：

- 每个 bin 的 \(\Delta X/\sigma_B\)；
- 对 \(f_{\rm NL}=0,\pm100\) 的总模型变化；
- `strict_lo` 与 `factorized_closure` 的差值；
- \(k_{\rm IR}\) variation；
- 开/关 BAO bins 对 \(f_{\rm NL}\) posterior、bias parameters 与 goodness-of-fit 的变化；
- mock injection 中是否产生可见的 \(f_{\rm NL}\) 偏移。

---

## 13. 分阶段实施与合并门槛

### 阶段 A：Python oracle（本次交付）

- 固定当前 native conventions；
- 实现 pre、fixed-\(b_{\rm rec}\) post、stochastic bases；
- 实现 `strict_lo` 与 `factorized_closure`；
- 不接 production likelihood。

**门槛**：独立测试全部通过；测试者确认公式与当前 main 的 native fields 一致。

### 阶段 B：Native triangle evaluator

- scalar-template/dual 实现；
- 逐 triangle、逐 field 与 Python oracle 比较；
- 添加 C++ unit tests；
- 不修改 frozen production result。

**门槛**：shape grid 上全部 atomic components 满足容差。

### 阶段 C：Projection integration

- 在 shell/quadrature 内部逐 triangle IR-resum；
- cache key 增加 IR provenance；
- exact-lattice/continuum 两种投影分别回归。

**门槛**：固定输入下确定性复现；没有旧 cache 污染；projection denominator 不变。

### 阶段 D：Pre-reconstruction scientific validation

- mock data 上比较 no-IR、strict、factorized；
- 扫描 \(k_{\rm IR}\)；
- 检查 \(f_{\rm NL}\) recovery 与 nuisance shifts。

**门槛**：确定 fiducial scheme 与允许的 \(k_{\max}\)，并更新 scientific-status 文档。

### 阶段 E：Post fixed-\(b_{\rm rec}\)、CIC-off

- 使用 \(\Sigma_{\rm rec}^2\)；
- 验证 post source kernels 与 reconstructed damping；
- 只在已有 post catalogue 科学状态允许的范围内使用。

### 阶段 F：CIC/adaptive reconstruction

需要 tensor displacement covariance 和 \(f_{\rm NL}^{\rm rec}\)-dependent exponent expansion。它应是独立项目，不能以固定标量 \(\Sigma^2\) 的补丁冒充完成。

---

## 14. 推荐的 production 选择

在完成验证前，推荐配置是：

```text
perturbative_order       = tree_only
ir_scheme                = strict_lo
power_split              = one frozen P_nw/P_w artifact
sigma_spectrum           = P_nw at the analysis redshift
reconstruction           = pre  (first acceptance target)
post_reconstruction      = fixed-b_rec, CIC-off only after pre acceptance
higher_wiggle_variant    = factorized_closure (systematics only)
```

并明确禁止：

```text
tree_nlo_matching        = off
one_loop                  = off
reuse_old_gaussian_ir_then_redamp_png = forbidden
adaptive_brec_ir          = not yet accepted
scalar_ir_with_cartesian_cic = not accepted
```

---

## 15. 运行方法

应用文件后：

```bash
python -m pip install -e '.[validate]'
python -m pytest tests/python/test_tree_png_ir_resummation.py -q
```

不安装 editable package 时：

```bash
PYTHONPATH=python \
python -m pytest tests/python/test_tree_png_ir_resummation.py -q
```

另一个模型在测试时应同时做两件事：

1. 运行这里的 standalone tests；
2. 从当前 native function 随机抽取 triangles、biases、pre/post configs，把每个 atomic field 与 Python oracle 做直接数值对照。

只运行本文件自带测试并不足以证明与未来变化后的仓库 main 一致。

---

## 16. 主要参考依据

1. D. Blas, M. Garny, M. M. Ivanov, S. Sibiryakov, *Time-Sliced Perturbation Theory II: Baryon Acoustic Oscillations and Infrared Resummation*, arXiv:1605.02149。
2. M. M. Ivanov, S. Sibiryakov, *Infrared Resummation for Biased Tracers in Redshift Space*, arXiv:1804.05080。特别是 bias vertices IR-safe，以及 LO tree 使用 dressed linear spectrum 的结论。
3. A. Vasudevan, M. M. Ivanov, S. Sibiryakov, J. Lesgourgues, *Time-sliced perturbation theory with primordial non-Gaussianity...*, arXiv:1906.08697。特别是 primordial non-Gaussian bispectrum 的三条 dressed-transfer 表达。
4. N. S. Sugiyama, *IR-Safe and IR-Resummed Bispectra Before and After Reconstruction in Unified Lagrangian Perturbation Theory*, arXiv:2602.20431。用于区分严格 \(O(P_{\rm w})\) TSPT、practical substitution 与 double-wiggle ULPT correction。
5. MARISA-B 当前 `src/marisa_b/marisa_b_native.{h,cpp}`、`python/marisa_b/model.py` 与 scientific-status 文档，作为代码字段和生产边界的事实来源。

---

## 17. 最终建议

现在最合理的下一步不是在现有总 bispectrum 外面乘一个经验 damping，也不是启用仓库已有的一环 matching。应当：

1. 以本 Python oracle 固定 tree-only 理论合同；
2. 先接受 pre-reconstruction `strict_lo`；
3. 逐 atomic field 对照 native；
4. 在 projection 之前逐 triangle 应用；
5. 用 `factorized_closure` 与 \(k_{\rm IR}\) variation 做系统学；
6. 只有在 mock inference 证明稳定后才进入 production；
7. CIC 与 adaptive reconstruction 单独开展完整 displacement-covariance 推导。

这样得到的结果才既是“覆盖当前完整 PNG halo tree operator basis”的 IR resummation，又不会在 tree-only 项目中无意混入 one-loop matching 或未经标注的高阶 closure。
