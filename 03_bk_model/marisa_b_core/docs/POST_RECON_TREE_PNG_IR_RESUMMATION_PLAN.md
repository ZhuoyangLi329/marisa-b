# MARISA-B：post-reconstruction tree-level finite-local-PNG halo bispectrum 的完整 IR resummation 设计、推导与实施计划

**版本日期：2026-09-30**
**适用范围：实空间；tree level；有限 local PNG，保留到 \(f_{\rm NL}^2\)；不使用任何 one-loop 或 tree–loop matching。**

---

## 0. 执行摘要

MARISA-B 目前的 post-reconstruction tree-level finite-PNG 模型已经包含了**重建对 source kernel 的改变**，但要形成一个内部闭合的 post-recon IR-resummed 模型，还必须同时处理**重建后长波位移场的 BAO 阻尼指数**。

这两件事不能混为一谈：

1. **source reconstruction** 改变 \(Z_{2,\rm rec}\) 以及 finite-PNG 的 \(G,L,Q\) source coefficients；
2. **displacement resummation** 把 reconstructed long displacement 的方差非微扰地指数化，产生 \(e^{-k^2\Sigma_{\rm rec}^2}\)。

对于 MARISA-B 当前的 finite-PNG reconstruction 语义，第二部分还有一个容易漏掉的事实：重建位移由 halo density 构造，因此

\[
Z_1(q;f)=b_1+f\frac{b_\phi}{\mathcal M(q)}
\]

本身依赖 \(f\equiv f_{\rm NL}\)。若使用 adaptive reconstruction denominator，

\[
b_{\rm rec}(q;f_{\rm NL}^{\rm rec})
=b_{\rm rec}+f_{\rm NL}^{\rm rec}
 \frac{b_{\phi,\rm rec}}{\mathcal M(q)}.
\]

所以 reconstructed BAO displacement variance 也有

\[
\Sigma_{\rm rec}^2(f)
=\Sigma_0^2+f\Sigma_1^2+f^2\Sigma_2^2+\cdots .
\]

于是完整的 tree-level finite-PNG post-recon IR 模型不仅要阻尼 source 的 \(G,L,Q\)，还必须加入

\[
\frac{\partial D}{\partial f}\,G_{\rm w},
\qquad
\frac{\partial D}{\partial f}\,L_{\rm w},
\qquad
\frac12\frac{\partial^2D}{\partial f^2}\,G_{\rm w}
\]

产生的有限-PNG mixing。这里 \(D=e^{-k^2\Sigma_{\rm rec}^2(f)}\)。

本次交付的 Python oracle 实现了：

- MARISA-B 当前 fixed-\(b_{\rm rec}\) post source；
- 当前 native adaptive-denominator 的三个 tree bases；
- reconstructed displacement variance 的 \(f^0,f^1,f^2\) 系数；
- strict leading-wiggle \(O(P_{\rm w})\) IR resummation；
- fixed denominator、matched/mismatched adaptive denominator；
- 无 CIC 时的各向同性一维积分；
- Cartesian CIC 打开时的逐外腿、逐方向角积分；
- 可选 reconstruction shift-noise 对阻尼尺度的贡献；
- stochastic source 与 damping-induced stochastic mixing 的显式分离。

当前参考实现**不包含**：

- Gaussian 或 PNG one-loop；
- \((1+k^2\Sigma^2)e^{-k^2\Sigma^2}\) 的 NLO matching；
- \(P_{\rm w}^2\) 的 ULPT wiggle–wiggle closure；
- RSD；
- primordial non-Gaussian long-displacement higher cumulants；
- survey mask/window 的真实卷积。

这里的“完整”是指：**在 MARISA-B 当前 tree-level、\(G+fL+f^2Q\)、strict \(O(P_{\rm w})\)、linear-Gaussian long-displacement 计数下，source 与 displacement exponent 同时闭合到 \(f^2\)。**

---

## 1. 仓库中现有的 post-reconstruction 路径

当前 `src/marisa_b/marisa_b_native.{h,cpp}` 中与本任务直接相关的入口可以分为以下几组。

### 1.1 Gaussian matter / halo post kernels

```cpp
compute_post_recon_gaussian(...)
compute_post_recon_halo_bias_v1_gaussian(...)
```

其中 halo 版本把同一套 pre-recon halo kernels \(K_n\) 通过 set-partition 公式映射成 reconstructed kernels \(Z_{{\rm rec},n}\)。

核心内部函数是：

```cpp
shift_factor_post(...)
halo_bias_v1_zrec_kernel_vec(...)
```

当前 reconstruction shift 使用：

- Gaussian smoothing radius `smoothing_radius`；
- denominator `bias_recon`；
- 可选 `recon_cellsize`；
- 可选 `recon_cic_window_power`。

### 1.2 现有 Gaussian post IR 路径

```cpp
compute_post_recon_gaussian_ir_nowiggle(...)
```

它使用 `NativeConfig.ir_sigma2`，内部定义半阻尼

\[
d_i=\exp\!\left[-\frac12 k_i^2\,{\tt ir\_sigma2}\right],
\]

而 one-wiggle 项使用 \(d_i^2\)。因此在本文件的记号中，

\[
\boxed{{\tt ir\_sigma2}=\Sigma_{\rm BAO,rec}^2}
\]

是直接对应关系。

但该入口是已有 Gaussian/legacy 路径，并没有自动构造 finite-PNG halo \(G,L,Q\) 的 reconstructed displacement response。

### 1.3 Fixed-denominator finite-PNG halo tree

```cpp
compute_post_recon_halo_bias_v1_local_png_tree_dfNL(...)
compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(...)
```

其中 vector 版本保留三角形相对于 Cartesian mesh 的绝对方向，因此是 CIC-aware shell projection 的正确入口。

这个路径已经包含：

- primordial \(B_{111}\)；
- \(b_\phi F_2\)；
- PNG advection；
- \(b_{\phi\delta}\)；
- \(b_\phi b_2\)、\(b_\phi b_{K^2}\)；
- fixed-\(b_{\rm rec}\) reconstruction source；
- \(f_{\rm NL}^2\) 的全部当前 deterministic tree terms；
- 当前 stochastic raw bases。

但这些是 reconstructed **source** terms；它们并不等于完整的 post-recon IR-resummed result。

### 1.4 Adaptive reconstruction denominator

当前仓库还提供：

```cpp
compute_post_recon_halo_local_png_brec_denominator_tree_basis_vectors(...)
compute_post_recon_halo_local_png_brec_finite_tree_bases_vectors(...)
```

其定义是

\[
b_{\rm rec}(k;f_{\rm NL}^{\rm rec})
=b_{\rm rec}
+f_{\rm NL}^{\rm rec}\frac{b_{\phi,\rm rec}}{\mathcal M(k)}.
\]

native 返回三种未乘 bias prefactor 的 bases：

\[
\begin{aligned}
G_1 &= b_1^4\frac{b_{\phi,\rm rec}}{b_{\rm rec}}
      \,{\tt gaussian\_linear},\\
G_2 &= b_1^4\left(\frac{b_{\phi,\rm rec}}{b_{\rm rec}}\right)^2
      \,{\tt gaussian\_quadratic},\\
Q_\times &= b_1^3b_\phi\frac{b_{\phi,\rm rec}}{b_{\rm rec}}
      \,{\tt png\_linear\_cross}.
\end{aligned}
\]

其中 \(G_2\) 是 \((f_{\rm NL}^{\rm rec})^2\) 的系数，\(Q_\times\) 乘
\(f_{\rm NL}f_{\rm NL}^{\rm rec}\)。

### 1.5 本项目暂不使用的 post paths

仓库还存在 post one-loop 和 matter-PNG uplift 入口。它们不属于本次实现范围，也不应该影响本文的公式或 API。代码中没有引入任何 tree–loop matching。

---

## 2. 为什么 post-recon 不能直接沿用 pre-recon 的 \(\Sigma^2\)

在 ULPT/重映射表述中，post-reconstructed density 保留和 pre-recon 相同的数学结构，但 displacement field 被替换成 reconstructed displacement field。2026 年 ULPT bispectrum 工作明确指出：post-recon IR-resummed bispectrum 的解析结构与 pre-recon 相同，区别在于 reconstructed displacement covariance 和 reconstructed nonlinear source kernels。

对无偏 matter 且 shift 完全匹配的理想情况，线性 reconstructed displacement 是

\[
\boldsymbol\Psi_{\rm rec}^{(1)}(\boldsymbol q)
=i\int\frac{d^3p}{(2\pi)^3}
 e^{i\boldsymbol p\cdot\boldsymbol q}
 \frac{\boldsymbol p}{p^2}
 [1-W(pR)]\delta^{(1)}(\boldsymbol p).
\]

所以 post-recon BAO variance 包含 \([1-W]^2\)，而不是 pre-recon 的 1。

MARISA-B 的情况更一般：

- shift 由 halo density 估计；
- 可以有 \(b_1/b_{\rm rec}\neq1\)；
- halo linear response 包含 \(f_{\rm NL}b_\phi/\mathcal M\)；
- denominator 可以 adaptive；
- mesh paint/read 可以带 Cartesian CIC。

因此 residual displacement filter 必须从项目的 reconstruction 定义重新推导，而不能简单写成 \(1-W\)。

---

## 3. MARISA-B reconstructed linear displacement

记

\[
f\equiv f_{\rm NL},
\qquad
f_{\rm NL}^{\rm rec}=\eta f.
\]

这里 \(\eta\) 用于统一表示不同 reconstruction catalog/algorithm：

- \(\eta=0\)：fixed denominator；
- \(\eta=1\)：reconstruction 使用与物理模型相同的 \(f_{\rm NL}\)；
- 其他值：mismatched/adaptive stress test。

halo linear kernel 为

\[
Z_1(q;f)=b_1+f\frac{b_\phi}{\mathcal M(q)}.
\]

adaptive denominator 为

\[
b_{\rm rec}(q;\eta f)
=b_{\rm rec}
+\eta f\frac{b_{\phi,\rm rec}}{\mathcal M(q)}.
\]

reconstruction window 写成

\[
W_{\rm rec}(\boldsymbol q)
=\exp\!\left[-\frac12q^2R^2\right]
 W_{\rm mesh}(\boldsymbol q).
\]

MARISA-B 的 Cartesian CIC convention 对应

\[
W_{\rm mesh}(\boldsymbol q)
=\left[
 \prod_{a=x,y,z}
 \operatorname{sinc}\!\left(\frac{q_a\Delta}{2}\right)
 \right]^{p_{\rm CIC}},
\]

其中 Python `np.sinc(x)` 的输入换成 \(q_a\Delta/(2\pi)\)。

重建后的长波位移相对于原 matter displacement 的系数为

\[
\boxed{
C(\boldsymbol q;f)
=1-W_{\rm rec}(\boldsymbol q)
\frac{b_1+f b_\phi/\mathcal M(q)}
     {b_{\rm rec}+\eta f b_{\phi,\rm rec}/\mathcal M(q)}.
}
\]

这条式子是 post-recon finite-PNG damping 的核心。

---

## 4. \(C^2\) 到 \(f^2\) 的解析展开

定义

\[
a=\frac{b_1}{b_{\rm rec}},
\qquad
n(q)=\frac{b_\phi}{b_{\rm rec}\mathcal M_{\rm nw}(q)},
\qquad
d(q)=\frac{b_{\phi,\rm rec}}
{b_{\rm rec}\mathcal M_{\rm nw}(q)}.
\]

所有 IR covariance integrals 使用 no-wiggle quantities，且保持

\[
P_\phi(q)=\frac{P_L(q)}{\mathcal M^2(q)}
\]

不变。因此

\[
\mathcal M_{\rm nw}(q)
=\mathcal M(q)\sqrt{\frac{P_{\rm nw}(q)}{P_L(q)}}.
\]

将

\[
C=r_0+fr_1+f^2r_2+O(f^3)
\]

展开，得到

\[
\begin{aligned}
r_0 &=1-Wa,\\
r_1 &=-W\,[n-\eta a d],\\
r_2 &=W\,[\eta nd-\eta^2ad^2].
\end{aligned}
\]

于是

\[
C^2=c_0+fc_1+f^2c_2+O(f^3)
\]

其中

\[
\boxed{
\begin{aligned}
c_0 &=r_0^2,\\
c_1 &=2r_0r_1,\\
c_2 &=r_1^2+2r_0r_2.
\end{aligned}
}
\]

### 4.1 fixed denominator

当 \(\eta=0\) 时，

\[
r_1=-Wn,
\qquad r_2=0,
\]

但

\[
c_2=W^2n^2\neq0.
\]

因此即使 denominator 完全固定，post-recon damping exponent 仍然会生成 \(f\) 和 \(f^2\) 响应。这是“只给已有 \(G,L,Q\) 乘固定 Gaussian damping”会遗漏的主要项。

### 4.2 matched adaptive cancellation

若

\[
b_{\rm rec}=b_1,
\qquad
b_{\phi,\rm rec}=b_\phi,
\qquad
\eta=1,
\]

则

\[
\frac{b_1+fb_\phi/\mathcal M}
{b_{\rm rec}+fb_{\phi,\rm rec}/\mathcal M}=1
\]

对任意 \(f\) 都成立。因此

\[
C=1-W,
\qquad
c_1=c_2=0.
\]

这提供了非常强的解析回归测试：matched adaptive reconstruction 必须使 finite-PNG damping response 精确消失，但 source 中的 physical/adaptive PNG terms 仍然存在。

---

## 5. 无 CIC 时的 reconstructed BAO displacement variance

无 Cartesian mesh dependence 时，三个外腿共享同一个标量 variance。定义

\[
\mathcal W_{\rm BAO}(q)
=1-j_0(qr_{\rm BAO})+2j_2(qr_{\rm BAO}).
\]

则

\[
\boxed{
\Sigma_m^2
=\frac{1}{6\pi^2}
 \int_{q_{\min}}^{\Lambda_{\rm IR}}dq\,
 P_{\rm nw}(q)c_m(q)\,
 \mathcal W_{\rm BAO}(q),
\qquad m=0,1,2.
}
\]

这里

\[
\Sigma_{\rm rec}^2(f)
=\Sigma_0^2+f\Sigma_1^2+f^2\Sigma_2^2.
\]

注意上标“2”表示位移方差，不表示系数本身必须为正；\(\Sigma_1^2\) 和 \(\Sigma_2^2\) 可以有符号。

### 5.1 为什么必须显式设置 \(q_{\min}>0\)

local PNG 在大尺度上

\[
\frac{1}{\mathcal M(q)}\propto q^{-2}.
\]

特别是 fixed denominator 的 \(c_2\) 包含

\[
\frac{b_\phi^2}{\mathcal M^2(q)}.
\]

虽然 BAO window 在 \(q\to0\) 时提供 \(q^2r_{\rm BAO}^2\) 抑制，\(f^2\) variance 仍可对 finite-box/survey cutoff 呈对数级敏感。因此：

1. 不能把积分下限写成数学上的 0；
2. `q_min` 应与 mock box fundamental mode 或 survey-window IR prescription 对齐；
3. 最好与 MARISA-B 已有 `png_ir_cutoff` 的分析 provenance 联合登记；
4. 必须对 \(q_{\min}\) 做稳定性扫描。

---

## 6. Cartesian CIC 时的方向依赖 variance

当 `recon_cellsize>0` 且 `recon_cic_window_power>0` 时，\(W_{\rm mesh}\) 依赖 \(q_x,q_y,q_z\)，不再只依赖 \(|q|\)。这时不能继续用一个方向无关的 \(\Sigma^2\)。

对第 \(i\) 条外腿，strict one-wiggle saddle approximation 下使用

\[
\boxed{
\Sigma_{i,m}^2
=\int_{q_{\min}}^{\Lambda_{\rm IR}}
 \frac{dq}{(2\pi)^3}P_{\rm nw}(q)
 \int d\Omega_{\hat q}\,
 (\hat k_i\!\cdot\!\hat q)^2
 \left[1-\cos\left(qr_{\rm BAO}
 \hat k_i\!\cdot\!\hat q\right)\right]
 c_m(q,\hat q).
}
\]

无 CIC 时角积分严格退化成上一节的

\[
\frac{1}{6\pi^2}
[1-j_0+2j_2].
\]

参考代码使用：

- \(\mu_q\)：Gauss–Legendre quadrature；
- \(\phi_q\)：周期 trapezoidal quadrature；
- 每一个实际 Cartesian triangle 分别计算三条腿；
- 先 IR resum，再 shell/orientation project。

因此，正确顺序是

```text
Cartesian triangle vectors
 -> source kernel with CIC
 -> per-leg CIC-aware Sigma_i,m^2
 -> finite-PNG damping mixing
 -> shell/orientation average
```

而不是

```text
先平均成 B000(k1,k2)
 -> 再乘一个标量 damping
```

后者会丢失 mesh orientation dependence。

---

## 7. Optional reconstruction shift noise

若 reconstruction shift 由有限数密度 halo catalogue 估计，位移估计包含 stochastic/noise field。若其功率记为 \(P_\epsilon(q)\)，且与 matter long mode 不相关，则 variance 中额外加入

\[
P_\epsilon(q)
\frac{W_{\rm rec}^2(\boldsymbol q)}
{[b_{\rm rec}+\eta f b_{\phi,\rm rec}/\mathcal M(q)]^2}.
\]

展开到 \(f^2\)：

\[
\begin{aligned}
e_0 &=\frac{W^2}{b_{\rm rec}^2},\\
e_1 &=-2\eta d\,e_0,\\
e_2 &=3\eta^2d^2\,e_0.
\end{aligned}
\]

这部分默认关闭，原因是：

- 当前 deterministic tree 模型没有注册统一的 reconstruction-noise closure；
- \(P_\epsilon\) 可能不是简单 \(1/\bar n\)，还受 mass weighting、assignment、deconvolution 和 catalogue construction 影响；
- 不能把它悄悄固定成普通 stochastic alpha 参数。

代码允许显式传入 array 或 callable，以便 mock/calibration 后启用。

---

## 8. Damping exponent 到 \(f^2\) 的展开

对第 \(i\) 条外腿，

\[
D_i(f)
=\exp\left[-k_i^2
 (\Sigma_{i,0}^2+f\Sigma_{i,1}^2+f^2\Sigma_{i,2}^2)
 \right].
\]

定义

\[
D_{i,0}=e^{-k_i^2\Sigma_{i,0}^2},
\]

则

\[
D_i(f)=D_{i,0}
[1+fd_{i,1}+f^2d_{i,2}]+O(f^3)
\]

其中

\[
\boxed{
\begin{aligned}
d_{i,1}&=-k_i^2\Sigma_{i,1}^2,\\
d_{i,2}&=\frac12k_i^4(\Sigma_{i,1}^2)^2
-k_i^2\Sigma_{i,2}^2.
\end{aligned}
}
\]

即使 \(\Sigma_{i,2}^2=0\)，平方指数展开仍会通过
\((\Sigma_{i,1}^2)^2\) 生成 \(f^2\) contribution。

---

## 9. Post source：fixed denominator 与 adaptive denominator

### 9.1 Fixed source

已有 `tree_png_ir_resummation.py` 中的 post source evaluator 对当前 native 约定给出

\[
B_{{\rm src},\rm fixed}(f)
=G_{\rm fixed}+fL_{\rm fixed}+f^2Q_{\rm fixed}.
\]

它已经包含 current native 的 reconstruction source terms。

### 9.2 Adaptive denominator bases

对循环 pair \((i,j)\)，定义 native shift factors \(s_i,s_j\)，

\[
U_{ij}=\frac1{\mathcal M_i}+\frac1{\mathcal M_j},
\]

\[
S_{ij}^{M}=\frac{s_i}{\mathcal M_i}
+\frac{s_j}{\mathcal M_j}.
\]

native bases 是

\[
\begin{aligned}
{\cal B}_{G_1}
&=-\sum_{\rm cyc}P_iP_jS_{ij}^{M},\\
{\cal B}_{G_2}
&=\sum_{\rm cyc}P_iP_j
\left(\frac{s_i}{\mathcal M_i^2}
     +\frac{s_j}{\mathcal M_j^2}\right),\\
{\cal B}_{\times}
&=-2\sum_{\rm cyc}P_iP_jU_{ij}S_{ij}^{M}.
\end{aligned}
\]

若 \(f_{\rm NL}^{\rm rec}=\eta f\)，则 source coefficients 是

\[
\boxed{
\begin{aligned}
L_{\rm src}
&=L_{\rm fixed}
+\eta b_1^4\frac{b_{\phi,\rm rec}}{b_{\rm rec}}
 {\cal B}_{G_1},\\
Q_{\rm src}
&=Q_{\rm fixed}
+\eta^2b_1^4
 \left(\frac{b_{\phi,\rm rec}}{b_{\rm rec}}\right)^2
 {\cal B}_{G_2}\\
&\quad
+\eta b_1^3b_\phi
 \frac{b_{\phi,\rm rec}}{b_{\rm rec}}
 {\cal B}_{\times}.
\end{aligned}
}
\]

这些 adaptive source terms 和 adaptive damping response 是两个不同的贡献；两者都必须加入。

---

## 10. Strict leading-wiggle source decomposition

继续使用

\[
P_L=P_{\rm nw}+P_{\rm w}.
\]

对每一条外腿引入独立参数 \(\epsilon_i\)：

\[
P_i(\epsilon_i)
=P_{{\rm nw},i}+\epsilon_iP_{{\rm w},i},
\]

\[
\mathcal M_i(\epsilon_i)
=\mathcal M_{{\rm nw},i}
\sqrt{1+\epsilon_i
\frac{P_{{\rm w},i}}{P_{{\rm nw},i}}}.
\]

这样逐腿变化时

\[
\frac{P_i(\epsilon_i)}{\mathcal M_i^2(\epsilon_i)}
=P_{\phi,i}
\]

严格保持不变。

对 post source 的 \(X\in\{G,L,Q\}\)，定义

\[
X_{\rm nw}=X[P_{\rm nw},\mathcal M_{\rm nw}],
\]

\[
X_{{\rm w},i}
=\left.\frac{\partial X}{\partial\epsilon_i}
\right|_{\epsilon=0}.
\]

参考实现使用 first-order dual numbers 精确传播该导数，不使用 finite-difference step。

---

## 11. 完整 post-recon finite-PNG IR 公式

将 source 与 damping exponent 同时展开，得到本项目最重要的公式：

\[
\boxed{
G_{\rm IR}
=G_{\rm nw}
+\sum_{i=1}^3D_{i,0}G_{{\rm w},i}.
}
\]

\[
\boxed{
L_{\rm IR}
=L_{\rm nw}
+\sum_{i=1}^3D_{i,0}
\left[L_{{\rm w},i}
+d_{i,1}G_{{\rm w},i}\right].
}
\]

\[
\boxed{
Q_{\rm IR}
=Q_{\rm nw}
+\sum_{i=1}^3D_{i,0}
\left[Q_{{\rm w},i}
+d_{i,1}L_{{\rm w},i}
+d_{i,2}G_{{\rm w},i}\right].
}
\]

因此

\[
B_{h,\rm post}^{\rm IR}(f)
=G_{\rm IR}+fL_{\rm IR}+f^2Q_{\rm IR}.
\]

### 11.1 新增的 mixing 项

相对于“固定 Gaussian damping 乘 source”的模型，新增：

\[
\Delta L_D
=\sum_iD_{i,0}d_{i,1}G_{{\rm w},i},
\]

\[
\Delta Q_{D,L}
=\sum_iD_{i,0}d_{i,1}L_{{\rm w},i},
\]

\[
\Delta Q_{D,G}
=\sum_iD_{i,0}d_{i,2}G_{{\rm w},i}.
\]

代码把这三项单独返回，便于判断它们是否数值可忽略；不能在推导阶段先假定为零。

### 11.2 为什么这里不加入 \(P_{\rm w}^2\)

本次 fiducial 是严格 \(O(P_{\rm w})\)。因此没有加入

\[
P_{{\rm w},i}P_{{\rm w},j}
\]

的 wiggle–wiggle contribution。2026 ULPT 推导给出了 post-recon wiggle–wiggle 的额外 displacement operator；直接使用两个 dressed spectra 相乘只是一种 closure，不属于当前 strict model。

如果以后需要 \(P_{\rm w}^2\)，应另建一个明确命名的
`ulpt_wiggle_wiggle_closure`，而不是改变本模块 `strict` 的字段语义。

---

## 12. Primordial term 和 PNG bias operators 的一致处理

上述逐腿变化同时作用在：

- \(P_iP_j\)；
- \(1/\mathcal M_i\)；
- primordial \(\widehat B_0\) 的三条 transfer legs；
- \(b_\phi F_2\)；
- advection；
- \(b_{\phi\delta}\)；
- \(b_\phi^2\)；
- 独立 \(b_{\phi^2}\)；
- fixed reconstruction source；
- adaptive denominator source bases。

因此不能只替换 Gaussian \(P_iP_j\)，也不能只对 primordial \(B_{111}\) 做单独阻尼。

---

## 13. Stochastic 项的处理

当前 native stochastic raw bases 包括：

\[
\begin{aligned}
S_G&=b_1^2\sum_iP_i,\\
S_L&=b_1b_\phi\sum_i\frac{P_i}{\mathcal M_i},\\
S_Q&=b_\phi^2\sum_i\frac{P_i}{\mathcal M_i^2},\\
S_{\alpha4}&=1.
\end{aligned}
\]

strict source dressing可以逐腿执行。但一旦 damping exponent 随 \(f\) 变化，Gaussian stochastic wiggle response 也会混入 linear/quadratic PNG orders。

是否将这些项并入同一个 alpha3 amplitude，取决于 likelihood 的 stochastic closure。参考代码因此返回：

- `source_stochastic`；
- `stochastic_mixing`；
- 一个必须显式调用的 `stochastic_if_common_alpha3()`。

默认不自动闭合，避免改变现有 nuisance parameter 语义。

---

## 14. Python reference implementation

### 14.1 文件

```text
python/marisa_b/tree_png_ir_resummation.py
python/marisa_b/post_recon_tree_png_ir.py
tests/python/test_post_recon_tree_png_ir.py
```

第二个文件依赖第一个文件的 native-compatible tree source oracle。

### 14.2 主要数据结构

```python
AdaptiveReconstructionConfig
PostReconIrSettings
DisplacementVarianceExpansion
LegDampingExpansion
DampingMixing
StochasticDampingMixing
PostReconTreeIrResult
```

### 14.3 核心 API

```python
post_recon_displacement_variance_expansion(...)
post_recon_displacement_variance_at_fnl(...)
damping_expansion_from_variance(...)
post_recon_source_operator_terms(...)
ir_resummed_post_recon_finite_png_tree(...)
```

### 14.4 最小 fixed-denominator 示例

```python
from marisa_b.tree_png_ir_resummation import (
    HaloTreeBias,
    ReconstructionConfig,
)
from marisa_b.post_recon_tree_png_ir import (
    PostReconIrSettings,
    ir_resummed_post_recon_finite_png_tree,
)

result = ir_resummed_post_recon_finite_png_tree(
    vectors=k_vectors,                 # shape (3,3), closed Cartesian triangle
    linear_power=P_external,
    no_wiggle_power=Pnw_external,
    transfer_m=M_external,
    bias=HaloTreeBias(...),
    reconstruction=ReconstructionConfig(
        smoothing_radius=15.0,
        bias_recon=2.0,
        cell_size=0.0,
        cic_window_power=0,
    ),
    q=q_grid,
    linear_power_q=Pq,
    no_wiggle_power_q=Pnw_q,
    transfer_m_q=Mq,
    settings=PostReconIrSettings(
        r_bao=105.0,
        k_ir=0.15,
        q_min=2.0 * 3.141592653589793 / box_size,
    ),
)

B = result.predict_deterministic(f_nl)
```

### 14.5 Matched adaptive 示例

```python
from marisa_b.post_recon_tree_png_ir import (
    AdaptiveReconstructionConfig,
)

adaptive = AdaptiveReconstructionConfig(
    bphi_recon=bias.bphi,
    fnl_recon_ratio=1.0,
)

result = ir_resummed_post_recon_finite_png_tree(
    ...,
    reconstruction=ReconstructionConfig(
        smoothing_radius=15.0,
        bias_recon=bias.b1,
    ),
    adaptive=adaptive,
)
```

此时应有

```python
result.displacement.sigma_f1 == (0,0,0)
result.displacement.sigma_f2 == (0,0,0)
```

到数值积分误差为止。

### 14.6 CIC-aware 示例

```python
reconstruction = ReconstructionConfig(
    smoothing_radius=15.0,
    bias_recon=2.0,
    cell_size=mesh_cell_size,
    cic_window_power=4,
)

settings = PostReconIrSettings(
    r_bao=105.0,
    k_ir=0.15,
    q_min=q_fundamental,
    mesh_damping="cartesian",
    angular_mu_order=32,
    angular_phi_order=64,
)
```

`mesh_damping="auto"` 在 CIC 有效时会自动选择 Cartesian 积分。若 CIC 已开启却强制 `isotropic`，代码会报错，而不是静默丢掉方向依赖。

---

## 15. 推荐的 native C++ 移植方案

Python oracle 的目标是先冻结公式与字段语义，再做 native 实现。建议不要直接修改已有 legacy `compute_post_recon_gaussian_ir_nowiggle`。

### 15.1 新增配置结构，而不是继续扩充一个标量 `ir_sigma2`

建议：

```cpp
struct PostReconTreePngIrConfig {
    real r_bao;
    real k_ir;
    real q_min;
    real bphi_recon;
    real fnl_recon_ratio;
    int angular_mu_order;
    int angular_phi_order;
    bool include_cartesian_cic;
    bool include_shift_noise;
};
```

### 15.2 新增返回结构

```cpp
struct PostReconTreePngIrResult {
    ComponentResult source_ir;

    real G_ir;
    real L_ir;
    real Q_ir;

    std::array<real,3> sigma_f0;
    std::array<real,3> sigma_f1;
    std::array<real,3> sigma_f2;

    std::array<real,3> damping_D0;
    std::array<real,3> damping_d1;
    std::array<real,3> damping_d2;

    real L_from_damping_G;
    real Q_from_damping_L;
    real Q_from_damping_G;
};
```

不要把 damping mixing 偷塞进已有 native fields；先以独立字段返回，便于回归审计。

### 15.3 Source evaluator 泛型化

现有 finite-PNG tree source 应允许 scalar type 为：

- `real`；
- first-order dual number。

这样逐腿 \(P_{\rm w}\) response 可解析传播。无需对 \(f_{\rm NL}\) 使用 dual number，因为 native 已经显式返回 \(G,L,Q\)。

### 15.4 Displacement integral

无 CIC：

- 一维 radial quadrature；
- 可预计算/缓存 \(\Sigma_0,\Sigma_1,\Sigma_2\)；
- cache key 必须包括 cosmology/spectrum identity、redshift、\(R\)、\(b_1\)、\(b_\phi\)、\(b_{\rm rec}\)、\(b_{\phi,\rm rec}\)、\(\eta\)、\(q_{\min}\)、\(\Lambda_{\rm IR}\)、\(r_{\rm BAO}\)。

CIC：

- radial × angular quadrature；
- variance 依赖 external leg direction；
- cache 至少按 normalized \(\hat k\) 或 shell orientation node；
- 不可只按 \(|k|\) 缓存。

### 15.5 建议的新入口

```cpp
PostReconTreePngIrResult
compute_post_recon_halo_bias_v1_local_png_tree_ir_vectors(
    const PowerSpectrum& P_L,
    const PowerSpectrum& P_nw,
    const PowerSpectrum& transfer_m,
    const ClosedTriangleVectors& vectors,
    const NativeConfig& recon_config,
    const HaloBiasV1Params& bias,
    const PostReconTreePngIrConfig& ir_config);
```

triangle-only convenience wrapper 可以随后添加，但 production shell projection 应调用 vectors 入口。

---

## 16. 与 shell projection 的集成顺序

对每一个 estimator orientation node：

1. 生成 closed Cartesian triangle；
2. 检查 finite-box allowed modes；
3. 读取 \(P_L,P_{\rm nw},\mathcal M\) 三条外腿；
4. 计算 no-wiggle post source；
5. 计算三条逐腿 wiggle responses；
6. 计算 CIC-aware \(\Sigma_{i,0},\Sigma_{i,1},\Sigma_{i,2}\)；
7. 组装完整 \(G_{\rm IR},L_{\rm IR},Q_{\rm IR}\)；
8. 应用现有 shell weight；
9. 使用原始 shell denominator，不因 forbidden modes 重新条件归一化。

建议把 displacement coefficients 先按 orientation node 缓存，以避免在 bias sampler 中反复积分。但如果 \(b_1,b_\phi\) 是采样参数，\(\Sigma_1,\Sigma_2\) 也依赖它们。可采用以下优化：

- 预计算只依赖 q/window 的基础积分；
- 将 \(a,n,d\) 的 bias polynomial 在 runtime 解析组合；
- CIC 情况下也可对 angular basis integrals 预计算。

---

## 17. 分阶段实施计划

### Phase A：Python oracle（本次已完成）

目标：冻结公式、字段语义、极限和测试。

验收：独立 pytest 全部通过。

### Phase B：fixed-\(b_{\rm rec}\)、无 CIC native 实现

这是第一生产版本，原因是：

- 一维 variance integral；
- source 已有；
- 最容易与 analytic limits 回归；
- 可直接评估 damping mixing 对当前分析的数值重要性。

必须输出 source-only 和 full exponent-response 两套结果做差。

### Phase C：adaptive denominator

加入：

- native 已有 \(G_1,G_2,Q_\times\) source bases；
- \(\Sigma_1,\Sigma_2\) 中的 denominator response；
- matched cancellation test。

### Phase D：Cartesian CIC

加入逐方向 variance；先在固定三角形上验证：

- CIC off rotation invariance；
- CIC on orientation dependence；
- angular order convergence；
- shell projection convergence。

### Phase E：reconstruction shift noise

只有在 mock/cat pipeline 明确给出 \(P_\epsilon(q)\) 或经过测量校准后启用。

### Phase F：可选 \(P_{\rm w}^2\) ULPT closure

单独模型标签，不改变 strict \(O(P_{\rm w})\) fiducial。

---

## 18. 必须执行的测试矩阵

### 18.1 Source tests

1. fixed source 逐项等于当前 native fields；
2. adaptive `gaussian_linear`；
3. adaptive `gaussian_quadratic`；
4. adaptive `png_linear_cross`；
5. \(R\to\infty\) source 回到 pre；
6. CIC off rotation invariance；
7. CIC on 保留绝对方向依赖。

### 18.2 Displacement tests

1. no-CIC 角积分退化到一维 Bessel 公式；
2. \(R\to\infty\)：\(\Sigma_0\to\Sigma_{\rm pre}\)，\(\Sigma_1,\Sigma_2\to0\)；
3. \(R\to0,b_{\rm rec}=b_1\)：Gaussian residual variance \(\to0\)；
4. matched adaptive：\(\Sigma_1=\Sigma_2=0\)；
5. fixed denominator：finite-f direct variance 与二次 polynomial 精确一致；
6. adaptive denominator：small-f finite differences 对上 \(\Sigma_1,\Sigma_2\)；
7. q-grid refinement；
8. \(q_{\min}\) sensitivity；
9. \(\Lambda_{\rm IR}\) sensitivity；
10. CIC angular quadrature convergence。

### 18.3 Full \(G,L,Q\) tests

以 direct finite-f strict expression

\[
B(f)=B_{\rm nw}(f)
+\sum_iD_i(f)B_{{\rm w},i}(f)
\]

做中心差分：

\[
G=B(0),
\]

\[
L=\frac{B(h)-B(-h)}{2h},
\]

\[
Q=\frac{B(h)+B(-h)-2B(0)}{2h^2}.
\]

与解析组装的 \(G_{\rm IR},L_{\rm IR},Q_{\rm IR}\) 对比。

### 18.4 Production mock tests

至少比较：

```text
post tree, no IR
post tree, Gaussian-only fixed damping
post tree, complete finite-PNG damping response
post tree, matched adaptive
post tree, CIC-aware
```

需要看：

- 每个 triangle/bin 的 \(\Delta G,\Delta L,\Delta Q\)；
- squeezed 与 BAO configurations；
- \(f_{\rm NL}=0,\pm100\)；
- shell projection 后差异；
- likelihood 的 \(\Delta f_{\rm NL}\)、误差和 bias covariance；
- \(q_{\min}\)、\(k_{\rm IR}\)、\(r_{\rm BAO}\)、angular order 稳定性。

---

## 19. 建议验收阈值

Python/native 逐三角形：

```text
source G/L/Q                 relative 1e-11, absolute scale-aware
Sigma0/Sigma1/Sigma2 no-CIC relative 1e-8
D0/d1/d2                     relative 1e-10
full G/L/Q                   relative 1e-9
```

CIC angular convergence：

```text
24x48 -> 32x64 -> 48x96
selected shell-node result changes < 1e-4 initially
production target set by covariance-weighted model error
```

最终标准不应只看逐点百分比，而应使用

\[
\Delta\chi^2
=\Delta\boldsymbol B^T C^{-1}\Delta\boldsymbol B
\]

和参数偏移测试。

---

## 20. 当前实现的测试状态

本交付环境运行：

```text
post-recon 专项：17 passed
与上一版 tree-level IR 基础测试合并：31 passed
```

覆盖：

- fixed-denominator analytic variance；
- pre limit；
- zero-smoothing limit；
- matched adaptive exact cancellation；
- fixed/adaptive finite-f variance finite differences；
- native adaptive source normalization；
- damping exponential Taylor coefficients；
- full \(G,L,Q\) direct finite-f derivatives；
- zero scale-dependent reconstruction response；
- Cartesian quadrature → isotropic limit；
- CIC orientation dependence；
- 禁止静默忽略 active CIC；
- shift noise；
- stochastic explicit closure；
- positive finite \(q_{\min}\)；
- current native `ir_sigma2` convention mapping。

这些是 standalone Python oracle tests，并不等于当前 MARISA-B C++ release suite 已通过。

---

## 21. 重要科学边界

### 21.1 这不是 one-loop model

没有 hard loop，没有 NLO matching，也没有 EFT counterterms。

### 21.2 这是 strict \(O(P_{\rm w})\)

没有把两个 dressed spectra 直接相乘所产生的 \(P_{\rm w}^2\) 当成严格结果。

### 21.3 long displacement 采用 linear-Gaussian cumulant

local-PNG primordial higher displacement cumulants没有被非微扰指数化。它们属于更高阶 PNG-IR 扩展，不是当前 tree source + leading BAO displacement approximation的一部分。

### 21.4 post catalogue/reconstruction validation 仍是独立 blocker

模型代码正确不代表现有 \(f_{\rm NL}=\pm100\) post catalogues 的 reconstruction procedure 已经验证。仓库当前 scientific-status 中的 catalogue quarantine 应继续保留，直到 data pipeline audit 完成。

### 21.5 finite-box cutoff 是模型参数/provenance

特别是在 \(f^2\) damping response 中，\(q_{\min}\) 不能被当成无关紧要的数值设置。

---

## 22. 推荐的当前 fiducial 与系统学变体

### 第一 fiducial

```text
fixed b_rec
CIC off in damping and source, or使用真实 pipeline 中确实关闭 CIC 的配置
strict O(Pw)
finite positive qmin = box fundamental mode
no shift noise
complete fNL-dependent damping response
```

### 第二阶段 fiducial

```text
actual Cartesian CIC
strict O(Pw)
orientation-before-shell projection
```

### 系统学变体

```text
Gaussian-only damping response disabled/enabled comparison
matched adaptive vs fixed denominator
qmin scan
kIR scan
rBAO scan
angular quadrature scan
optional shift noise
future ULPT Pw^2 closure
```

---

## 23. 参考文献与代码依据

1. **MARISA-B current main**
   `src/marisa_b/marisa_b_native.h`
   `src/marisa_b/marisa_b_native.cpp`
   `docs/SCIENTIFIC_STATUS.md`

2. **N. Sugiyama (2024)**, *Developing a Theoretical Model for the Resummation of Infrared Effects in the Pre- and Post-Reconstruction Galaxy Bispectra*, arXiv:2403.18262.

3. **IR-Safe and IR-Resummed Bispectra Before and After Reconstruction in Unified Lagrangian Perturbation Theory** (2026), arXiv:2602.20431.

4. **A. Vasudevan et al. (2019)**, *Time-sliced perturbation theory with primordial non-Gaussianity and effects of large bulk flows on inflationary oscillating features*, arXiv:1906.08697.

5. **N. Sugiyama (2024)**, *Discreteness Effects in the Post-Reconstruction Galaxy Power Spectrum*, arXiv:2406.01001.

6. **MARISA-B prior oracle delivery**
   `python/marisa_b/tree_png_ir_resummation.py`
   `docs/TREE_LEVEL_PNG_IR_RESUMMATION_PLAN.md`

---

## 24. 最终建议

对 MARISA-B 当前 tree-level post-recon 分析，正确的下一步不是把一个 pre-recon \(\Sigma^2\) 填入 `ir_sigma2` 后结束，而是：

1. 使用现有 post source kernels；
2. 根据 reconstruction field 的实际 \(Z_1/b_{\rm rec}\) 构造 residual long displacement；
3. 将 reconstructed variance 展开到 \(f^2\)；
4. 加入 damping exponent 产生的 \(G\to L\)、\(L\to Q\)、\(G\to Q\) mixing；
5. adaptive denominator 同时进入 source 和 displacement；
6. CIC 开启时逐 Cartesian orientation 计算；
7. 保持 finite \(q_{\min}\)；
8. 用 standalone oracle 固定结果，再移植到 native C++ 和 shell projector。

本交付代码已经完成第 1–7 项的 Python reference implementation，可直接交给独立 runner 做完整仓库测试和 native 对照。
