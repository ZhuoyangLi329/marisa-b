# MARISA-B tree-level finite-local-PNG halo bispectrum 的 RSD 公式

**目标范围**：在 MARISA-B 当前的

\[
B_h(f_{\rm NL})=G+f_{\rm NL}L+f_{\rm NL}^2Q
\]

pre/post-reconstruction halo tree 模型中加入平行视线近似下的红移空间畸变（RSD）。本文只保留 tree-level；不引入 Gaussian 或 PNG one-loop，不把 FoG counterterm 冒充成严格 tree 项。

本文使用以下符号，以避免两个常见的记号冲突：

- \(\lambda\equiv f_{\rm NL}\)：local PNG 幅度；
- \(f_g\equiv d\ln D/d\ln a\)：线性增长率；
- \(\nu_{ij}\equiv\hat{\boldsymbol k}_i\!\cdot\!\hat{\boldsymbol k}_j\)：三角形内部夹角余弦；
- \(\mu_i\equiv\hat{\boldsymbol k}_i\!\cdot\!\hat{\boldsymbol n}\)：第 \(i\) 条腿与 LOS 的夹角余弦；
- \(\mu_{ij}^{\rm los}\equiv(\boldsymbol k_i+\boldsymbol k_j)\!\cdot\!\hat{\boldsymbol n}/|\boldsymbol k_i+\boldsymbol k_j|\)。

特别注意：MARISA-B 当前 `Triangle.mu12` 表示 \(\nu_{12}\)，而很多 RSD 文献的 \(\mu_{12}\) 表示 \(\mu_{12}^{\rm los}\)。原生实现中必须使用不同变量名，不能继续都叫 `mu12`。

---

## 1. 文献结论与对 MARISA-B 的含义

### 1.1 标准 tree-level RSD bispectrum

Scoccimarro、Couchman & Frieman 建立了以 \(Z_n\) kernel 表示的红移空间 bispectrum。对不含 LOS selection、velocity bias 和 wide-angle 的 biased tracer，tree bispectrum由 \(Z_1\) 和 \(Z_2\) 完全决定。

Ivanov et al. (2022) 对这一系统 tree 模型做了高精度数值验证，并同时处理 bias、非线性 RSD、IR resummation、binning 和 monopole projection。他们在其 PT challenge、\(z=0.61\) 的特定设置中发现 tree bispectrum monopole 可用至约 \(k_{\max}=0.08\,h\,{\rm Mpc}^{-1}\)。这个数值可作为 MARISA-B mock 检验的起点，但不是对所有 halo sample、红移和 reconstruction 方法的普适保证。

### 1.2 DESI DR1 的两种做法不能混为一谈

DESI DR1 官方 joint power-spectrum/bispectrum full-shape 分析展示了标准 SPT 的 \(Z_1,Z_2\) 公式，但生产模型使用 GEO-FPT、非线性谱和经验性 FoG damping。它不是本项目所要求的“纯 tree-level MARISA-B”。因此：

- 可用其 \(Z_1,Z_2\) 公式作 Gaussian RSD 回归；
- 不应把 GEO-FPT 修正、非线性 \(P\) 或 Lorentzian FoG 直接移入当前严格 tree 模型。

独立的 DESI DR1 reanalysis 使用 tree bispectrum、stochastic 和一个形式上属于更高阶的 \(k^2\) FoG counterterm，并在 \(k\in[0.02,0.08)\,h\,{\rm Mpc}^{-1}\) 内拟合 bispectrum monopole。它对 MARISA-B 最有价值的部分是：

1. LOS orientation average 与 bin average 必须作为模型的一部分；
2. stochastic 在红移空间分裂成不同的角依赖 shape；
3. AP remapping、window/binning 不能与 RSD kernel 混在同一个函数里；
4. 其 \(c_1k^2\) 项不是严格 tree，默认不应加入本次实现。

### 1.3 post-reconstruction 的直接依据

Shirasaki et al. (2021) 研究了红移空间 post-reconstruction PNG bispectrum。他们的 catalog reconstruction 与 MARISA-B 当前 map 相同：位移由**包含 RSD 的 halo density**估计，只除以 \(b_{\rm fid}\)，且不通过 reconstruction 去除线性 RSD；halo 和 random 使用同一个位移。其关键结果是

\[
Z_{1,\rm rec}=Z_1,
\]

\[
Z_{2,\rm rec}=Z_2+\Delta Z_2,
\]

其中 \(\Delta Z_2\) 正是 MARISA-B 当前 `shift_factor_post` / partition map 的二阶形式。Sugiyama (2024) 从一般 post-reconstruction kernel 的角度给出了相同结构。

因此，MARISA-B 不需要另造一个 post-RSD reconstruction formalism；应复用现有 partition engine，把 density kernel 从实空间 \(K_n\) 换成红移空间 \(Z_n\)。

### 1.4 “完整”的边界

Desjacques, Jeong & Schmidt (2018) 说明，真实 survey 的一般 redshift-space bias expansion 还包含 LOS selection operators；若不能用物理样本选择论证其可忽略，就不能称为一般 survey 的完整模型。

本方案的“完整”含义严格限定为：

- periodic box 或 distant-observer halo mock；
- 固定全局 LOS；
- 无 selection effects；
- 无 velocity bias；
- 无 wide-angle；
- 当前 MARISA-B 的 \(b_1,b_2,b_{K^2},b_\phi,b_{\phi\delta},b_{\phi^2}\) deterministic bias basis；
- 截断到 \(\lambda^2\)；
- tree-level density perturbation theory。

在这些假设下，下面的 deterministic model 是闭合的。它不是完整 DESI survey likelihood。

---

## 2. 当前 MARISA-B 的 real-space finite-PNG kernel

定义

\[
P_i=P_{\rm L}(k_i),\qquad
\nu_{ij}=\hat{\boldsymbol k}_i\cdot\hat{\boldsymbol k}_j,
\qquad
S_2(i,j)=\nu_{ij}^2-\frac13,
\]

\[
u_i\equiv \frac{1}{\mathcal M_i},\qquad
p_i\equiv b_\phi u_i.
\]

为避免与 \(u_i\) 排版混淆，以下仍写 \(u_i=1/\mathcal M_i\)。EdS kernels 为

\[
F_2(i,j)=\frac57+\frac12\nu_{ij}
\left(\frac{k_i}{k_j}+\frac{k_j}{k_i}\right)
+\frac27\nu_{ij}^2,
\]

\[
G_2(i,j)=\frac37+\frac12\nu_{ij}
\left(\frac{k_i}{k_j}+\frac{k_j}{k_i}\right)
+\frac47\nu_{ij}^2,
\]

\[
S_2(i,j)=\nu_{ij}^2-\frac13.
\]

当前 real-space 一阶 kernel 是

\[
K_1^h(i;\lambda)=b_1+\lambda p_i.
\]

二阶 kernel 写成 \(\lambda\) 的二次多项式：

\[
K_2^h(i,j;\lambda)=H_{ij}+\lambda D_{ij}+\lambda^2E_{ij},
\]

其中

\[
H_{ij}=b_1F_2(i,j)+\frac{b_2}{2}+b_{K^2}S_2(i,j),
\]

\[
A_{ij}=\nu_{ij}\left(\frac{k_i}{k_j}u_i+\frac{k_j}{k_i}u_j\right),
\]

\[
D_{ij}=\frac12\left[b_\phi A_{ij}+b_{\phi\delta}(u_i+u_j)\right],
\]

\[
E_{ij}=\frac12 b_{\phi^2}u_iu_j.
\]

这里的 \(b_{\phi^2}\) 是独立 operator coefficient，不是 \(b_\phi^2\)。

单位 \(f_{\rm NL}\) 的 primordial matter template 为

\[
\widehat B_0
=2\sum_{\rm cyc}P_iP_j\frac{\mathcal M_k}{\mathcal M_i\mathcal M_j}.
\]

---

## 3. 从 real-space kernel 到 redshift-space kernel

### 3.1 一阶 kernel

在 plane-parallel approximation 下，minimal RSD kernel 为

\[
\boxed{
Z_1^s(i;\lambda)
=K_1^h(i;\lambda)+f_g\mu_i^2
=a_i+\lambda p_i,
}
\]

其中

\[
a_i=b_1+f_g\mu_i^2.
\]

这表明 PNG scale-dependent bias 和 Kaiser response 应在同一个外腿 kernel 中组合，而不是在最终 bispectrum 上分别乘两个 correction factor。

### 3.2 二阶 kernel

令

\[
\boldsymbol K_{ij}=\boldsymbol k_i+\boldsymbol k_j,
\quad K_{ij}=|\boldsymbol K_{ij}|,
\quad K_{ij,\parallel}=\boldsymbol K_{ij}\cdot\hat{\boldsymbol n},
\]

\[
\mu_{ij}^{\rm los}=\frac{K_{ij,\parallel}}{K_{ij}}.
\]

minimal redshift-space kernel 为

\[
\boxed{
\begin{aligned}
Z_2^s(i,j;\lambda)={}&K_2^h(i,j;\lambda)
+f_g(\mu_{ij}^{\rm los})^2G_2(i,j)\\
&+\frac{f_gK_{ij,\parallel}}{2}
\left[
\frac{\mu_i}{k_i}Z_1^s(j;\lambda)
+\frac{\mu_j}{k_j}Z_1^s(i;\lambda)
\right].
\end{aligned}
}
\]

第二行是 nonlinear redshift-space mapping。把 \(Z_1\) 的 PNG 部分也放入这一项非常重要：它自动产生 RSD–\(b_\phi\) cross term；若只把标准 Gaussian \(Z_2\) 与 real-space PNG correction 相加，会漏掉它。

把 \(Z_2^s\) 展开为

\[
Z_2^s(i,j;\lambda)=c_{ij}+\lambda d_{ij}+\lambda^2e_{ij},
\]

得到

\[
\boxed{
\begin{aligned}
c_{ij}={}&H_{ij}
+f_g(\mu_{ij}^{\rm los})^2G_2(i,j)\\
&+\frac{f_gK_{ij,\parallel}}{2}
\left(\frac{\mu_i}{k_i}a_j+\frac{\mu_j}{k_j}a_i\right),
\end{aligned}
}
\]

\[
\boxed{
d_{ij}=D_{ij}
+\frac{f_gK_{ij,\parallel}}{2}
\left(\frac{\mu_i}{k_i}p_j+\frac{\mu_j}{k_j}p_i\right),
}
\]

\[
\boxed{e_{ij}=E_{ij}.}
\]

当 \(f_g\to0\) 时，精确回到 MARISA-B 当前的 \(K_1,K_2\)。当 \(b_\phi=b_{\phi\delta}=b_{\phi^2}=0\) 时，回到标准 Gaussian biased-tracer \(Z_1,Z_2\)。

### 3.3 Gaussian 极限与 DESI 常见展开式的逐项对应

把 compact mapping 展开，Gaussian 二阶 kernel 为

\[
\boxed{
\begin{aligned}
Z_{2,G}^s(i,j)={}&b_1F_2+\frac{b_2}{2}+b_{K^2}S_2
+f_g(\mu_{ij}^{\rm los})^2G_2\\
&+\frac{b_1f_g}{2}
\left[
\mu_i^2+\mu_j^2+\mu_i\mu_j
\left(\frac{k_i}{k_j}+\frac{k_j}{k_i}\right)
\right]\\
&+f_g^2\mu_i\mu_j
\left[
\mu_i\mu_j+\frac12
\left(
\mu_i^2\frac{k_i}{k_j}
+\mu_j^2\frac{k_j}{k_i}
\right)
\right].
\end{aligned}
}
\]

这就是 DESI/标准 SPT 常用的展开形式。需要特别注意 tidal-bias normalization：若外部公式写作

\[
\frac12\left[b_2+b_{s^2}S_2\right],
\]

而 MARISA-B native 写作

\[
\frac{b_2}{2}+b_{K^2}S_2,
\]

则两者的参数转换是

\[
\boxed{b_{s^2}^{\rm external}=2b_{K^2}^{\rm MARISA-B}.}
\]

不能把同一个数值同时代入两个 convention。本文与代码采用 EdS 的 \(F_2,G_2\) 以及通常的 velocity-divergence sign convention；若 native 以后支持精确时变 kernels，\(F_2,G_2\) 应作为同一接口的可替换输入，而不是修改 RSD mapping 的代数结构。

---

## 4. pre-reconstruction 的 \(G,L,Q\)

对每个 cyclic pair \((i,j;k)\)，deterministic gravity/bias tree 是

\[
2P_iP_j Z_1^s(i;\lambda)Z_1^s(j;\lambda)Z_2^s(i,j;\lambda).
\]

使用

\[
Z_1^s(i)=a_i+\lambda p_i,
\qquad
Z_2^s(i,j)=c_{ij}+\lambda d_{ij}+\lambda^2e_{ij},
\]

可直接读取三阶系数：

\[
\boxed{
G_{ij}=2P_iP_j\,a_ia_jc_{ij},
}
\]

\[
\boxed{
L_{ij}=2P_iP_j
\left[(p_ia_j+a_ip_j)c_{ij}+a_ia_jd_{ij}\right],
}
\]

\[
\boxed{
Q_{ij}=2P_iP_j
\left[
 p_ip_jc_{ij}
 +(p_ia_j+a_ip_j)d_{ij}
 +a_ia_je_{ij}
\right].
}
\]

primordial tree 是

\[
\lambda\,\widehat B_0
\prod_{i=1}^{3}Z_1^s(i;\lambda).
\]

截断到 \(\lambda^2\) 后，

\[
\boxed{
L_{\rm prim}=\widehat B_0a_1a_2a_3,
}
\]

\[
\boxed{
Q_{\rm prim}=\widehat B_0
\left(p_1a_2a_3+a_1p_2a_3+a_1a_2p_3\right).
}
\]

最终

\[
\boxed{
G=\sum_{\rm cyc}G_{ij},
\quad
L=\sum_{\rm cyc}L_{ij}+L_{\rm prim},
\quad
Q=\sum_{\rm cyc}Q_{ij}+Q_{\rm prim}.
}
\]

这套写法有三个实现优点：

1. 不需要对有限 \(f_{\rm NL}\) 做数值差分；
2. 所有 RSD–PNG cross terms 由多项式乘法自动产生；
3. \(f_g=0\)、PNG bias=0、primordial-only 等极限都能逐项测试。

---

## 5. post-reconstruction：当前 MARISA-B 方法

### 5.1 当前 reconstruction contract

MARISA-B 当前采用同一位移移动 halo 与 random 的 map：

\[
\delta_{h,\rm rec}^s(\boldsymbol k)
=\int d^3x\,e^{-i\boldsymbol k\cdot\boldsymbol x}
 e^{-i\boldsymbol k\cdot\boldsymbol s(\boldsymbol x)}
 \delta_h^s(\boldsymbol x).
\]

位移由包含 RSD 的 tracer density 估计，但默认只除以 \(b_{\rm rec}\)，并不去除 Kaiser factor。这与 Shirasaki et al. 的 post-PNG RSD setup 相同。

对 pair \((i,j)\)，定义

\[
r_i^{(ij)}
=-\frac{\boldsymbol K_{ij}\cdot\boldsymbol k_i}{k_i^2}
\frac{W_{\rm rec}(\boldsymbol k_i)}{b_{\rm rec}},
\]

\[
W_{\rm rec}(\boldsymbol k_i)
=e^{-k_i^2R^2/2}
\prod_{a=x,y,z}{\rm sinc}^{p}_{\rm CIC}
\left(\frac{k_{i,a}\Delta}{2}\right),
\]

其中当前 JAXRecon contract 使用 density-paint 与 displacement-read 合并后的 CIC power。

二阶 reconstructed kernel 为

\[
\boxed{
Z_{2,\rm rec}^s(i,j;\lambda)
=Z_2^s(i,j;\lambda)
+\frac12\left(r_i^{(ij)}+r_j^{(ij)}\right)
 Z_1^s(i;\lambda)Z_1^s(j;\lambda).
}
\]

并且

\[
\boxed{Z_{1,\rm rec}^s=Z_1^s.}
\]

因此 primordial tree 的 post/pre source 完全相同；变化发生在 gravity/bias 的 \(Z_2\) source 和后续 IR damping/covariance，而不是 \(Z_1\) 本身。

### 5.2 fixed-\(b_{\rm rec}\) 的显式系数

写

\[
r_{ij}=r_i^{(ij)}+r_j^{(ij)}.
\]

则

\[
Z_{2,\rm rec}^s=c_{ij}^{\rm rec}
+\lambda d_{ij}^{\rm rec}
+\lambda^2e_{ij}^{\rm rec},
\]

其中

\[
\boxed{
c_{ij}^{\rm rec}=c_{ij}+\frac12r_{ij}a_ia_j,
}
\]

\[
\boxed{
d_{ij}^{\rm rec}=d_{ij}
+\frac12r_{ij}(a_ip_j+p_ia_j),
}
\]

\[
\boxed{
e_{ij}^{\rm rec}=e_{ij}+\frac12r_{ij}p_ip_j.
}
\]

把 \((c,d,e)\) 替换为 \((c^{\rm rec},d^{\rm rec},e^{\rm rec})\)，即可直接使用第 4 节的 \(G,L,Q\) 公式。

这证明现有 MARISA-B partition engine 可以复用；只需让 density block 和 displacement-source block 调用 redshift-space finite-PNG kernel polynomial。

### 5.3 adaptive local-PNG reconstruction denominator

若 reconstruction denominator 取

\[
d_i(\lambda)=d_{0i}+\lambda d_{1i},
\]

其中当前 adaptive local-PNG proposal 对应

\[
d_{0i}=b_{\rm rec},
\qquad
d_{1i}=\eta\,b_{\phi,\rm rec}u_i,
\]

则令

\[
N_i=-\frac{\boldsymbol K_{ij}\cdot\boldsymbol k_i}{k_i^2}
 W_{\rm rec}(\boldsymbol k_i),
\]

有

\[
r_i(\lambda)=\frac{N_i}{d_{0i}+\lambda d_{1i}}
=r_i^{(0)}+\lambda r_i^{(1)}+\lambda^2r_i^{(2)}+\cdots,
\]

\[
\boxed{
r_i^{(0)}=\frac{N_i}{d_{0i}},
\quad
r_i^{(1)}=-\frac{N_id_{1i}}{d_{0i}^2},
\quad
r_i^{(2)}=\frac{N_id_{1i}^2}{d_{0i}^3}.
}
\]

令 \(r_\Sigma^{(m)}=r_i^{(m)}+r_j^{(m)}\)，则 reconstruction 对 \(Z_2\) 的系数修正为

\[
\boxed{
\Delta c_{ij}=\frac12r_\Sigma^{(0)}a_ia_j,
}
\]

\[
\boxed{
\Delta d_{ij}=\frac12\left[
 r_\Sigma^{(1)}a_ia_j
 +r_\Sigma^{(0)}(a_ip_j+p_ia_j)
\right],
}
\]

\[
\boxed{
\Delta e_{ij}=\frac12\left[
 r_\Sigma^{(2)}a_ia_j
 +r_\Sigma^{(1)}(a_ip_j+p_ia_j)
 +r_\Sigma^{(0)}p_ip_j
\right].
}
\]

当前仓库的 adaptive basis 是在 real-space source 下构造的。加入 RSD 后，\(a_i=b_1+f_g\mu_i^2\) 使其角结构发生变化，不能只沿用原 basis 再乘一个全局 Kaiser prefactor；应由上述多项式重新生成。

### 5.4 不同 reconstruction 方法必须显式分支

当前默认应命名为

```text
same_shift_no_linear_rsd_removal
```

而不要含糊地叫 `standard`。若未来 catalog 使用

\[
d_{0i}=b_{\rm rec}+f_{\rm rec}\mu_i^2
\]

或 RecIso/RecSym 中 halo 与 random 不同的 LOS displacement，则必须建立新的 reconstruction enum 和新的 kernel map。不能只改一个 denominator 就声称支持完整 RecIso/RecSym。

---

## 6. orientation 与 multipole projection

RSD 后，一个只给 \((k_1,k_2,k_3)\) 的 scalar triangle API 不再足以定义模型。必须保留 LOS 或完整 Cartesian triangle。

对固定 shape，定义

\[
\nu_{12}=\frac{k_3^2-k_1^2-k_2^2}{2k_1k_2}.
\]

选择 \(\mu_1=\mu\) 和绕 \(\boldsymbol k_1\) 的方位角 \(\phi\)，则可取

\[
\boxed{
\mu_2=\mu\nu_{12}
-\sqrt{1-\mu^2}\sqrt{1-\nu_{12}^2}\cos\phi,
}
\]

\[
\boxed{
\mu_3=-\frac{k_1\mu_1+k_2\mu_2}{k_3}.
}
\]

常用 Legendre/Scoccimarro multipole 定义为

\[
\boxed{
B_\ell(k_1,k_2,k_3)
=\frac{2\ell+1}{4\pi}
\int_{-1}^{1}d\mu
\int_0^{2\pi}d\phi\,
\mathcal L_\ell(\mu)B^s.
}
\]

monopole 是

\[
\boxed{
B_0=\frac1{4\pi}\int d\mu\,d\phi\,B^s.
}
\]

代码 oracle 实现的是这一 normalization。若测量使用 TriPoSH \(B_{000}\)、\(B_{202}\) 等，需要按 estimator 的 spherical-harmonic normalization 建立独立 projector，不能仅靠重命名 `B0`。

### 连续积分与 exact lattice

DESI DR1 reanalysis 采用连续 mode integral 与 Gauss–Legendre quadrature，并报告其设置下离散性约为百分比量级。MARISA-B 的 periodic box 可以做得更严格：

1. 对每个实际 Cartesian mode triplet 计算 \(G,L,Q\)；
2. 使用 box 的固定 LOS；
3. 再按 estimator 的 mode count 和 bin normalization 平均。

若 reconstruction CIC window 开启，它依赖绝对 Cartesian orientation，更不能先做连续 LOS 平均后再乘 mesh correction。

---

## 7. stochastic sector

独立 DESI DR1 reanalysis 的 Gaussian tree stochastic 模型可重写为

\[
B_{\rm stoch}
=\frac{B_{\rm shot}}{\bar n}
\sum_i P_i\,b_1\left(b_1+f_g\mu_i^2\right)
+\frac{P_{\rm shot}}{\bar n}
\sum_i P_i\,f_g\mu_i^2\left(b_1+f_g\mu_i^2\right)
+\frac{A_{\rm shot}}{\bar n^2}.
\]

因此可定义 raw bases

\[
\boxed{
S_B=\sum_iP_i b_1a_i,
\quad
S_P=\sum_iP_i f_g\mu_i^2a_i,
\quad
S_A=1.
}
\]

当 \(f_g=0\) 时，

\[
S_B=b_1^2\sum_iP_i,
\qquad S_P=0,
\]

恰好恢复当前 MARISA-B Gaussian `alpha3` raw shape。

但是，当前 MARISA-B 的 PNG stochastic fields 并不是简单的
\((b_1+\lambda b_\phi/\mathcal M)^2P\) 展开；例如线性项没有自动的因子 2。因此不能未经推导就把 \(b_1\to Z_1\) 代入并宣称得到完整 finite-PNG RSD stochastic model。

建议第一版：

- deterministic \(G,L,Q\) 全部支持 RSD；
- Gaussian stochastic 提供 `desi_tree` 三个 raw bases；
- PNG stochastic 保留现有 real-space closure 作为独立、明确标记的模型变体，或暂时关闭；
- 在文档与 likelihood metadata 中记录这一边界。

---

## 8. FoG、counterterm 与严格 tree-level 的边界

DESI 官方分析使用经验 FoG damping；独立 reanalysis 和 Ivanov tree pipeline 加入一个 \(k^2\) RSD counterterm。它们在数据建模上可能必要，但在 power counting 上并非纯 tree contribution。

本项目应至少提供两个明确分开的 model tier：

```text
strict_tree_rsd
    Z1 + Z2 + tree stochastic
    无 FoG damping、无 k^2 counterterm

tree_rsd_plus_fog_nuisance
    strict_tree_rsd + 明确的经验/有效 FoG 项
    只在 mock closure 证明需要时启用
```

绝不能把第二类默认打开后仍把模型称为“严格 tree-level”。对当前 local-PNG 大尺度目标，首先应在 \(k_{\max}\le 0.08\,h\,{\rm Mpc}^{-1}\) 附近测试 strict model，再根据 residual 的 \(k\mu\) 结构决定是否增加 nuisance，而不是直接复制 DESI 的 \(\sigma_B\) 或 \(c_1\) prior。

---

## 9. 与 IR resummation 的组合

RSD 加入后，BAO damping 不再是单一的 \(e^{-k^2\Sigma^2}\)。线性红移空间 displacement matrix 为

\[
R_{ab}=\delta_{ab}+f_g n_an_b.
\]

若 real-space covariance 在 MARISA-B convention 下为 \(\Sigma^2I\)，则

\[
\Sigma_{ab}^s=\Sigma^2(RR^T)_{ab},
\]

每条腿的 damping 为

\[
\boxed{
D_i^s=\exp[-k_{i,a}\Sigma_{ab}^s k_{i,b}]
=\exp\left[-k_i^2\Sigma^2
\left(1+f_g(2+f_g)\mu_i^2\right)\right].
}
\]

这条公式使用 MARISA-B 既有 `exp(-k^2 sigma2)` 的标量 convention；若其他文献把 \(1/2\) 吸收到 \(\Sigma^2\) 中，不能直接比较参数数值。

正确顺序是

```text
Cartesian triangle + LOS
    -> RSD source G,L,Q
    -> legwise anisotropic IR dressing
    -> reconstruction-specific post covariance（若 post）
    -> orientation/shell projection
```

不能先得到 real-space monopole，再乘一个“平均 Kaiser factor”；也不能先角平均，再应用 \(\mu\)-dependent damping。

post-reconstruction 的 covariance 还依赖 reconstruction shift convention。此前的 scalar post-IR \(S_0,S_1,S_2\) 结构应升级为 LOS tensor，或至少为每条 oriented leg 返回 damping。本文交付的代码只提供通用 covariance hook和 pre-RSD covariance；不猜测一个未经验证的 post-RSD scalar \(\Sigma\)。

---

## 10. AP、wide-angle 与 survey selection

这些都不属于 halo-box RSD kernel，但在 DESI-like survey 应作为独立 wrapper：

- **AP remapping**：先把 fiducial \((k_\parallel,k_\perp)\) 映射到 trial cosmology，再调用 oriented RSD kernel；bispectrum amplitude 还带体积 Jacobian。
- **wide-angle**：全局固定 LOS 的 plane-parallel projector 不适用于大开角 survey；需要 local-LOS estimator 对应的 expansion。
- **selection operators**：真实 target selection 可生成与 RSD 退化但不完全相同的 LOS-dependent bias；不能无条件忽略。
- **window convolution**：应在理论 multipoles/bins 之后由 survey window operator处理，而不是写进 \(Z_2\)。

MARISA-B 当前 periodic halo mock 的第一版可以明确关闭这些模块。

---

## 11. 最小生产模型建议

### Fiducial v1

```text
plane_parallel = true
global_los = box LOS
selection_effects = off
velocity_bias = off
wide_angle = off
fog = off
ap = off
reconstruction = same_shift_no_linear_rsd_removal
finite_png_order = 2
projection = estimator-matched exact lattice（优先）或连续 B0 oracle
```

### 必须保存的 provenance

```text
f_growth
LOS vector
RSD convention / velocity-divergence sign
F2/G2 convention
bias basis conversion
reconstruction method
b_rec, R, CIC cell size/power
adaptive denominator parameters
multipole normalization
orientation quadrature order 或 exact mode list/hash
IR covariance convention
stochastic model enum
```

---

## 12. 主要参考文献

1. Scoccimarro, Couchman & Frieman, *The Bispectrum as a Signature of Gravitational Instability in Redshift Space*, ApJ 517 (1999), arXiv:astro-ph/9808305.
2. Desjacques, Jeong & Schmidt, *The Galaxy Power Spectrum and Bispectrum in Redshift Space*, JCAP 12 (2018) 035, arXiv:1806.04015.
3. Ivanov et al., *Precision analysis of the redshift-space galaxy bispectrum*, Phys. Rev. D 105, 063512 (2022), arXiv:2110.10161.
4. Shirasaki et al., *Constraining Primordial Non-Gaussianity with Post-reconstructed Galaxy Bispectrum in Redshift Space*, Phys. Rev. D 103, 023506 (2021), arXiv:2010.04567.
5. Sugiyama, *Developing a Theoretical Model for the Resummation of Infrared Effects in the Pre- and Post-Reconstruction Galaxy Bispectra*, arXiv:2403.18262.
6. DESI Collaboration, *Full-Shape analysis of the power spectrum and bispectrum of DESI DR1 LRG and QSO samples*, arXiv:2503.09714.
7. Philcox et al., *Reanalyzing DESI DR1: 1. ΛCDM Constraints from the Power Spectrum & Bispectrum*, arXiv:2507.13433.
8. Barreira, *On the impact of galaxy bias uncertainties on primordial non-Gaussianity constraints*, JCAP 12 (2022) 031, arXiv:2107.06887.
9. Moradinezhad Dizgah et al., finite local-PNG halo bispectrum bias expansion, arXiv:2010.14523.
