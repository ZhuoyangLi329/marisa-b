# MARISA-B tree-level finite-local-PNG halo bispectrum：RSD 代码实施计划

## 0. 文档目的

本文给出一套可以直接交给代码审阅者或另一模型执行的实施方案，用于在 MARISA-B 当前

\[
B_h(f_{\rm NL})=G+f_{\rm NL}L+f_{\rm NL}^2Q
\]

pre/post-reconstruction tree-level halo bispectrum 中加入 plane-parallel redshift-space distortions（RSD）。

本文是**代码架构与验收文档**。完整公式见同目录：

```text
docs/TREE_LEVEL_PNG_RSD_FORMULAE.md
```

Python 数值 oracle 见：

```text
python/marisa_b/tree_png_rsd_reference.py
```

专项测试见：

```text
tests/python/test_tree_png_rsd_reference.py
```

本方案严格不加入 Gaussian 或 PNG one-loop。FoG damping、\(k^2\) counterterm、GEO-FPT、非线性输入功率谱均不属于 fiducial v1。

---

## 1. 先给出实现结论

### 1.1 应做的核心改动

MARISA-B 不需要重写 tree bispectrum engine。最小且科学上闭合的改动是：

1. 将当前实空间一阶 kernel
   \[
   K_1^h=b_1+f_{\rm NL}b_\phi/\mathcal M
   \]
   升级为含 LOS 的红移空间多项式 \(Z_1^s\)。
2. 将当前实空间二阶 kernel \(K_2^h\) 升级为标准红移空间 \(Z_2^s\)，且 nonlinear RSD mapping 内必须使用**完整 finite-PNG \(Z_1^s\)**。
3. 保留现有 \(G+f_{\rm NL}L+f_{\rm NL}^2Q\) 多项式结构，用解析多项式乘法生成所有 RSD–PNG 交叉项。
4. pre-recon 使用 oriented Cartesian triangle + global LOS；RSD 后仅凭 \((k_1,k_2,k_3)\) 已不足以定义 bispectrum。
5. post-recon 复用当前 same-shift partition map；把 partition 中的 density kernels 换为 \(Z_1^s,Z_2^s\)，不要另造一个独立 post-RSD 公式分支。
6. 增加 estimator-matched orientation/multipole projector；优先支持 periodic-box exact lattice average，再提供 continuum Gauss–Legendre oracle。
7. Gaussian stochastic 使用独立的 RSD raw bases；finite-PNG stochastic 暂不宣称唯一闭合。
8. RSD source 与 IR resummation 分层：先计算 oriented \(G,L,Q\)，再做各向异性 IR dressing，最后投影。

### 1.2 不应做的改动

以下做法应被 code review 拒绝：

- 在 real-space bispectrum 上乘一个平均 Kaiser factor；
- 只把 \(b_1\) 换成 \(b_1+f_g\mu^2\)，但不加入 \(G_2\) 和 nonlinear mapping；
- 在 Gaussian \(Z_2\) 后面直接加 real-space PNG correction，而遗漏 mapping 中的 \(b_\phi/\mathcal M\)；
- 先算 monopole，再乘 \(\mu\)-dependent IR damping；
- 在 CIC 开启时先丢掉 Cartesian orientation；
- 默认复制 DESI 的 FoG 或 \(k^2\) nuisance，同时仍把模型称为严格 tree-level；
- 把官方 DESI 的 GEO-FPT production model 当作纯 SPT tree kernel；
- 将 current real-space PNG stochastic bases 用 \(b_1\to b_1+f_g\mu^2\) 机械替换后称为“完整 finite-PNG RSD stochastic”。

---

## 2. 文献调研转化为工程约束

### 2.1 DESI DR1 官方分析

官方 DESI DR1 joint power-spectrum/bispectrum 分析列出了标准 \(Z_1,Z_2\) tree expression，但 production likelihood 使用 GEO-FPT、非线性谱和经验 FoG。对 MARISA-B 的工程含义是：

- 标准 \(Z_1,Z_2\) 可作为 Gaussian 极限回归；
- GEO-FPT calibration、Lorentzian/Gaussian FoG、非线性输入谱不能进入 strict-tree v1；
- DESI 的 estimator/bin/window/AP 层应与 source kernel 解耦。

### 2.2 DESI DR1 independent reanalysis

独立 reanalysis 更接近本项目目标：bispectrum 部分使用 tree source，同时加入 stochastic 和一个形式上更高阶的 RSD counterterm。其工程启示：

- orientation average 是模型的一部分；
- Gaussian stochastic 在红移空间至少分裂为两个 \(P_L/\bar n\) angular shapes 和一个常数 \(1/\bar n^2\) shape；
- continuum quadrature 可以作为 oracle，但 periodic box 最终应匹配实际离散 modes；
- \(k^2\) counterterm 是可选 nuisance 层，不应进入 strict-tree core。

### 2.3 Ivanov et al. tree-level precision model

该工作系统验证了 bias、nonlinear RSD、IR、binning 与 multipole projection 的组合。工程上最重要的是：

- kernel、IR、projection 三层不能颠倒顺序；
- 即使只拟合 monopole，也必须先保留 LOS orientation；
- \(k_{\max}\simeq0.08\,h\,\mathrm{Mpc}^{-1}\) 只能作为 mock-validation 起始点，不能硬编码成理论保证。

### 2.4 post-reconstruction 文献

Shirasaki et al. 和 Sugiyama 的结果与 MARISA-B 当前 same-shift map 相容：

\[
Z_{1,\rm rec}=Z_1,
\qquad
Z_{2,\rm rec}=Z_2+\frac12(r_i+r_j)Z_{1i}Z_{1j}.
\]

因此工程原则是**复用 current partition engine**。若未来 catalog 使用 RecIso、RecSym、Kaiser-divided displacement 或 halo/random 不同 shift，则新增 reconstruction enum 和对应 map，不得修改当前默认语义。

### 2.5 一般 survey 的限制

一般 redshift-space bias expansion 可以含 LOS selection operators、velocity bias 和 wide-angle terms。periodic halo box 的 v1 可以关闭这些项，但必须在 API 和 metadata 中显式声明；未来 survey likelihood 不能静默沿用“无 selection”的假设。

---

## 3. 当前 MARISA-B 代码的可复用入口

根据当前仓库结构，RSD 实现应围绕以下逻辑扩展，而不是复制另一套模型：

```text
src/marisa_b/marisa_b_native.cpp
src/marisa_b/marisa_b_native.h
```

现有重要入口/辅助结构包括：

```text
compute_pre_recon_halo_bias_v1_local_png_tree_dfNL(...)
compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(...)
compute_post_recon_halo_local_png_brec_finite_tree_bases_vectors(...)
shift_factor_post(...)
halo_bias_v1_zrec_kernel_vec(...)
```

当前 post vector API 已经保留 Cartesian orientation，以支持 CIC window。这恰好是 RSD 所需要的接口形态。

### 3.1 不建议修改旧函数语义

不应给旧 real-space function 增加一个默认 LOS 参数后改变其输出。建议新增明确命名的 RSD API，旧 API 继续作为：

- \(f_g=0\) 回归基准；
- 现有 production reproducibility contract；
- RSD 代码的极限测试。

---

## 4. 建议的 native C++ 数据结构

### 4.1 二阶 \(f_{\rm NL}\) 多项式

建议新增内部轻量类型：

```cpp
struct FnlPoly2 {
  real c0 = 0.0;  // Gaussian
  real c1 = 0.0;  // coefficient of f_NL
  real c2 = 0.0;  // coefficient of f_NL^2
};
```

至少实现：

```cpp
FnlPoly2 operator+(FnlPoly2, FnlPoly2);
FnlPoly2 operator-(FnlPoly2, FnlPoly2);
FnlPoly2 operator*(FnlPoly2, FnlPoly2);  // truncate after c2
FnlPoly2 operator*(real, FnlPoly2);
FnlPoly2 reciprocal(FnlPoly2);           // require c0 != 0
FnlPoly2 operator/(FnlPoly2, FnlPoly2);
```

乘法必须是：

```cpp
out.c0 = a.c0*b.c0;
out.c1 = a.c0*b.c1 + a.c1*b.c0;
out.c2 = a.c0*b.c2 + a.c1*b.c1 + a.c2*b.c0;
```

这样 source、adaptive reconstruction denominator 和 primordial external legs 的所有 cross terms 均自动生成，避免手工维护数十个分量。

### 4.2 RSD 配置

```cpp
struct RsdTreeConfig {
  real f_growth = 0.0;
  Vec3 los = {0.0, 0.0, 1.0};

  enum class SelectionModel {
    none
  } selection = SelectionModel::none;

  enum class VelocityBiasModel {
    none
  } velocity_bias = VelocityBiasModel::none;
};
```

v1 对非 `none` 值应直接报错，不要静默忽略。

### 4.3 reconstruction 方法必须显式枚举

```cpp
enum class RsdReconstructionMethod {
  same_shift_no_linear_rsd_removal,
  // Future only; do not implement by aliasing the first method:
  kaiser_divided_same_shift,
  rec_iso,
  rec_sym
};
```

建议配置：

```cpp
struct RsdReconstructionConfig {
  RsdReconstructionMethod method;
  real smoothing_radius;
  real bias_recon;
  real cell_size;
  int cic_window_power;

  // Optional adaptive local-PNG denominator
  real f_nl_recon_ratio = 0.0;  // eta = f_NL^rec/f_NL
  real bphi_recon = 0.0;

  // Only used by future Kaiser-divided method
  real f_growth_recon = 0.0;
};
```

### 4.4 返回值

不建议复用语义可能含混的旧 `ComponentResult`。建议：

```cpp
struct FinitePngTreeComponents {
  real G = 0.0;
  real L = 0.0;
  real Q = 0.0;
};

struct RsdFinitePngTreeResult {
  FinitePngTreeComponents total;
  FinitePngTreeComponents gravitational_and_bias;
  FinitePngTreeComponents primordial;
  std::array<FinitePngTreeComponents, 3> cyclic_pairs;

  std::array<FnlPoly2, 3> z1;
  std::array<FnlPoly2, 3> z2;
};
```

保留 pair-level 输出非常重要：它能定位 permutation、LOS sign、pair-sum 和 reconstruction 错误。

---

## 5. 建议的新 native API

### 5.1 oriented pre-reconstruction

```cpp
RsdFinitePngTreeResult
compute_pre_recon_halo_bias_v1_local_png_tree_rsd_vectors(
    const std::array<Vec3, 3>& kvec,
    const std::array<real, 3>& p_linear,
    const std::array<real, 3>& transfer_m,
    const HaloBiasV1& bias,
    const RsdTreeConfig& rsd);
```

函数必须验证：

- 三条腿非零；
- \(\boldsymbol k_1+\boldsymbol k_2+\boldsymbol k_3=0\)；
- \(P_i>0\)、\(\mathcal M_i>0\)；
- LOS 非零并归一化；
- 所有输入有限。

### 5.2 oriented post-reconstruction

```cpp
RsdFinitePngTreeResult
compute_post_recon_halo_bias_v1_local_png_tree_rsd_vectors(
    const std::array<Vec3, 3>& kvec,
    const std::array<real, 3>& p_linear,
    const std::array<real, 3>& transfer_m,
    const HaloBiasV1& bias,
    const RsdTreeConfig& rsd,
    const RsdReconstructionConfig& recon);
```

v1 只接受：

```text
same_shift_no_linear_rsd_removal
```

其他 enum 先报 `not implemented`，避免错误模型静默运行。

### 5.3 multipole projector

continuum oracle：

```cpp
FinitePngTreeComponents
project_rsd_tree_legendre_multipole(
    real k1, real k2, real k3,
    int ell,
    const SpectrumAccessor& spectra,
    const HaloBiasV1& bias,
    const RsdTreeConfig& rsd,
    const OptionalReconConfig& recon,
    int n_mu,
    int n_phi);
```

periodic box production 优先另设：

```cpp
FinitePngTreeComponents
project_rsd_tree_exact_lattice(
    const TriangleModeList& modes,
    ...);
```

二者不得共享一个含糊的 `project_monopole` 名称，因为 normalization 与 mode weighting 不同。

---

## 6. 核心实现伪代码

### 6.1 \(Z_1\)

```cpp
FnlPoly2 z1_rsd(
    const Vec3& k,
    real M,
    const HaloBiasV1& bias,
    const RsdTreeConfig& rsd) {
  const real mu = dot(unit(k), rsd.los);
  return {
      .c0 = bias.b1 + rsd.f_growth * mu * mu,
      .c1 = bias.bphi / M,
      .c2 = 0.0,
  };
}
```

### 6.2 real-space \(K_2\)

直接重用 current MARISA-B 的 bias convention：

```cpp
FnlPoly2 k2_real_png(...){
  c0 = b1*F2 + 0.5*b2 + bK2*S2;
  c1 = 0.5*(bphi*Aadv + bphidelta*(1/M1 + 1/M2));
  c2 = 0.5*bphi2/(M1*M2);
}
```

### 6.3 pre-recon \(Z_2\)

```cpp
FnlPoly2 z2_rsd(...) {
  const FnlPoly2 z1a = z1_rsd(k1, M1, bias, rsd);
  const FnlPoly2 z1b = z1_rsd(k2, M2, bias, rsd);
  const Vec3 K = k1 + k2;

  FnlPoly2 out = k2_real_png(...);
  out += f_growth * square(dot(unit(K), los)) * G2(k1,k2);
  out += 0.5 * f_growth * dot(K,los)
       * (mu1/norm(k1) * z1b + mu2/norm(k2) * z1a);
  return out;
}
```

注意：mapping 中使用的是 `z1a/z1b` 多项式，不是只含 \(b_1+f_g\mu^2\) 的 Gaussian scalar。

### 6.4 post-recon shift block

current method：

```cpp
FnlPoly2 shift_block(
    const Vec3& K,
    const Vec3& p,
    real M_p,
    const RsdReconstructionConfig& recon) {
  const real numerator = -dot(K,p)/dot(p,p) * Wrec(p,recon);
  FnlPoly2 denominator{
      .c0 = recon.bias_recon,
      .c1 = recon.f_nl_recon_ratio * recon.bphi_recon / M_p,
      .c2 = 0.0,
  };
  return numerator / denominator;
}
```

post kernel：

```cpp
z2_rec = z2_pre
       + 0.5*(shift_i + shift_j)*z1_i*z1_j;
```

这会自动恢复 current fixed-\(b_{\rm rec}\) basis，并产生 RSD 后 adaptive denominator 的新角结构。

### 6.5 tree bispectrum

```cpp
FnlPoly2 grav{};
for (auto [i,j,k] : cyclic_pairs) {
  const FnlPoly2 pair = 2*P[i]*P[j]*z1[i]*z1[j]*z2[i,j];
  grav += pair;
}

const real B0hat = primordial_template(P,M);
const FnlPoly2 primordial = FnlPoly2{0.0,B0hat,0.0}
                          * z1[0]*z1[1]*z1[2];
const FnlPoly2 total = grav + primordial;
```

严禁先在某个有限 \(f_{\rm NL}\) 上求值再用 finite difference 恢复 \(L,Q\)。

---

## 7. projection 层设计

### 7.1 continuum monopole/multipole oracle

固定三角形边长，使用 \((\mu_1,\phi)\) 构造 Cartesian triangle：

```text
mu1 in [-1,1]
phi in [0,2pi]
```

对每个 orientation 调用 oriented kernel，然后积分：

\[
B_\ell=\frac{2\ell+1}{4\pi}
\int d\mu_1d\phi\,\mathcal L_\ell(\mu_1)B^s.
\]

建议默认 quadrature 起点：

```text
n_mu = 18
n_phi = 20
```

但生产值必须做 convergence scan；不应把 oracle 默认值当作普适保证。

### 7.2 periodic box exact lattice

对 MARISA-B 的 periodic box，首选流程：

```text
actual mode triplet
  -> k-vectors + fixed box LOS
  -> oriented G,L,Q
  -> optional oriented IR damping
  -> estimator weight
  -> exact mode-count normalization
```

优点：

- 自动包含 finite box anisotropy；
- 与实际 estimator 完全匹配；
- CIC window 的方向依赖不丢失；
- 无需假设 continuum orientation distribution。

### 7.3 TriPoSH 与普通 Legendre multipoles

Python oracle 实现的是以 \(\boldsymbol k_1\) 为轴的常用 Legendre/Scoccimarro \(B_\ell\)。它不自动等于 estimator 的 TriPoSH \(B_{\ell_1\ell_2L}\)。

生产实现必须：

- 读取 estimator 的明确定义；
- 用相同 spherical harmonics、phase 和 normalization；
- 通过 isotropic real-space 极限与已知 analytic normalization 回归。

---

## 8. stochastic 层

### 8.1 Gaussian RSD raw bases

第一版建议返回三个不含 nuisance amplitude 和 \(\bar n\) 的 raw shapes：

```cpp
struct GaussianRsdStochasticBases {
  real Bshot_basis;
  real Pshot_basis;
  real Ashot_basis;
};
```

其中：

\[
S_B=\sum_iP_i b_1(b_1+f_g\mu_i^2),
\]

\[
S_P=\sum_iP_i f_g\mu_i^2(b_1+f_g\mu_i^2),
\qquad S_A=1.
\]

likelihood 层再组装：

\[
B_{\rm stoch}=B_{\rm shot}S_B/\bar n
+P_{\rm shot}S_P/\bar n
+A_{\rm shot}S_A/\bar n^2.
\]

### 8.2 finite-PNG stochastic

v1 不应给出未经推导的唯一 closure。可选策略：

1. `png_stochastic = off`：最保守；
2. `png_stochastic = legacy_real_space`：保留 current real-space shape，清楚标记其未含完整 RSD；
3. 未来建立基于 redshift-space stochastic operator expansion 的独立模块。

默认建议为 1，并在 mock closure 中比较 1/2 的影响。

---

## 9. 与 IR resummation 的接口

RSD source core 不应内部计算 \(\Sigma\)。建议返回 oriented source 与必要 kernel audit 信息，交给 IR wrapper：

```text
oriented triangle
 -> RSD source polynomial G,L,Q
 -> legwise/tensor RSD IR dressing
 -> post-recon displacement covariance（如适用）
 -> projection
```

pre-recon 最低阶 covariance 可写成：

```cpp
Mat3 R = I + f_growth * outer(los,los);
Mat3 Sigma_s = sigma2 * R * transpose(R);
D_i = exp(-dot(k_i, Sigma_s*k_i));
```

post-recon 不能复用一个随意缩小的标量 \(\Sigma\)；必须使用 reconstruction method 对应的 residual displacement tensor。此项应在之前 post-IR 模块基础上扩展，而不是塞入 `z2_rsd()`。

---

## 10. AP、window、wide-angle 的模块边界

建议调用顺序：

```text
fiducial observed bin/mode
 -> AP remapping to trial physical k-vectors
 -> RSD source
 -> IR dressing
 -> multipole/bin projection
 -> survey window convolution
```

periodic box v1 全部关闭 AP/window/wide-angle。

真实 survey 后续必须增加：

- AP Jacobian 和 \(k_\parallel,k_\perp\) remapping；
- estimator-specific local LOS；
- wide-angle expansion；
- LOS selection bias operators；
- survey window matrix。

这些不是 `Z2` 内部的 if/else。

---

## 11. 建议修改的文件

### 11.1 native core

建议新增：

```text
src/marisa_b/finite_png_poly2.h
src/marisa_b/tree_png_rsd.h
src/marisa_b/tree_png_rsd.cpp
src/marisa_b/tree_png_rsd_projection.h
src/marisa_b/tree_png_rsd_projection.cpp
```

或者将 kernel 接入现有 `marisa_b_native.*`，但仍建议把 projector 独立成文件，避免 native source 继续膨胀。

### 11.2 Python binding

暴露：

```text
pre_tree_rsd_vectors(...)
post_tree_rsd_vectors(...)
project_tree_rsd_multipole(...)
gaussian_rsd_stochastic_bases(...)
```

返回字段至少包含：

```text
total.G/L/Q
gravity_bias.G/L/Q
primordial.G/L/Q
pair_terms[3]
z1[3]
z2[3]
```

### 11.3 production runner

新增显式配置：

```yaml
rsd:
  enabled: true
  f_growth: ...
  los: [0, 0, 1]
  plane_parallel: true
  selection_model: none
  velocity_bias_model: none

projection:
  kind: exact_lattice   # or continuum_legendre_oracle
  multipoles: [0]

reconstruction:
  method: same_shift_no_linear_rsd_removal

fog:
  enabled: false
```

runner 启动时打印/保存全部 provenance。

---

## 12. 缓存键与 provenance

RSD 后 cache key 不能只有 \((k_1,k_2,k_3)\)。至少包含：

```text
three Cartesian k-vectors or orientation identifier
LOS vector
f_growth
bias parameters and basis convention
f_NL order
pre/post flag
reconstruction method
R, b_rec, CIC cell/power
adaptive denominator parameters
projection definition and quadrature/mode-list hash
IR covariance convention/hash
stochastic model enum
AP/window state
code commit and spectrum-table hash
```

如果 projection 在 oriented evaluation 后进行，可缓存 kernel 级结果；若 exact lattice mode list 改变，projection cache 必须失效。

---

## 13. 测试矩阵

### 13.1 kernel 单元测试

必须覆盖：

1. `f_growth=0` 精确恢复 current real-space pre \(G,L,Q\)；
2. PNG biases=0 时 \(Z_1,Z_2\) 与标准 Gaussian RSD 公式一致；
3. \(Z_2.c1\) 包含 nonlinear mapping 的 RSD–\(b_\phi\) cross term；
4. `bphi2` 与 `bphi*bphi` 不混淆；
5. primordial \(L,Q\) external-leg 系数正确；
6. 全局同时旋转三角形和 LOS 时结果不变；
7. cyclic/permutation invariance；
8. LOS 取反时 parity-even tree 结果不变；
9. squeezed、equilateral、folded-nearby、generic scalene triangles；
10. 非闭合或零向量输入失败。

### 13.2 post-recon 测试

1. \(Z_{1,\rm rec}=Z_1\)；
2. fixed-\(b_{\rm rec}\) 与显式 same-shift 公式一致；
3. \(R\to\infty\) 或 \(W\to0\) 回到 pre；
4. adaptive denominator 的 \(c0,c1,c2\) 与独立 finite difference 一致；
5. CIC off 时同时旋转三角形、LOS 后不变；
6. CIC on 时相对网格旋转可改变结果，但同时旋转网格定义后应恢复；
7. 当前 legacy real-space adaptive bases 在 \(f_g=0\) 下恢复。

### 13.3 projection 测试

1. \(f_g=0\) 时 monopole 等于 orientation-independent real-space值；
2. quadrature 收敛：提高 \(n_\mu,n_\phi\) 后 \(G,L,Q\) 稳定；
3. exact lattice 与高分辨率 continuum 在大盒子/宽 bin 极限趋近；
4. analytic Kaiser-only toy model 的角平均；
5. Legendre odd multipoles在 parity-symmetric global-LOS setup 下为零（数值容差内）；
6. TriPoSH projector 与 estimator normalization 的独立小样本回归。

### 13.4 stochastic 测试

1. \(f_g=0\)：\(S_B=b_1^2\sum_iP_i\)，\(S_P=0\)；
2. LOS 取反不变；
3. permutation invariance；
4. nuisance amplitude 和 \(\bar n\) 只在 likelihood 层组装。

### 13.5 IR 组合测试

1. \(\Sigma\to0\) 回到 undamped RSD source；
2. \(f_g\to0\) 回到 real-space IR；
3. damping 使用实际 oriented \(\mu_i\)；
4. projection 前后次序不能交换的回归测试；
5. post-RSD covariance 未配置时禁止静默使用 pre covariance。

---

## 14. 建议验收阈值

对纯解析双精度 kernel：

```text
relative tolerance: 1e-12 至 1e-11
absolute floor:     1e-12 或按量纲设定
```

对 finite-difference cross-check：

```text
relative tolerance: 1e-7 至 1e-6
```

对 continuum quadrature：

```text
G/L/Q 分别检查相邻 quadrature order 的收敛；
不要只检查总 B(f_NL)，以免 cancellation 掩盖错误。
```

production acceptance 不应只看 unit tests，还必须包括：

- Gaussian RSD mock closure；
- finite-PNG \(f_{\rm NL}=0,\pm100\) response closure；
- pre/post 同一 estimator 与 covariance；
- \(k_{\max}\) scan；
- stochastic/FoG nuisance 开关对 \(f_{\rm NL}\) 的稳定性；
- exact lattice 与 continuum projector 的差异。

---

## 15. 分阶段实施路线

### Phase A：Python oracle（本交付已完成）

- oriented pre \(Z_1,Z_2\)；
- finite-PNG \(G,L,Q\)；
- current same-shift post；
- adaptive denominator；
- continuum multipole projector；
- Gaussian DESI-style stochastic raw bases；
- pre-RSD IR covariance helper；
- 22 个专项测试。

### Phase B：native kernel 移植

- 实现 `FnlPoly2`；
- 新增 oriented pre/post APIs；
- 对 Python oracle 做随机三角形逐项回归；
- 不接 production likelihood。

### Phase C：projector 与 estimator 对齐

- continuum projector；
- periodic exact-lattice projector；
- estimator normalization/TriPoSH 回归；
- CIC orientation cache。

### Phase D：IR 组合

- pre 各向异性 RSD IR；
- post residual displacement tensor；
- strict one-wiggle 或既定 IR scheme；
- source–IR–projection 顺序回归。

### Phase E：mock closure

依次测试：

```text
Gaussian pre
Gaussian post
finite-PNG pre
finite-PNG post
```

每一步先做 oriented/kernel closure，再做 projected data vector closure。

### Phase F：可选 nuisance

仅当 strict tree 在选定 \(k_{\max}\) 出现有结构的 residual 时，才评估：

- \(k^2\) RSD counterterm；
- empirical FoG；
- additional stochastic operators。

这些作为单独 model tier，不修改 strict-tree 定义。

---

## 16. 本交付 Python oracle 的 API

### 16.1 oriented pre/post evaluation

```python
from marisa_b.tree_png_rsd_reference import (
    HaloBiasV1,
    ReconstructionConfig,
    RsdConfig,
    finite_png_halo_bispectrum_rsd,
)

result = finite_png_halo_bispectrum_rsd(
    vectors=k_vectors,              # shape (3,3), closed
    power_linear=[P1, P2, P3],
    transfer_m=[M1, M2, M3],
    bias=HaloBiasV1(...),
    rsd=RsdConfig(f_growth=f_growth, los=(0,0,1)),
    reconstruction=None,            # pre
)

post = finite_png_halo_bispectrum_rsd(
    ...,
    reconstruction=ReconstructionConfig(
        smoothing_radius=15.0,
        bias_recon=2.2,
        denominator_mode="bias_only",
    ),
)

B = result.total.predict(f_nl)
```

### 16.2 continuum monopole

```python
from marisa_b.tree_png_rsd_reference import project_scoccimarro_multipole

b0 = project_scoccimarro_multipole(
    k1, k2, k3,
    [P1, P2, P3],
    [M1, M2, M3],
    bias,
    rsd,
    ell=0,
    n_mu=18,
    n_phi=20,
    reconstruction=None,
)
```

### 16.3 Gaussian stochastic bases

```python
bases = desi_gaussian_tree_stochastic_bases(
    k_vectors, [P1,P2,P3], b1, rsd
)

B_stoch = (
    B_shot / nbar * bases.Bshot_basis
    + P_shot / nbar * bases.Pshot_basis
    + A_shot / nbar**2 * bases.Ashot_basis
)
```

---

## 17. 运行命令

在本交付目录：

```bash
PYTHONPATH=python \
python -m pytest tests/python/test_tree_png_rsd_reference.py -q
```

运行示例：

```bash
PYTHONPATH=python \
python examples/tree_png_rsd_example.py
```

静态语法检查：

```bash
python -m compileall -q python tests examples
```

---

## 18. 对另一代码审阅者的最小验收任务

建议让执行者按以下顺序报告：

1. 记录 Python、NumPy、pytest 版本和 commit/patch hash；
2. 运行全部 22 项专项测试；
3. 对至少 1000 个随机闭合三角形比较：
   - Python oracle；
   - native C++ pre；
   - native C++ post；
   - 每个结果分别比较 \(G,L,Q\)，不能只比较有限 \(f_{\rm NL}\) 总和；
4. 单独扫描 squeezed triangles，因为 \(1/\mathcal M\) 增强最明显；
5. 检查 LOS 旋转、LOS 反向、permutation；
6. 对 monopole 做 quadrature convergence；
7. 若接 exact lattice，报告 continuum–lattice 差值随 box/bin 的变化；
8. 不得因为 unit tests 通过就宣称 post-recon survey production-ready。

---

## 19. 当前交付的明确边界

已覆盖：

```text
tree-level deterministic halo bispectrum
finite local PNG through f_NL^2
plane-parallel/global LOS RSD
pre reconstruction
current MARISA-B same-shift post reconstruction
adaptive local-PNG reconstruction denominator through f_NL^2
continuum Legendre multipoles
Gaussian RSD stochastic raw bases
pre-RSD IR covariance helper
```

未覆盖：

```text
one-loop
GEO-FPT
nonlinear P(k) substitution
FoG/counterterm as fiducial
survey selection operators
velocity bias
wide angle
AP/window production wrapper
unique finite-PNG RSD stochastic closure
post-reconstruction RSD IR covariance completion
TriPoSH estimator normalization
native C++ integration and full repository build
```

这些边界应随代码一同保留，防止参考 oracle 被误标记为完整 survey likelihood。

---

## 20. 参考文献

1. Scoccimarro, Couchman & Frieman, *The Bispectrum as a Signature of Gravitational Instability in Redshift Space*, ApJ 517 (1999), arXiv:astro-ph/9808305.
2. Desjacques, Jeong & Schmidt, *The Galaxy Power Spectrum and Bispectrum in Redshift Space*, JCAP 12 (2018) 035, arXiv:1806.04015.
3. Ivanov et al., *Precision analysis of the redshift-space galaxy bispectrum*, Phys. Rev. D 105, 063512 (2022), arXiv:2110.10161.
4. Shirasaki et al., *Constraining Primordial Non-Gaussianity with Post-reconstructed Galaxy Bispectrum in Redshift Space*, Phys. Rev. D 103, 023506 (2021), arXiv:2010.04567.
5. Sugiyama, *Developing a Theoretical Model for the Resummation of Infrared Effects in the Pre- and Post-Reconstruction Galaxy Bispectra*, arXiv:2403.18262.
6. DESI Collaboration, *Full-Shape analysis of the power spectrum and bispectrum of DESI DR1 LRG and QSO samples*, arXiv:2503.09714.
7. Philcox et al., *Reanalyzing DESI DR1: 1. ΛCDM Constraints from the Power Spectrum & Bispectrum*, arXiv:2507.13433.
8. Barreira, *On the impact of galaxy bias uncertainties on primordial non-Gaussianity constraints*, JCAP 12 (2022) 031, arXiv:2107.06887.
9. Moradinezhad Dizgah et al., finite local-PNG halo bispectrum bias expansion, arXiv:2010.14523.
