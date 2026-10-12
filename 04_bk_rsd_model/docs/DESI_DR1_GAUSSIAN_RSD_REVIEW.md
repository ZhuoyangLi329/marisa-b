# DESI DR1 PNG 分析中的 Gaussian bispectrum RSD 建模调研

更新：2026-09-30

## 结论先行

主人提到的文章是 Chudaykin、Ivanov、Philcox 的 **Reanalyzing DESI DR1: 3. Constraints on Inflation from Galaxy Power Spectra & Bispectra**（arXiv:2512.04266）。它是独立重分析；官方 DESI DR1 PNG 论文主要使用 power spectrum 的 scale-dependent bias，并没有这套 bispectrum RSD 模型。

核心不是给 real-space bispectrum 乘一个 Kaiser 因子或全局 FoG 阻尼，而是从 redshift-space coordinate mapping 逐阶生成 Z_n kernel，再做 loop、counterterm、stochastic、IR、AP 和 multipole projection。

## 1. Tree-level Gaussian RSD

采用 plane-parallel approximation。令 \(\mu_i=\hat{\boldsymbol{k}}_i\cdot\hat{\boldsymbol{z}}\)，\(\boldsymbol{k}_{12}=\boldsymbol{k}_1+\boldsymbol{k}_2\)，\(\mu_{12}=\hat{\boldsymbol{k}}_{12}\cdot\hat{\boldsymbol{z}}\)，则

\[
Z_1(\boldsymbol{k})=b_1+f\mu^2.
\]

写 \(K_1=b_1\)，

\[
K_2(\boldsymbol{k}_1,\boldsymbol{k}_2)=b_1F_2+\frac{b_2}{2}+b_{\mathcal G_2}\kappa_{12},
\qquad
\kappa_{12}=(\hat{\boldsymbol{k}}_1\cdot\hat{\boldsymbol{k}}_2)^2-1.
\]

然后

\[
\begin{aligned}
Z_2={}&K_2+f\mu_{12}^2G_2
+\frac{f\mu_{12}k_{12}}{2}K_1
\left(\frac{\mu_1}{k_1}+\frac{\mu_2}{k_2}\right)\\
&+\frac{(f\mu_{12}k_{12})^2}{2}
\frac{\mu_1}{k_1}\frac{\mu_2}{k_2}.
\end{aligned}
\]

Gaussian tree bispectrum 为

\[
B_{211}^{s}=2Z_2(\boldsymbol{k}_1,\boldsymbol{k}_2)Z_1(\boldsymbol{k}_1)Z_1(\boldsymbol{k}_2)P_L(k_1)P_L(k_2)+2\,\mathrm{cyc.}
\]

这已经包含完整 tree-level RSD 角度结构，不能替换成 real-space bispectrum 再乘一个经验 Kaiser/FoG 因子。

当前 MARISA-B 使用 \(b_{K^2}(\mu_{12}^2-1/3)\)，而 DESI/COBRA 文献使用 \(b_{\mathcal G_2}[(\hat k_1\cdot\hat k_2)^2-1]\)，必须在 adapter 中明确转换。

## 2. FoG

DESI tree bispectrum 没有采用 phenomenological global FoG factor。FoG 被作为 redshift-space EFT counterterm 处理。Paper I 的 tree implementation 保留主要的 \(k^2\mu^2\) 项：

\[
Z_1\rightarrow Z_1^{\rm FoG}
=b_1+f\mu^2-c_1\mu^2\left(\frac{k}{k_{\rm NL}^{r}}\right)^2,
\]

其中 \(k_{\rm NL}^{r}\simeq0.3\,h/{\rm Mpc}\)。只拟合 monopole 时 \(k^2\mu^4\) 与 \(k^2\mu^2\) 高度退化，常设 \(c_2=0\)；加入 bispectrum quadrupole 时应保留更完整的 \(c_1,c_2\) 结构。

## 3. Stochastic 与 mixed terms

Paper I 在 Poisson shot noise 已从估计量中减去的约定下使用

\[
B_{\rm stoch}= \frac{b_1^2P_{\rm tree}(k_1)}{\bar n}
\left[B_{\rm shot}+\beta\mu_1^2(P_{\rm shot}+B_{\rm shot})
+\beta^2\mu_1^4P_{\rm shot}\right]
+\mathrm{cyc.}+\frac{A_{\rm shot}}{\bar n^2},
\]

其中 \(\beta=f/b_1\)。\(P_{\rm shot}\) 与 power spectrum 共用，\(B_{\rm shot}\)、\(A_{\rm shot}\) 是 bispectrum 残差参数，Poisson limit 对应三者趋近于零。

完整 COBRA one-loop 模型进一步加入 anisotropic stochastic 与 deterministic-stochastic mixed tree/one-loop/counterterm。其 power counting 显示 mixed tree 与 Gaussian tree 同阶，不能只放一个常数 shot-noise。

## 4. IR resummation

RSD 下的 BAO damping 是各向异性的：

\[
P_{\rm IR}(k,\mu)=P_{\rm nw}(k)+P_{\rm w}(k)e^{-k^2\Sigma_{\rm tot}^2(\mu)},
\]

\[
\Sigma_{\rm tot}^2(\mu)=\Sigma^2[1+f(2+f)\mu^2]
+\delta\Sigma^2 f^2\mu^2(\mu^2-1),
\]

\[
\Sigma^2=\frac1{6\pi^2}\int_0^{\Lambda_{\rm IR}}dq\,P_{\rm nw}(q)
[1-j_0(qr_s)+2j_2(qr_s)],
\]

\[
\delta\Sigma^2=\frac1{2\pi^2}\int_0^{\Lambda_{\rm IR}}dq\,P_{\rm nw}(q)j_2(qr_s).
\]

早期 tree analysis 使用约 \(k_S=0.05\,h/{\rm Mpc}\)，COBRA one-loop implementation 使用 \(\Lambda_{\rm IR}=0.1\,h/{\rm Mpc}\)。这应作为明确的 theory-systematics choice。

纯 tree-level LO 实现不应提前加入 tree+loop matching factor；完整 one-loop 模型再单独处理 matching。

## 5. Multipoles、AP 与 binning

固定 \(\boldsymbol{k}_1\) 为极轴，以 \(\mu=\hat{\boldsymbol{k}}_1\cdot\hat{\boldsymbol{z}}\) 和方位角 \(\phi\) 积分，\(\mu_2,\mu_3\) 由闭合 triangle 几何得到：

\[
B_\ell=(2\ell+1)\int_0^{2\pi}\frac{d\phi}{2\pi}
\int_{-1}^{1}\frac{d\mu}{2}
B^s(k_1,k_2,k_3,\mu_i)\mathcal L_\ell(\mu).
\]

DESI 一般 galaxy/QSO 样本使用 \(B_0,B_2\)，QSO-all 为 local PNG 只使用大尺度 \(B_0\)。

AP 同时改变每条腿长度和 line-of-sight cosine：

\[
k_i'=k_i\left(\frac{\mu_i^2}{\alpha_\parallel^2}
+\frac{1-\mu_i^2}{\alpha_\perp^2}\right)^{1/2},
\]

\[
\mu_i'=\frac{\mu_i}{\alpha_\parallel}
\left(\frac{\mu_i^2}{\alpha_\parallel^2}
+\frac{1-\mu_i^2}{\alpha_\perp^2}\right)^{-1/2}.
\]

有限 bin 中还要对三条腿做积分、保留 triangle indicator，并乘 Jacobian \((\alpha_\parallel^2\alpha_\perp^4)^{-1}\)，不能只在 bin center 上计算 multipole。

## 6. DESI DR1 PNG 中的层级

- 普通 galaxies/QSO：Gaussian bispectrum 使用 one-loop EFT，主分析 \(B_0+B_2\)，baseline \(k_{\max}^{B}\simeq0.16\,h/{\rm Mpc}\)。
- tree-level 对照：\(k_{\max}^{B}=0.08\,h/{\rm Mpc}\)。
- QSO-all：使用 large-scale tree EFT，\(P_0,P_2+B_0\)，到 \(k_{\max}=0.10\,h/{\rm Mpc}\)。
- PNG 只保留 leading \(B_{111}\) 与 local PNG bias 产生的 \(B_{112}^{f_{\rm NL}}\)，省略 \(B_{221}\)、\(B_{311}\) 等 PNG one-loop corrections。
- 因此 “one-loop DESI PNG analysis” 主要表示 Gaussian/RSD EFT background 达到 one-loop，PNG sector 本身仍是 leading order。

## 7. 对第四部分的实现建议

第一阶段只实现可审计的 Gaussian tree RSD：

1. orientation-aware closed-triangle object，输入三条 \(\boldsymbol{k}_i\) 与 line of sight。
2. 实现 \(K_1,K_2,Z_1,Z_2\)，并加入 tidal convention adapter。
3. 计算 deterministic \(B_{211}\)、\(c_1\) FoG counterterm 和 \(P_{\rm shot},B_{\rm shot},A_{\rm shot}\) residual stochastic。
4. 实现各向异性 \(P_{\rm IR}(k,\mu)\) 和 \(B_0,B_2\) 投影。
5. 加入 AP、有限-bin triangle indicator，用 direct numerical average 验证 bin-center 近似误差。
6. 先做 Gaussian \(f_{\rm NL}=0\) mock recovery，再接入 PNG \(B_{111}\)、\(b_\phi\) 和 \(b_{\delta\phi}\)。

第二阶段才考虑 \(Z_3,Z_4\)、四个 Gaussian one-loop diagrams、完整 45-parameter EFT 和 mixed stochastic。这能避免把 tree-level RSD 的实现困难与 one-loop parameter explosion 同时引入。

## 参考文献

- https://arxiv.org/abs/2512.04266
- https://arxiv.org/html/2507.13433
- https://arxiv.org/html/2507.22110
- https://arxiv.org/html/2110.10161
- https://arxiv.org/html/2302.04414
