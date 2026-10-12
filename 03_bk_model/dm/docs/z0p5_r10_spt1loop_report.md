# Quijote DM z=0.5, R=10: one-loop finite-fNL recovery

更新日期：2026-09-08。当前状态：最终 matched G/L/Q 模型、四个 kmax 的 profile 与 MCMC recovery、同掩膜 Fisher 对照、数值收敛检查和模型审计均已完成。

## 1. 数据和 likelihood 口径

- Quijote dark matter，z=0.5，重构平滑尺度 R=10 h^-1 Mpc。
- 三个 cosmology 使用共同的 500 个 realization，ID 为 0..499；没有重新运行模拟。
- 数据向量为 diagonal B000(k,k)。按要求排除前两个 diagonal bin，即 k=0.0154351、0.0331589 h Mpc^-1。
- kmax=0.08、0.10、0.15、0.20 分别使用 2、3、5、8 个 bin。
- covariance 是每个阶段各自的 500 个单盒样本 covariance，使用维数相关 Hartlap 修正，V=1 (h^-1 Gpc)^3；不是 500 盒均值的 covariance。
- 固定宇宙学，无 nuisance parameter。MCMC prior 为 -1000<fNL<1000 的均匀先验。

## 2. 最终理论模型

likelihood 使用

    B(fNL) = G + fNL L + fNL^2 Q.

G 包含 Btree、B222、B321I、B321II、B411。L 包含 primordial B111 以及 B122I、B122II、B113I、B113II。Q 为 local-PNG primordial trispectrum 对应的 B112II 系数。pre 和 post 均使用上述完整 G/L/Q；post 没有沿用只适用于 Gaussian initial condition 的 Sugiyama 截断式。

`MODEL_AUDIT.json` 显式验证：四个 Gaussian loop 项重建 B1loop，四个线性 PNG loop 项重建 dB/dfNL 的一环部分，B111 加上线性 loop 重建 L，B112II 在 pre/post 中均为有限非零并实际进入 likelihood。各分量闭合的最大相对残差低于 1.2e-15。

最终 pre 的 positions 2、3、4 使用严格 loopcheck 模板（Gaussian epsrel=0.002、PNG QMC power=14）；其余位置使用 fine 模板。post 的 n_mu、n_rad 和 loopcheck 加密均远低于预设的 0.02 单盒 sigma 门槛，因此保留统一 fine 模板。

z=0.5 的 Plin/Pphi/M 表由当前脚本重新导出；它与归档的 z=0.5 四列表逐元素完全相同，SHA256 为 `c748ba2bdd217581071513fe918d93e5a08b48390451c69af60159d3dc73feeb`。

## 3. MCMC recovery

下表的中心和区间为 input fNL=0 时的 posterior median 与 equal-tail 68% CI；提升为同一 k mask 下 pre/post 半宽之比。

| kmax | Nbin | pre median [16%,84%] | post median [16%,84%] | pre/post 半宽 | 三个 truth 落入 68% CI (pre/post) |
|---:|---:|---:|---:|---:|---:|
| 0.08 | 2 | -34.9 [-193.0,+118.6] | +17.8 [-123.1,+160.9] | 1.097 | 3/3, 3/3 |
| 0.10 | 3 | -64.4 [-184.9,+53.9] | +8.4 [-98.5,+116.6] | 1.110 | 3/3, 3/3 |
| 0.15 | 5 | -152.9 [-231.1,-77.2] | -96.2 [-162.4,-30.5] | 1.167 | 0/3, 0/3 |
| 0.20 | 8 | -268.2 [-327.5,-209.9] | -249.9 [-291.6,-208.5] | 1.417 | 0/3, 0/3 |

结论：kmax=0.08、0.10 时 post-recon 对 input=-100、0、+100 均完成 68% recovery，且 input-output slope 分别为 1.043、1.031。kmax=0.15、0.20 的误差虽然更窄，但 pre/post 都出现显著共同负偏差；不能把这两个 cut 当成可靠 recovery 或无偏约束。

24 个 posterior 全部通过诊断：最大 rank-normalized split-Rhat=1.00729，最小 bulk ESS=14171，最小 tail ESS=28098，最小 retained length/tau=260.7。另用 200001 点一维确定性积分复核后验分位数，MCMC 与确定性结果的最大差为 0.0200 posterior sigma。

## 4. 同掩膜 Fisher 与 MCMC

| kmax | Fisher sigma pre/post | Fisher 提升 | MCMC 提升 | 仅 covariance 的反事实提升 | 仅 response 的反事实提升 |
|---:|---:|---:|---:|---:|---:|
| 0.08 | 155.1 / 141.4 | 1.097 | 1.097 | 1.621 | 0.685 |
| 0.10 | 118.3 / 108.6 | 1.090 | 1.110 | 1.800 | 0.616 |
| 0.15 | 75.3 / 66.4 | 1.134 | 1.167 | 2.069 | 0.549 |
| 0.20 | 55.6 / 40.2 | 1.384 | 1.417 | 2.434 | 0.575 |

同一个 z、R、diagonal mask、排除 bin、模型和 covariance 口径下，local Fisher 与 MCMC 宽度只差约 0.4% 到 5.5%。因此旧 Fisher 图比原 MCMC 显示更大提升，不是 Fisher 算法本身造成的，主要是统计量和 mask 不同：旧图使用完整上三角 B000(k1,k2)，包含大量非对角/挤压形状及前两个低 k bin；当前 recovery 只用 diagonal 且排除了前两个 bin。

反事实拆分还表明，post covariance 单独可带来约 1.6--2.4 倍改善，但 post 的理论 fNL response 在固定 pre covariance 下反而较弱，只保留约 0.55--0.68 的改善因子，抵消了大量 covariance 收益。旧完整上三角形状对 post covariance 的利用更强，所以净 Fisher 提升更大。

## 5. 偏差来源

将模拟和理论差异投影到实际 fNL 方向，得到：

| kmax | post odd response / theory L | post Gaussian 残差等效 fNL | 10000 Q 等效 fNL |
|---:|---:|---:|---:|
| 0.08 | 1.0326 | +17.64 | -0.15 |
| 0.10 | 1.0207 | +8.42 | +0.02 |
| 0.15 | 0.9902 | -95.58 | +0.51 |
| 0.20 | 0.9732 | -246.38 | +0.92 |

Gaussian 残差等效 fNL 与 MCMC 的共同 offset 几乎逐项一致；odd PNG response 在所有 cut 都接近 1。Q 必须保留以保证有限 fNL 模型完整，但在 |fNL|=100 时只投影为约 0--1 个等效 fNL，无法解释 -96 或 -250 的高 k 偏差。

因此当前 DM 模型的主要限制不是缺少已知的 fNL 线性 loop 项，也不是遗漏 fNL^2 B112II，而是 k>=0.1 h Mpc^-1 后 Gaussian one-loop baseline 与测量均值的失配。下一步若要扩大 kmax，应优先改进 Gaussian sector，包括两环/EFT counterterm、离散格点和 CIC/alias/重构传递函数匹配；不应继续用额外 PNG 项或经验 gain 吸收共同 offset。

## 6. 数值和实现审计

- Slurm convergence job 58070906：COMPLETED 0:0，墙钟 24:31；四个独立任务全部无积分警告。
- post n_mu 24->48 的最大 recovery 中心漂移为 0.65 fNL（0.0060 单盒 sigma）。
- post n_rad 3->5 的最大漂移为 0.41 fNL（0.0029 sigma）。
- post epsrel 0.01->0.002、QMC power 12->14 的最大漂移为 0.18 fNL（0.0017 sigma）。
- pre 严格 low-k 模板已被采用到最终 bundle，所以对应残余漂移为 0。
- 来源 binary、native source、谱表、simulation、组装 JSON 的 SHA256 全部匹配；NPZ 在有效 positions 与 JSON 逐元素完全一致。
- synthetic G+fNL L+fNL^2 Q 对 input=-100、0、+100 的 recovery 闭合通过。
- 最终化 job 58072417：COMPLETED 0:0，墙钟 05:50。

## 7. 交付文件

- 主 recovery PDF：`9.11meeting/dm/quijote_dm_z0p5_r10_1loop_b000_fnl_profile_matched_finite_v1.pdf`。
- 同掩膜 Fisher/MCMC PDF：`9.11meeting/dm/quijote_dm_z0p5_r10_diagonal_fisher_mcmc_prepost_comparison_v1.pdf`。
- 两页模型审计 PDF：`9.11meeting/dm/quijote_dm_z0p5_r10_matched_model_audit_v1.pdf`。
- 清理后的旧 Fisher PDF：`9.11meeting/dm/quijote_dm_fnl_fig5_prepost_sim_tree_marisa1loop_kmax0p3.pdf`。为保持会议引用路径，文件名仍含 tree，但当前图中已移除所有 tree-level 曲线。
- 可机读审计：`MODEL_AUDIT.json`、`quijote_dm_z0p5_r10_1loop_b000_fnl_mcmc_v1.json`、`quijote_dm_z0p5_r10_diagonal_fisher_mcmc_comparison_v1.json`。

从头复现时依次提交：

```bash
base=$(sbatch --parsable codes/submit_quijote_dm_z0p5_r10_matched_fnl.slurm)
conv=$(sbatch --parsable --dependency=afterok:${base} codes/submit_quijote_dm_z0p5_r10_convergence.slurm)
sbatch --dependency=afterok:${conv} codes/submit_quijote_dm_z0p5_r10_finalize.slurm
```

最终科学图只保留 PDF；人工检查使用的栅格预览位于 `/tmp`，不在任务仓库中。
