# 环境分类 multitracer P(k) 全谱系 fNL 分析——详细报告

日期：2026-09-17 ｜ 分析链：fullspec v8.3（sha 见 FULLSPEC_VERSION.txt）
仓库：/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk

## 1. 执行摘要

用 STO 环境标签（lam1_dm_nenv，void/sheet/filament/knot 四类）把 halo 拆成多个
tracer，联合自谱+互谱（"全谱系"）做 fNL 约束。核心结论：

| 组合 | 数据向量 | fNL0 | fNL100 | σ(fNL100) | 收敛 |
|---|---|---|---|---|---|
| **vsf 全谱系** | 3 自谱+3 互谱 (72 维) | **−0.03 ± 10.39** | **99.93 ± 10.64** | **10.6** | ✓ |
| skf 全谱系 | 3 自谱+3 互谱 (72 维) | −0.09 ± 13.40 | 99.60 ± 13.53 | 13.5 | ✓ |
| sf 全谱系 | 2 自谱+1 互谱 (36 维) | −0.17 ± 13.34 | 99.84 ± 13.82 | 13.8 | ✓ |
| （基准）pre 单 tracer | 12 bin | ~0 | ~93–102 | 12.1 | — |
| （对照）jointA vsf 自谱 | 3 自谱 (36 维) | — | 88.6 ± 15.6（GoF 0.01 边缘） | 15.6 | ✓ |
| （对照）jointA sf 自谱 | 2 自谱 (24 维) | 5.4 ± 15.6 | 98.7 ± 14.6 | 14.6 | ✓ |
| （对照）jointA fsk 自谱 | 3 自谱 (36 维) | 7.1 ± 17.1 | 100.0 ± 16.0 | 16.0 | ✓ |

- **vsf 全谱系的 σ 点估计（10.6）为所有组合最小，低于 pre（12.1）约 12%**。
  增益主要来自 void（β=bφ/b1=−0.056 与其他类反号）+ 互谱块对宇宙方差的约束；
  sf 13.8→vsf 10.4（加 void 收紧 24%）。**注意**：72 维向量仅用 95–100 套
  估协方差（Hartlap 0.22–0.26），σ 继承协方差噪声，与 pre 的 12% 差异
  在该噪声水平下不具显著性——vsf 的排序是点估计的排序，不构成"稳健
  超越 pre"的结论（严格声明需独立协方差分割等 σ 不确定度评估）。
- 含 knot 的组合（skf 13.5）不敌 vsf：knot β=1.72 全场最高但仅 ~7600 halo
  （sn=V/n≈1.3e5），shot noise 抵消响应优势。
- jointA 自谱版里 vsf/vfk 的 GoF 只有 0.01（理论模板失配）；全谱系 G/R 框架
  修复后全部组合 GoF 干净、中心恢复精确。

## 2. 数据资产（全部 0-99 套，缺口 0）

- FKP 口径自谱产品（full/{cosmo}/realNNN/{class}_pk.npz）：本次补产 20 套
  （原批次边界粒子出界崩溃的遗留；repair_fkp_gaps.py 带 wrap v3 修复重测）。
- 裸涂场自谱缓存（fullspec/bare_auto_cache）：v/s/f 各 100+100；knot 本次新产
  100+100（knot_bare_one.py，自包含避免 h5py/mpich 冲突）。
- 互谱（full/{cosmo}/realNNN/cross_{pair}_pk.npz）：vs/vf/sf 原有 100+100；
  sk/fk 本次补产 200 套（crosspk_worker.sh，与原生产同版本同口径）。
- 校准（calibration/ENVCLASS_CALIBRATION.json）：四类 p/b1/Pshot/β，
  β: void −0.056 / sheet 1.132 / filament 1.079 / knot 1.721。

## 3. 口径问题破案链（v2 跑飞 → v8 干净）

**症状**：fullspec v2 直接用裸自谱+互谱减理论模板，MAP 跑飞
（fNL=−589，chi2/dof≈3800×，两个 target 同值——fNL 完全没跟数据对话）。

**三层根因与证据**：

1. **归一化错位**（主导）：裸涂场路径 P=value·nmodes·V/norm 中
   norm=(256³)²/V=281475 是网格常数（与粒子数无关），输出谱 ∝ n_c²；FKP 路径
   norm=n²/V 输出物理密度谱且解析扣 sn。两条路径 paint 参数完全相同
   （tsc/interlacing=3/compensate），差异纯粹是 norm 语义。
   分解实验（exp_decompose）确立解析转换并经 200 套×3 类验证：
   **P_FKP,c(k) = K_c/(n_c²·nmodes(k))·P_bare,c(k) − sn_c**，
   K: v 8.0054e10 / s 7.9474e10 / f 7.9472e10 / k 7.9861e10，
   跨套 scatter 0.16–0.32%，跨宇宙一致（probe_k_universal / probe_k_knot）。
   void 偏高 0.7% 源于 FKP sn 估计（ns/norm）与 V/n 的定义差，逐类 K 自洽吸收。
   转换后与 FKP 产品逐 bin 一致（抽查 <1%，bin1 ~7% 为 26-modes 涨落）。

2. **互谱空间分离**：环境类占据空间上分离的区域，P_cd ≠ b_c·b_d·PL。
   实测 r_cd(k)=⟨P_cd⟩/√(P_cc P_dd)：sf 0.87→0.71 平滑（fNL=100 下变化<1%）、
   vs ~0.2、vf ~0（且 vs/vf 低 k 有 10–20% fNL 依赖——β_void 反号的真实效应）。
   v4 引入 r_cd 修正后 chi2 从 3800× 降到 90×。

3. **响应形状失配**：实测差分 dP=⟨P^LC_p⟩−⟨P^fid⟩（=100×真响应）在高 k 衰减
   到 0，远快于理论 2β·PL/M 与 T_c·2β·PL/M（sheet/filament 的
   dP/(T·理论) 从 ~0.9 单调降到 0）——这是 Quijote-PNG 二次 φ 变换在模拟里
   的真实响应形状。v5（响应乘 T_c）fNL→58、v6（不乘）→546：**理论修正之间
   标度互相打架**，单点 MAP 不可靠。

   高斯形状差的归因（修正版）：void 的 42–83% 偏离形状 ≈ 1+sn/(b1²·PL)
   （sn 残余的特征 1/PL 形状可解释大半：bin1 预测 1.46 vs 实测 1.42），
   叠加 b1 校准值偏差；filament 的 19–40% 中 sn 项占 10–23%，其余为校准
   偏差。1-loop halo bias 在 k<0.08 仅几个百分点，非主因。G/R 框架不依赖
   该归因（G 直接取 fid 实测）。

**v7/v8 解法——G/R 差分模板**：
P_cc = b1_c²·(G_c + fNL·R_c) + sn0_c；P_cd = b1_c·b1_d·(G_cd + fNL·R_cd)
- G = fid 100 套平均谱（吸收高斯形状的全部真实结构：bias 1-loop、校准偏差、
  sn 残余、互谱空间分离）
- R = (LC_p 平均 − fid 平均)/100（实测响应）
- 理论模板的三个失配源全部数据化吸收；fNL = 响应系数；MCMC（jointA 同款
  iminuit profile + emcee 4×32 + rank-normalized Rhat/ESS 收敛门）全部通过。

## 4. 结果解读

- **σ 排序与物理解释**：vsf(10.6) < pre(12.1) < skf(13.5) ≈ sf(13.8) <
  jointA 自谱(14.6–20.9)。multitracer 增益条件（类间响应反差大+数密度高）
  只有 void 满足得好：n 最大（18 万）且 β 反号；knot 响应最强但数密度最低；
  sheet/filament β 接近（1.13/1.08）反差不足。
- jointA 自谱 vsf（88.6±15.6, GoF 0.01）与 vfk（102.1, GoF 0.01）的失配
  正是理论模板形状差的镜像证据；同一数据在 G/R 框架下干净且更紧。
- Hartlap：72 维×95–100 套 → 0.22–0.26，已按惯例除入协方差。

## 5. 局限与 caveats（重要）

0. **G/R 框架的定位**：模拟-模拟差分的自洽检验与理论模板失配时的诊断工具，
   不是观测分析的标准形态——实际巡天 fNL 约束的主流是解析模板拟合
   （DESI full-shape、eBOSS multitracer：CLASS + scale-dependent bias +
   nuisance 直接拟合观测谱），模拟承担校准/验证/协方差角色。观测应用时
   G/R 只作为响应校准对照或 fallback。
1. **同数据自洽检验**：G/R 与 target 共用同一批模拟——fNL0/fNL100 的中心
   恢复（−0.03/99.93）是构造性的，不是盲测成绩；有意义的量是 σ 与组合对比
   （与 jointA 的 "+100 自洽检验" 同精神）。且 σ 继承协方差估计噪声
   （见执行摘要显著性说明）。
2. **模板噪声**：R 自身是 100 套差分（~10% 样本误差），未边际化——σ 可能
   偏乐观几个百分点。
3. **R 的模拟依赖**：实测响应形状（高 k 衰减）是 Quijote-PNG 实现的性质，
   不是 local PNG 的普适性质；观测应用中响应应回到理论/解析模板，
   G/R 版结果作为对照。
4. 待做（若需要）：G 实测+R 理论混合模板对照；独立模拟盲测；σ 的协方差
   噪声不确定度评估（独立协方差分割）。

## 6. 产物清单

- 结果：fullspec/{vsf,skf,sf}_RESULTS.json + {combo}_{target}_posterior.npz
- 链：fullspec_v8.3（FULLSPEC_VERSION.txt 记 sha）；失败版本产物归档
  （vsf_MAP_v2_failed.json、sf_MAP_v7_constructive.json）
- 诊断证据：probe_caliber_result / exp_decompose_v2_result /
  probe_k_universal_result / probe_k_knot_result / r_cd_measurement /
  gap_audit（fullspec/ 下）
- 代码：run_envclass_fullspec_mcmc.py（v8.3）+ probe/diag/repair 脚本
  （code/）+ 生产 worker（crosspk_worker.sh、knot_bare_one.py、
  repair_fkp_gaps.py）
