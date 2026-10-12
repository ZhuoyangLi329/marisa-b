# pre+STO-only 联合 P0-only MCMC 预注册修正案（联合数据向量扩展）

创建时间：2026-09-16；本文件是 2026-08-31 冻结的四方法预注册
（`p0only_preregistration.md`）与 2026-09-16 五方法修正案
（`p0only_sto_only_amendment.md`）之后的第二个 post-hoc 扩展，在任何
pre+post 联合拟合结果产生前冻结。已有 v1 四方法与五方法结果全部冻结不动。

## 科学问题

检验"加场不加 k"能否增强 P0-only 的 fNL 限制：把同一 realization 的
pre 与 STO-only post P0 堆叠成 24-bin 联合数据向量，用完整 24x24
cross-covariance 联合拟合共享的 fNL。只做 pre+STO-only 一对组合；
P0 端的其他重构通道（standard/Marisa 系）不进入本修正案。

## 数据与 covariance

- pre 与 post 的 P0 来自同一 realization 的同一快照、同一生产任务
  （per-realization `pre_pk.npz`/`post_pk.npz`），realization 序为
  fiducial 0--499、LC_p 0--99，与五方法矩阵逐方法一致。
- 联合 target：matched100 堆叠均值（fiducial 0--99 / LC_p 0--99）。
- 联合 covariance：Fid500 堆叠样本的单盒 24x24 covariance（自动包含
  pre--post cross 块，实测同 bin 相关系数 0.91--1.00，禁止 block
  diagonal）；likelihood covariance 为 `C24/(42.875 * Hartlap_24)`，
  不除以 100；`Hartlap_24=(500-24-2)/499=0.9499`。

## 理论与参数

每通道模型与 v1 相同的 finite local-PNG tree-level lattice：

```text
P_ch(k) = b1_ch^2 PL + 2 fNL b1_ch bphi_ch PL/M + fNL^2 bphi_ch^2 Pphi + sn0_ch,
bphi_ch = 2 delta_c (b1_ch - 1.12), delta_c=1.686,
ch in {pre, post}.
```

- 参数为 `(fNL, b1_pre, b1_post, sn0_pre, sn0_post)`：**fNL 全局共享**，
  每通道有独立的线性 bias 与 shot-noise 常数（即 v1 的 Pshot_residual，
  v1 结果中它本来就是每 case 自由参数）。
- 平坦先验：`fNL in [-300,300]`、每 `b1 in [0.1,6]`、每
  `sn0 in [-1e5,1e5]`。cosmology 与线性 matter amplitude 固定。

## 已知风险声明（先于结果冻结）

1. **response 失配泄漏**：post 通道的 PNG response 与 lattice 模型
   `bphi=2*delta_c*(b1-1.12)` 约定的失配会经由共享 fNL 泄入联合估计；
   五方法结果显示 sto_only 的 +100 单通道低偏约 20，本联合的
   pre+STO-only +100 中心预期介于 pre-only（93.03）与 STO-only（79.95）
   之间——若如此则证实泄漏，如实报告，不作为 unbiased constraint。
2. **pilot 增益的乐观性**：sn0/b1 固定的线性 pilot 给出 pre+STO-only
   联合 sigma 约 6.1（比 pre-only 9.3 好 35%），但该下限忽略了自由
   sn0 的吸收与两通道响应形状差异；真实 5 参数结果预期显著缩水，
   以 MCMC 为准。

## MCMC 与判据（沿用 v1 预注册原文，逐条不变）

- 每 case 4 个独立 emcee ensemble、Hessian-whitened 坐标（5 维）。
- 初始 6000 steps；未收敛时每次 +2000，最多 20000；前 50% 丢弃。
- 收敛门：rank-normalized split-Rhat `<1.01`；fNL bulk/tail ESS `>2000`；
  其余参数 ESS `>1000`；retained length/tau `>50`；half-chain median drift
  `<0.1 sigma`；ensemble-median spread `<0.2 sigma`；边界质量 `<1%`；
  acceptance `[0.1,0.7]`。
- 报告 median、central 68%/95%、truth percentile、以 pre-only 冻结结果
  为基准的误差比与信息比；coverage 校准检查 0.68/0.95。
- 不根据结果改变 k-cut、先验、体积、样本选择或 covariance 规则。
- 图只输出 PDF。

## 本轮执行范围

- 新 TAG `quijote_halo_z0p5_pre_post_joint_p0only_fid0_p100matched100_mcmc_v1`、
  新 seed；只写本部分 `work/`、`results/`、`figures/`、`docs/`。
- 只做 pre+STO-only 一对 x 2 target = 2 个新 case；对照列（pre-only、
  STO-only 单通道）取冻结数字（v1 与五方法），不重跑。
