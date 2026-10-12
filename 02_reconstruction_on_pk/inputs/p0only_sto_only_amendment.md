# 五方法 P0-only matched100 MCMC 预注册修正案（STO-only 扩展）

创建时间：2026-09-16；本文件是 2026-08-31 冻结的四方法预注册
（`p0only_preregistration.md`）的 **post-hoc 扩展**，在任何 STO-only P0-only
拟合结果产生前冻结。四方法 v1 的链、结果和图保持冻结不动；本修正案只新增
`sto_only` 方法条目，并把主结果表升级为五方法。

## 扩展动机

- 五方法 Fisher 基线（`results/five_method_pk_b_fisher_results.json`）本来就包含
  `sto_only`（P-only 无 nuisance sigma 约 6.45，相对 pre 7.21 约有 10% 信息增益），
  但 P0-only MCMC 主表此前缺少 sto_only 行。
- 加入 sto_only 后，`sto_only` 与 `sto_marisa` 使用**同一套** STO 权重生产
  （`r10_lam1_dm_nenv` 目录族），两者的差值可以干净地归因于"是否再接
  Marisa gra-recon"。

## 新增方法定义

`sto_only`：matter-assisted N+Env STO 加权 halo catalog 的 post P0。
- 组件为 void/sheet/filament/knot 四个纯环境类（单一质量 bin、单一密度 bin），
  `env_source=dm`、`weight_source=dm`、`lambda_th=1`、平滑 `R=10 Mpc/h`、
  number weighting。
- catalog 政策：halo 位置不变、只挂 `INDWEIGHT=sto` 权重，**不做 gra-recon**；
  uniform unit-weight random catalog 配对。

## 数据与 covariance（沿用四方法预注册原文，逐条不变）

- 数据源：`sto_only_matrix.npz` 的 `fiducial_post_P0`（500 个 realization）与
  `LC_p_post_P0`（100 个 realization）前 12 bins；geometry 与理论 lattice
  逐 bin 强制一致（edges/nmodes 精确相等、k 全部接近）。
- 四方法 v1 的矩阵数组原样继承（bit 级不变）；五方法矩阵只新增
  `sto_only__*` 键。
- sto_only target 同样使用 matched100 mean（realization ID 0--99）；
  covariance 由 sto_only 自己的 Fiducial `fNL=0` 500 个单盒样本估计，
  likelihood covariance 为 `C_fid500_one_box / (42.875 * Hartlap_12)`，
  不除以 100；`Hartlap_12=(500-12-2)/(500-1)=486/499`。

## 理论与参数（沿用四方法预注册原文）

```text
P_hh = b1^2 PL + 2 fNL b1 bphi PL/M + fNL^2 bphi^2 Pphi + Pshot_residual,
bphi = 2 delta_c (b1 - 1.12), delta_c=1.686.
```

- `(fNL,b1,Pshot_residual)` 独立采样；平坦先验 `fNL in [-300,300]`、
  `b1 in [0.1,6]`、`Pshot_residual in [-1e5,1e5]`。
- STO 加权场的 operational bias 约 2.2（`sto_bias.json` 推荐
  `b1=2.197952896776342`）；按预注册原文，operational bias 不进入 posterior，
  `b1` 是自由物理参数，其宽先验已覆盖该量级。
- 已知风险声明：STO 加权场的 PNG response 与 lattice 模型的
  `bphi=2*delta_c*(b1-p_halo)` 约定之间同样存在 sto_marisa 已观察到的
  response 失配风险（+100 低偏）；本扩展如实报告 recovery gate，不因结果
  调整模型。

## MCMC 与判据（沿用四方法预注册原文，逐条不变）

- 每 case 4 个独立 emcee ensemble、32 walkers、Hessian-whitened 坐标。
- 初始 6000 steps；未收敛时每次 +2000，最多 20000；前 50% 丢弃。
- 收敛门：rank-normalized split-Rhat `<1.01`；fNL bulk/tail ESS `>2000`；
  其余参数 ESS `>1000`；retained length/tau `>50`；half-chain median drift
  `<0.1 sigma`；ensemble-median spread `<0.2 sigma`；边界质量 `<1%`；
  acceptance `[0.1,0.7]`。
- 报告 median、central 68%/95%、truth percentile、truth profile Delta-chi2、
  以 pre 为基准的误差比与信息比；coverage 校准检查 0.68/0.95。
- 不根据结果改变 k-cut、先验、体积、样本选择或 covariance 规则。
- 图只输出 PDF。

## 本轮执行范围

- 只新跑 `sto_only` 的 `fNL0` 与 `fNL100` 两个 case（新 TAG、新 seed）。
- pre/standard/marisa/sto_marisa 四行数字复制自冻结 v1 结果，不重跑。
- 五方法主表、图与报告把冻结 v1 行与新 sto_only 行合并呈现，并明确标注
  行来源。
