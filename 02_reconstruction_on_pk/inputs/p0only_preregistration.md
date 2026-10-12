# 四方法 P0-only matched100 MCMC 预注册

创建时间：2026-08-31；在 ordinary-Marisa Fid500/LCp100 P0 正式生产和任何四方法拟合结果完成前冻结。

## 科学问题

比较 pre、standard post、ordinary Marisa post、STO+Marisa post 对 local-PNG
`P0(k)`-only 的 `fNL` 恢复与限制能力。只拟合输入 `fNL=0` 和 `+100`。

## 数据与 covariance

- 两个 target 都使用 realization ID 0--99 的 matched100 mean。
- 每种方法的 covariance 都只由该方法 Fiducial `fNL=0` 的 500 个单盒样本估计。
- 四种方法不共用数值 covariance，但 covariance cosmology、样本数、估计方式完全一致。
- likelihood covariance 为 `C_fid500_one_box / (42.875 * Hartlap_12)`；不除以 target 数目 100。
- `Hartlap_12=(500-12-2)/(500-1)=486/499`。
- P0 使用从 `2*pi/Lbox` 开始的 12 个 fundamental-mode bins；最后中心约
  `0.07828 h/Mpc`，最后上边界约 `0.08168 h/Mpc`。

## 理论与参数

模型固定为 finite local-PNG 线性 matter P0：

```text
P_hh = b1^2 PL + 2 fNL b1 bphi PL/M + fNL^2 bphi^2 Pphi + Pshot_residual,
bphi = 2 delta_c (b1 - 1.12), delta_c=1.686.
```

每个方法和 target 独立采样 `(fNL,b1,Pshot_residual)`；不固定 `b1`，也不把
reconstruction 使用的 operational bias 当作 posterior 中的物理 `b1`。

平坦先验：`fNL in [-300,300]`、`b1 in [0.1,6]`、
`Pshot_residual in [-1e5,1e5]`。cosmology 和线性 matter amplitude 固定。

## MCMC 与判据

- 每 case 使用 4 个独立 emcee ensemble、32 walkers、Hessian-whitened 坐标。
- 初始 6000 steps；未收敛时每次增加 2000，最多 20000；前 50% 丢弃。
- 必须同时满足：rank-normalized split-Rhat `<1.01`；fNL bulk/tail ESS `>2000`；
  其余参数 bulk/tail ESS `>1000`；retained length/tau `>50`；half-chain median
  drift `<0.1 sigma`；ensemble-median spread `<0.2 sigma`；边界质量 `<1%`；
  acceptance fraction 在 `[0.1,0.7]`。
- 报告 median、central 68%/95%、truth percentile、truth profile Delta-chi2，
  并以 pre 为基准报告误差比与 Fisher-like 信息比 `1/ratio^2`。
- 不根据结果改变 k-cut、先验、体积、样本选择或 covariance 规则。
- 图只输出 PDF。
