# P(k) fNL 限制：结果与结论

## P0-only 主结果（2026-09-16 p 校准版，现行）

p 固定为校准值：pre 通道 `p=1.1789`、STO 通道 `p=1.3395`（由 fNL=+100 固定、
p 自由的校准 MCMC 测得，`results/fnlfixed100_freep_results.json`）。三方法
（pre / STO only / pre+STO only joint）重新拟合。**+100 行是同数据自洽检查**
（校准用了同一 +100 样本），独立无偏检验需拆分 realization 或其他 fNL 样本。

| target | pre | STO only | pre+STO only (joint) |
|---|---:|---:|---:|
| fNL=0 | -1.63 +/- 11.89 | -0.64 +/- 11.58 | -1.18 +/- 11.82 |
| fNL=+100 | 100.57 +/- 12.14 | 100.08 +/- 11.68 | 99.61 +/- 11.76 |

全部 6 case 通过 68%/95% recovery gate（GoF p 0.61--0.93，Rhat<1.001）。
关键读数：**p 校准消除了全部偏倚**，但也移除了此前"STO 收紧 20%"的表象——
那部分增益是被高估的响应幅度换来的统计幻觉（响应弱 ~20% -> Fisher 信息低
~36%）。校准后的诚实结论是：**在本体积与 kmax=0.08 的 P0-only 口径下，
STO 加权与 pre+post 联合都不带来显著的 fNL 精度增益**（宽度比 0.96--0.99）；
STO 的真实增益在 bispectrum 端（见五方法 Fisher：B-only 4.71 vs pre 9.55）。

## 历史表：P0-only 五方法 MCMC（p=1.12 口径，冻结存档，已被校准版取代）

12 个 P0 bins、`kmax=0.08 h/Mpc`，参数为 `fNL,b1,Pshot_residual`。四方法行冻结自 v1；
`STO only` 行按 `inputs/p0only_sto_only_amendment.md` 修正案新增（matter-assisted N+Env
STO 加权、位置不动、无 gra-recon）。

| target | pre | standard post | Marisa post | STO only | STO+Marisa |
|---|---:|---:|---:|---:|---:|
| fNL=0 | -1.57 +/- 11.09 | 4.36 +/- 11.18 | 4.61 +/- 11.01 | -0.55 +/- 9.29 | 6.49 +/- 9.14 |
| fNL=+100 | 93.03 +/- 11.21 | 100.02 +/- 11.16 | 99.95 +/- 11.03 | 79.95 +/- 9.04 | 85.38 +/- 8.92 |

所有 fNL=0 情形都覆盖真值；fNL=+100 时 pre、standard、Marisa 覆盖 68%，STO+Marisa
的真值落在 95% 内但不在 68% 内，STO-only 的真值连 95% 都不覆盖（q97.5 约 89.1）。
相对于 pre，fNL=+100 的 P0 信息量约为 standard=1.008、Marisa=1.033、STO-only=1.538、
STO+Marisa=1.580；+100 中心低估 STO-only 约 20.0、STO+Marisa 约 14.6。两个 STO
变体共用同一套权重生产，其差值说明 STO 加权本身就是低偏主因，后接 Marisa 重构
反而把中心从 79.95 拉回 85.38。

## pre+STO-only 联合 24-bin MCMC（2026-09-16 joint 修正案）

"加场不加 k"：同一 realization 的 pre 与 STO-only post P0 堆叠成 24-bin 向量，
完整 24x24 cross-covariance（同 bin 相关系数 0.97--0.99），五参数
`(fNL, b1_pre, b1_post, sn0_pre, sn0_post)`、fNL 共享。只做 pre+STO-only 一对。

| target | joint fNL | width/pre | 对照 pre-only | 对照 STO-only | truth in 68/95 |
|---|---:|---:|---:|---:|---:|
| fNL=0 | 0.10 +/- 7.40 | 0.667 | 11.09 | 9.29 | True/True |
| fNL=+100 | 49.84 +/- 6.92 | 0.617 | 93.03 | 79.95 | False/False |

fNL=0：联合比 pre 收紧 33%、比 STO-only 单通道再收 20%，中心 0.10 无偏——
共享 fNL 的噪声平均是真增益。fNL=+100：**偏倚泄漏被放大**，中心 49.84 远低于
STO-only 单通道的 79.95（profile GoF p=0.002 拒绝模型）；共享 fNL 让失配的
post response 以大权重拖走无偏的 pre 估计。结论：联合向量在 response 闭合前
不可用于 unbiased fNL 限制；response 闭合后，fNL0 档 33% 的收紧才是可用增益。

## P+B 交叉检查

另一个 matched100、`kP<=0.08`、`kB<=0.20` 的联合分析中，pre 的 P-only 为 `93.12 +/- 11.15`，P+B 为 `101.36 +/- 9.69`；STO+Marisa 的 P-only 为 `85.27 +/- 8.85`，P+B 为 `95.39 +/- 6.97`，加入 B 后宽度分别缩小约 13.1% 和 21.3%。更正后的 common-P scope 给出 STO+Marisa P+B `102.69 +/- 8.45`，不能与 P0-only 表直接混用。

## 五方法 Fisher 基线

固定体积、无 nuisance 的单参数经验 Fisher `sigma(fNL)` 为：

| method | P-only | B-only | P+B |
|---|---:|---:|---:|
| pre | 7.2086 | 9.5471 | 7.0103 |
| standard post | 7.0997 | 13.2431 | 6.9406 |
| STO-only | 6.4460 | 7.9058 | 6.0569 |
| STO+standard | 6.2275 | 8.5419 | 5.9532 |
| STO+Marisa | 6.2372 | 4.7092 | 4.4479 |

这是信息基线，不是 P0-only MCMC 误差；STO 系的 Fisher 优势不能单独证明 unbiased
recovery，STO-only 的 MCMC +100 低偏已直接说明这一点。

## 结论

1. P0 来自同一套 jaxpower 周期盒 monopole 测量，post-recon 必须使用 reconstructed data 和 shifted randoms。
2. standard 对 P0 的 fNL 信息基本保持，不能宣称明显增益；Marisa 在当前 `kmax=0.08` 约保持或小幅增加信息。
3. STO 系收紧 P0-only 统计误差（STO-only 0.81、STO+Marisa 0.80 倍 pre 宽度），但 +100 target 低偏：STO-only 约 20.0 且 95% gate 未过，STO+Marisa 约 14.6 且 68% gate 未过。STO 加权是低偏主因，Marisa 重构部分缓解；两者都只能称为“更精确但需 response 校准”。
4. pre+STO-only 联合向量的 fNL0 增益真实（0.67 倍 pre 宽度、中心无偏），但 +100 偏倚泄漏被共享 fNL 放大到约 -50 且 GoF 拒绝；response 闭合是联合推断可用的前置条件。
5. P+B 比 P-only 更紧，但必须使用完整 P--B cross covariance。
6. 当前最稳妥的 Pk 结论是：STO 有潜在信息收益，但其加权场的 PNG response 与 lattice 模型 `bphi=2*delta_c*(b1-p_halo)` 约定的失配必须先闭合；完成 centered `LC_m`/`LC_p` response 和 bias/nuisance 校准前，不升级为最终 unbiased fNL constraint。
