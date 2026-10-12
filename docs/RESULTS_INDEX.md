# RESULTS INDEX —— p-aware 项目结果权威索引

生成：2026-09-20 ｜ 维护：本文件是全部现行结果的唯一权威索引；任何结果被取代时必须更新此文件。
路径约定：02/ = `02_reconstruction_on_pk`，03/ = `03_bk_model/envclass_b`，OLD = 旧仓库 `/pscratch/sd/l/lzy/reconstruction-png`。

## 表 1：现行结果矩阵（CURRENT）

### P0 / P(k) 线（02/）

| 对象 | 统计量 | 文件 | 状态 | 关键数字（fNL=100 注入） |
|---|---|---|---|---|
| 五方法（pre/standard/marisa/sto_only/sto_marisa） | P0-only MCMC（p 固定校准值） | `02/results/five_method_p0only_mcmc_results.json` | FROZEN 主结果 | σ(pre)≈12.1；各方法见文件 |
| pre+sto_only | P0 joint 24-bin | `02/results/pre_post_joint_p0only_mcmc_results.json` | FROZEN | — |
| 五方法 p 校准 | freep（fNL=100 固定，p 自由） | `02/results/fnlfixed100_freep_results.json` + `02/work/*_freep_calib_v1/`（std_marisa、sto_marisa 两目录为 09-18 补格） | FROZEN | 见表 3 |
| 环境类 multitracer P0（自谱 jointA） | MCMC | `02/envclass_p0only/jointA/{sf,fsk,vfk,vsf}_RESULTS.json` | EXPLORATORY 收官 | fsk 100.0±16.0 最优；结论=数密度天花板 |
| 环境类 multitracer 全谱系（G/R 差分模板 v8.3） | MCMC | `02/envclass_p0only/fullspec/{vsf,sf,skf}_RESULTS.json` | EXPLORATORY 收官 | vsf 99.93±10.64（点估计，见报告 caveat） |

### B(k) 线（03/，fixed-b1 口径除特别注明）

| 方法 | 统计量 | 文件 | 数据版本 | 关键数字 |
|---|---|---|---|---|
| pre | B-only kmax 扫描 | OLD `analysis/quijote_halo_z0p5_direct_finite_tree_kmax_validity_scan_v42p875/DIRECT_FINITE_TREE_KMAX_VALIDITY_PROFILES.json` | FNL250 旧产 | 98.5±23.6 @0.08 → 120.5±10.5 @0.20，GoF 干净到 0.15 |
| **sto_marisa** | **B-only kmax 扫描（现行）** | `03/analysis/STO_MARISA_B_V4_PAWARE_REPRO_SCAN.json` | **v2 重产（png_p=1.2822）** | 74.2±18.3 @0.08 / 84.3±6.9 / **96.9±7.5 @0.12**（GoF 过线到 0.12） |
| marisa | B-only kmax 扫描 | `03/analysis/MARISA_V2_TRILOGY.json` | v2 重产（png_p=1.1217，600 套全自产 B） | 115.4±56.3 / 98.7±17.5 / 129.2±16.7 @0.12（GoF>0.7 到 0.15；渐偏与老 marisa 相同） |
| sto_only | B-only（借 standard 模板） | `03/analysis/STO_B_FIXEDB1_KMAX_SCAN.json` | 旧产 | 负结果（GoF 全崩、b2 顶界）：缺专属模板 |
| 环境类 f/s/f+s | B-only | `03/analysis/ENVCLASS_B_KMAX_SCAN.json` | 新产 | 负结果：拆类不推高 kmax 边界 |

### P+B / P-only 联合线（03/，b1 自由、p 每方法统一）

| 方法 | 统计量 | 文件 | 关键数字 |
|---|---|---|---|
| pre | P-only（12 bins @k≤0.08） | `03/analysis/P_ONLY_PRE_VS_STOMARISA.json` | 110.0 ± 12.5 |
| sto_marisa | P-only | 同上 | **100.0 ± 10.7**（P 端无偏） |
| pre | P+B 联合 vs B-kmax | `03/analysis/JOINT_PB_PRE_VS_STOMARISA_V4_MATCHED250.json` | 109.8±12.6 @0.08；120.3±11.4 @0.12（P250+B250，被 B 高 k 渐偏拖累） |
| **sto_marisa** | **P+B 联合（现行）** | 同上 | 97.1±10.1 / 97.2±8.6 / **103.1 +7.3/-7.2 @0.12**（P250+B250，GoF=0.329） |
| marisa | P-only / P+B 联合 | `03/analysis/MARISA_V2_TRILOGY.json` | P-only 100.0±73.6（caveat：P 块用五方法矩阵，cov 口径偏保守）；P+B 103.0±25.7 @0.12（块对角 cov 近似） |
| — | 三方法全景图 | `03/figures/fnl_vs_bkmax_panorama_3methods.pdf` | pre / marisa v2 / sto_marisa v4 三面板；sto_marisa P+B 已更新为 P250+B250 |

## 表 2：版本谱系（SUPERSEDED，均存 `03/analysis/_archive/`）

| 被取代版本 | 取代者 | 作废原因（一句话） |
|---|---|---|
| `STO_MARISA_B_FIXEDB1_KMAX_SCAN.json`（v1：halo p=1.282、adaptive 旧常量） | v4 | 参数考古中间版（adaptive 端取值错位） |
| `STO_MARISA_B_FIXEDB1_KMAX_SCAN_V2_RECPARAMS.json` | v4 | adaptive 用物理 p（1.282/2.228）而非生产实录，数据也未重产 |
| `STO_MARISA_B_FIXEDB1_KMAX_SCAN_V3_OPERATOR.json`（80/86/97） | v4 | 对旧数据（png_p=1.0 生产）的如实建模，数据本身被重产取代 |
| `JOINT_PB_PRE_VS_STOMARISA.json`（v1 固定 b1） | V2_FREEB1 | 主人裁定：P+B 联合不能固定 b1 |
| `JOINT_PB_PRE_VS_STOMARISA_V2_FREEB1.json`（sto_marisa P100+B100） | V4_MATCHED250 | sto_marisa 目标系综扩为 realization 对齐的 P250+B250，与 pre 的 250 套统一 |
| `02/.../fullspec/vsf_MAP_v2_failed.json`、`sf_MAP_v7_constructive.json`、`vsf_MAP.json` | fullspec v8.3 RESULTS | v2→v8.3 演化过程（口径转换/r_cd/G-R 框架定型） |

## 表 3：p 权威表（png_p 进 marisa 重构分母 b_eff=b1+bφ·fNL_loc·α(k)）

| 方法 | P0 实测 p（freep，唯一校准通道） | 旧生产 png_p 实录 | 现行生产（重产 v2） | 理论端 halo p | 理论端 adaptive (b_rec, p_rec) |
|---|---|---|---|---|---|
| pre | 1.1789 | —（无重构） | — | 1.179 | —（pre 模板无 adaptive） |
| standard | 1.1212 | 1.0 | 未重产 | 1.121 | — |
| marisa | 1.1217 | 1.0 | **1.1217**（生产中） | 1.1217 | (1.9364, 1.1217) |
| sto_only | 1.3395 | 1.0 | 未重产 | 1.339 | — |
| sto_marisa | 1.2822 | 1.0 | **1.2822**（完成） | 1.2822 | (2.2281, 1.2822) |

物理立场（主人确认）：p 是场属性，P/B 共用；B 端无法独立校准 p（简并），只能消费 P 端校准值。

## 核心结论（p-aware 重产战役收官，2026-09-19）
1. **p 收敛验证 ✓**：marisa 1.1217→1.1217（重产数据 freep 精确回归）；sto_marisa 1.2822 全链一致。
2. **生产端 png_p=1.0 的历史错误不是 B 中心偏差的主因**（两条线独立确认：marisa v2 ≈ 老 marisa；sto_marisa v4 ≈ v3）——低 k 偏差与高 k 渐偏来自 tree 级模板对重构场响应的描述精度，属建模层课题。
3. **现行最优交付**：sto_marisa v4 P250+B250 @kmax 0.12 = 103.1 +7.3/-7.2（无偏、GoF=0.329、比 pre 联合平均 MINOS 半宽好 36%）。

## 口径备忘
- B-only 全系固定 b1（B 端惯例，简并所迫）；P-only 与 P+B 全系 b1 自由（P 幅度锚定）。
- P 块恒为 12 bins paperbin k≤0.08；B 块 kmax 扫描 0.08–0.20（certified 44 三角形）。
- 协方差：fid 样本协方差 / 42.875 / Hartlap；target 为注入系综均值（truth=100 恢复检验）。
- sto_marisa 现行联合线的 target P/B 均来自同一 realization 的 p-aware 生产（real000–249）；fid500 P-B 联合协方差及其 cross block 冻结沿用现行矩阵。
