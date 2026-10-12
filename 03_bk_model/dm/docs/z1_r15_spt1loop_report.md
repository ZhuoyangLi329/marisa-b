# Quijote DM：低 k 的一环 fNL recovery 修正与审计

更新日期：2026-09-08。当前状态：本轮修正、完整低 k 加密积分检查和 MCMC recovery 均已完成；主恢复图已改用 posterior median 与 equal-tail 68% credible interval。保留并量化非零的 Gaussian 基线残差，不宣称已经实现均值级别无偏闭合。

## 1. 本轮问题和判断标准

本轮针对 DM、z=1、R=15 h^-1 Mpc 的 diagonal B000(k,k)，复用三个 cosmology 公共的 446 个 realization（ID 0..445），不重跑模拟。输入 fNL 为 -100、0、+100。按用户要求保留排除前两个 diagonal bin 的拟合设置，重点检验 kmax=0.08、0.10；0.15、0.20 仅作为较高 k 的对照，不以修复其偏移为本轮目标。

接受约 +12～15 的低 k 偏移，并不等于模型已经对 446-realization 均值实现无偏闭合。本报告分别给出 recovery 中心、单盒误差和均值层面的残差，避免混淆这三个标准。

## 2. 当前采用的模型及修正

采用固定理论模板

    B(fNL) = G + fNL L + fNL^2 Q

G 为 Gaussian tree+one-loop，L 包含 B111 及 B122I、B122II、B113I、B113II，Q 为当前 local-PNG 的 B112II 二次系数。它是显式截断到所保留 G/L/Q 扇区的预测，不是宣称任意有限 fNL 的精确多项式模型。

| PNG 项 | pre / post 的核对应 | 当前输出字段 | 本轮 likelihood |
|---|---|---|---|
| B111 | DM 的一阶核不因重构改变 | dBdfNL_png_tree | 乘 fNL，已接入 |
| B122I | 外部 F2 与 pre P12 / Zrec,2 与 post P12 | dBdfNL_png_B122I | 乘 fNL，已接入 |
| B122II | 两个 F2 / 两个 Zrec,2 的 connected PNG loop | dBdfNL_png_B122II | 乘 fNL，已接入 |
| B113I | pre 三阶 propagator / post 三阶 propagator | dBdfNL_png_B113I | 乘 fNL，已接入 |
| B113II | F3 / Zrec,3 的 connected PNG loop | dBdfNL_png_B113II | 乘 fNL，已接入 |
| B112II | F2 T0 / Zrec,2 T0，local primordial trispectrum | B112II_fNL2_coefficient | 乘 fNL^2，本轮补接 |

审计入口为 `codes/marisa_b/src/marisa_b/marisa_b_native.cpp` 中的 `compute_pre_recon_png_1loop_dfNL`、`compute_post_recon_png_1loop_dfNL_vectors_impl`、`local_png_B112II_fNL2_orientation_integrand` 及有限 fNL assembler。输出端又检查 G 和 L 各分量相加能重建保存的总量，避免“内核算了但 profile 未读入”的遗漏。

- 现有 native pre/post PNG 路径都包含上述四个线性 loop 项；没有用仅含 Gaussian 项的重构公式代替 PNG loop。
- 旧 profile 未接入 Q。本轮在 pre 和 post 的有限 fNL likelihood 中都加入 Q，并对 Q=0 做消融。
- G/L/Q 使用同一组 shell 径向与角积分节点、同一个 z=1 谱表、相同的外部边支持及径向 loop 截断配置。post 使用 R=15、b_rec=1、cellsize=5、CIC paint/read 的 sinc^4 窗口；pre 不施加重构窗口。
- qmin=2 pi/1000 h Mpc^-1，qmax=20 h Mpc^-1。外部第三边也要求 k3>=qmin；被去掉的支持保留原 shell 归一化，不重新归一化剩余区间。
- kmax 只作用于实测 B000 的 k1、k2，不能额外截断被积分的 k3；即便低 kmax，k3 也可能接近 2 kmax。
- 旧 pre-L adaptive 路径存在采样预算耗尽的警告。最终 pre-L 使用重构强度为 10^-30 的 post vector/QMC 路径，积分相同的无重构核，并与原 pre 路径独立比较；原始警告保留在 fine.json，不能因 return code 为零就忽略。

这里仍是连续 shell、标准方向三角形以及无 alias 的 CIC 窗口近似，不是精确的三维格点一环预测，也没有完成 CIC 的全 SO(3) 方位平均。qmin 的径向积分截断、PNG 的 exchange IR cutoff 与所有内部传播子逐一落在非零 Fourier 格点上并不等价。使用“matched”表示显式统一这些已建模的约定，不表示所有测量细节已被精确模拟。

最大幅度的低 k 改善来自 Gaussian 重构模板及测量约定的修正，而不是靠任意 gain、nuisance 或把模拟 fiducial 均值塞进最终模型。保持旧 L 的窗口消融已经能把旧 post fNL=0 的约 +52.1/+56.7 降到约 +14.0/+12.1；完整 G/L/Q 结果见下表。

## 3. MAP/profile 定位结果

使用 `matched.npz`，已排除 k=0.0154351、0.0331589 h Mpc^-1。kmax=0.08 选两个 bin，kmax=0.10 选三个 bin。covariance 来自 446 个 fiducial realization 的单盒散布，使用 Hartlap 修正，V=1 (h^-1 Gpc)^3；它不是 446 盒均值的 covariance。

| 阶段 | kmax | input -100 | input 0 | input +100 |
|---|---:|---:|---:|---:|
| pre | 0.08 | -100.731 | -1.136 | +98.807 |
| pre | 0.10 | -113.037 | -14.200 | +85.017 |
| post | 0.08 | -88.040 | +12.364 | +113.177 |
| post | 0.10 | -89.167 | +10.596 | +110.591 |

表内是每个 input 的实际 profile 最小值，只保留作 MAP 定位和 MCMC 初始化审计。主恢复图不再使用这些中心的 Delta-chi2=1 区间；当前图例的 offset 来自三个 MCMC posterior median 对 input 做直线回归。

post 在 input=0 附近的单盒 68% 半宽约为 137.3、105.7。因此 +12.36、+10.60 分别约为单盒误差的 0.09、0.10 倍。对应沿 fNL 方向的模拟均值 SEM 为 6.48、4.99，这些残差仍约为 1.9、2.1 个均值 SEM，不能表述为已经证明无偏。

Gaussian 全向量的均值残差二次型为 14.08（2 bin）、15.10（3 bin），说明 fNL 投影以外也有残差；单参数中心接近 input 不等于完整 B000 均值拟合已通过。

## 4. G / odd / even 的定位证据

逐 realization 构造：G_sim=B_fid，L_sim=(B_+100-B_-100)/200，E_sim=(B_+100+B_-100)/2-B_fid。保持 realization 配对后计算均值和 SEM；E_sim 包括全部偶次效应，不应强行等同于纯 B112II。

在实际 covariance 定义的 fNL 方向上，post 的 odd 响应相对理论为：

| kmax | odd 响应幅度 | 幅度 SEM | even_sim 等效 fNL | 10000 Q 等效 fNL | even SEM |
|---|---:|---:|---:|---:|---:|
| 0.08 | 1.005997 | 0.004768 | 0.38767 | 0.18017 | 0.65105 |
| 0.10 | 0.999395 | 0.003452 | 0.50165 | 0.38134 | 0.49019 |

加入 Q 对 post 的 input=+/-100 recovery 中心的净影响分别不超过 0.189（kmax=0.08）和 0.448（kmax=0.10）；对 input=0 的影响不到 0.004。因此 Q 不应该漏接，但它不能解释原来的约 +50 或当前约 +10～12 的共同偏移。

另做了互斥两折 fiducial 均值锚定定位实验：训练折只提供 G，测试折提供目标均值和 covariance，理论 L/Q 不变。post 两折平均恢复为：kmax=0.08 时 (-100.58,-0.21,+100.64)，kmax=0.10 时 (-99.50,+0.31,+100.38)。此项仅支持“残差主要在 G”这一定位，绝不将锚定模板作为最终物理模型，也不将两折平均称为独立的理论闭合证明。

## 5. 数值和实现审计

主模板：n_mu=24、n_rad=3、Gaussian epsrel=0.01、PNG QMC power=12、4 replicates。低 k 加密检查分别使用 n_mu=48、n_rad=5，或者 epsrel=0.002 与 QMC power=14；必须覆盖实际拟合的 bin 2、3、4，不能用一个精选 bin 代表全部拟合范围。

已完成的独立检查：

- synthetic G+fL+f^2Q 在 input=-100、0、100 的 recovery 闭合，Q=0 与旧解析 profile 精确一致；同时用直接残差、solve(C)、独立一维最小化和根求解复核中心及 68% 边界。
- 现有 native 测试在当前源码对应的已更新测试 binary 上重跑，输出 `post-R1 finite-PNG native checks passed: 185`；测试覆盖闭合三角形、无重构极限、finite-box PNG、QMC 测度、B112 极限/重构强度依赖及 generic/local 路径一致性等。命令为 `OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 codes/marisa_b/build/halo_v1/test_eft_v2_post_r1_png_native`。这是所列实现性质的测试，不是对所有物理误差的保证。
- `MODEL_AUDIT.json` 核对 binary、native source、谱表、simulation 以及模型组装来源的 SHA256，并检查 NPZ 数组与对应 metadata 完全一致。
- pre G 加密带来的等效 fNL 漂移：角积分约 0.27/1.00，径向约 0.34/1.36，loop 约 1.17/1.80，依次对应 kmax=0.08/0.10。
- pre L 的原 adaptive 路径与独立零重构强度 QMC 的差异，投影到 fNL=100 时约 -0.056/+0.090；最终组装模板不再包含原 pre-L 警告批次，原文件仍保留作来源。
- 全枚举低 k 格点 mode pairs 的独立 tree 验证通过 shell 计数核对；两个独立实现的 tree 数值一致。仅替换离散 tree 的混合诊断仍不能消除 post 的约 +12～15 偏移，且它不等于完成精确格点一环。

完整 post 三-bin 精度检查已完成，结果分别保存在 `mucheck_lowk.json`、`radcheck_lowk.json`、`loopcheck_lowk.json` 并进入最终审计。三份文件均覆盖实际拟合位置 2、3、4，均无积分警告。对每份加密模板实际重做有限 fNL 拟合后得到：

| 独立加密项 | kmax=0.08 最大 recovery 漂移 | kmax=0.10 最大 recovery 漂移 |
|---|---:|---:|
| n_mu: 24 → 48 | 0.10712 | 0.35892 |
| n_rad: 3 → 5 | 0.32897 | 0.28232 |
| epsrel: 0.01 → 0.002，QMC power: 12 → 14 | 0.03302 | 0.01409 |

每格是三个 input 中中心漂移绝对值的最大值；这是各单项加密的实测稳定性，不是任意组合加密或所有数值误差的严格界。它们明显小于保留的约 +10～12 偏移，因此无需为本轮结果再替换主模板。pre 的对应 Gaussian 加密重拟合最大变化为约 1.84；低 k pre-L 独立路径差异和 G/L/Q 的 coarse/fine 对比也分别保留在 audit 中，不用 post 的测试替代 pre 的证据。

完整加密任务在 Slurm job 58065144 上完成（COMPLETED，exit 0:0，墙钟 18 分 23 秒），三项检查共享一个 CPU 测试节点、各用 8 个 CPU；没有重跑模拟。原 shared 排队任务仅取消本轮的 58064609_2、58064810_0、58064810_1，未改动其他任务。

## 6. 产物与复现入口

最终模型输入：`matched.npz` / `matched.json`。组装来源为 `fine.json` 和 `pre_linear_qmc_fine.json`。`integral_cache.sqlite` 用于精确续算，键包含物理配置、三角形、输入和 binary/source 哈希，不是可随意删除的无用缓存。

主恢复图：`9.11meeting/dm/quijote_dm_z1_r15_1loop_b000_fnl_profile_matched_finite_v3.pdf`，四个面板为 kmax=0.08、0.10、0.15、0.20。尽管为保持会议引用路径而保留旧文件名，其中当前内容已是 MCMC posterior median 与 equal-tail 68% CI；旧 v1 保留。

低 k G/odd/even 诊断图：`9.11meeting/dm/quijote_dm_z1_r15_matched_model_audit_v3.pdf`，两页，只展示排除前两 bin 后 k<=0.10 的实测点，误差为均值 SEM。

在仓库根目录执行（现有环境，不安装新包）：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python codes/prepare_quijote_dm_z1_r15_matched_fnl_model.py --level fine --workers 8 --export-bundle
python codes/prepare_quijote_dm_z1_r15_matched_fnl_model.py --level fine --stages pre --sectors L --pre-linear-zero-shift-qmc --workers 8 --name pre_linear_qmc_fine
python codes/prepare_quijote_dm_z1_r15_matched_fnl_model.py --assemble-from analysis/quijote_dm_z1_r15_matched_fnl_model_v3/fine.json --replace-pre-linear-from analysis/quijote_dm_z1_r15_matched_fnl_model_v3/pre_linear_qmc_fine.json --name matched
python codes/run_quijote_dm_z1_r15_1loop_fnl_profiles.py --theory-bundle analysis/quijote_dm_z1_r15_matched_fnl_model_v3/matched.npz --exclude-first-diagonal-bins 2 --primary-kmax 0.10 --analysis-dir analysis/quijote_dm_z1_r15_matched_fnl_model_v3 --prefix quijote_dm_z1_r15_1loop_b000_fnl_profile_matched_finite_v3
```

复现 audit 时使用 `codes/audit_quijote_dm_z1_r15_matched_fnl_model.py --bundle .../matched.npz`，并以重复的 `--variant` 指定 `coarse.json`、`fine.json`、`pre_gaussian_mucheck.json`、`pre_gaussian_radcheck.json`、`pre_gaussian_loopcheck.json`、`mucheck_lowk.json`、`radcheck_lowk.json`、`loopcheck_lowk.json`。三项 post 加密可复用 `codes/submit_quijote_dm_z1_r15_lowk_convergence.slurm`，常规提交时显式使用 `sbatch --array=0-2`。不修改 native 物理内核，不重新生成 446 个 realization，也不读取历史弃用目录。

## 7. 适用范围与下一步

当前结果支持把 kmax=0.08、0.10 的这个 diagonal、单盒 covariance、固定宇宙学、input fNL=0/+/-100 的 recovery 作为改进后的报告结果；完整 post 低 k 积分审计已通过。不能推广为全 120-bin、任意 fNL、其他 R/z 或 446 盒均值级别的无偏验证。

若后续要把约 +10～12 进一步压小，优先检查 Gaussian 基线的测量定义：CIC 方位平均、alias/离散格点以及重构位移场的实际数值传递函数，再判断剩余一环截断误差。不应在没有这类证据前继续任意追加 PNG 项或用拟合 gain 把偏差吸收掉。较高 k 的偏移保留为一环适用范围的参考。

## 8. 交付与目录自查

核心代码为 `codes/prepare_quijote_dm_z1_r15_matched_fnl_model.py`（同约定模板、缓存和组装）、`codes/run_quijote_dm_z1_r15_1loop_fnl_profiles.py`（有限 fNL MAP/profile 参考）、`codes/run_quijote_dm_z1_r15_1loop_fnl_mcmc.py`（MCMC、收敛诊断、精确积分复核和最终四面板 PDF）、`codes/audit_quijote_dm_z1_r15_matched_fnl_model.py`（来源/分量闭合/旧版回归/实际加密重拟合/G-odd-even 审计），以及本轮的 Slurm 提交脚本。

主 recovery PDF 为 1 页，低 k 诊断 PDF 为 2 页，均已逐页查看。最终科学图只保留 PDF；显示检查用的临时栅格文件已清理，没有生成 PNG。

本轮 7 份被完整结果替代的局部诊断 JSON、1 份同数值的重复 recovery 草稿及其 JSON/NPZ、6 份排错/完成日志（合计 16 个文件），已可恢复地移动到 `old_doc_codes/archive_20260908_dm_z1_matched_fnl_v3_m5dgZi/`。没有删除这些分析资料，没有改动旧 v1/v2 或其他任务产物。后续工作只需阅读当前目录的 REPORT、MODEL_AUDIT、最终 profile 和被引用的模型/精度输入，不需要读取归档目录。

## 9. MCMC 替换 profile errorbar

主恢复图现在采用固定一维后验

    p(fNL | B) proportional to exp[-chi2(fNL)/2],  -1000 < fNL < 1000,

其中 likelihood、G/L/Q、数据 mask、Hartlap precision 和 V=1 单盒 covariance 与第 3 节完全相同。先验是明确的 uniform prior。每个 stage / kmax / input 都独立运行四个 emcee ensemble，每个 ensemble 有 16 个 interacting walkers，使用 0.8 StretchMove(a=2) + 0.2 DEMove；前 50% 丢弃，初始 8000 步，不满足门槛时每次延长 4000 步，最多 24000 步。MAP/profile 只用于初始化及对照，不参与当前图的中心和 errorbar。

低 k 的 MCMC 结果为：

| 阶段 | kmax | input -100: median [16%,84%] | input 0: median [16%,84%] | input +100: median [16%,84%] |
|---|---:|---:|---:|---:|
| pre | 0.08 | -104.73 [-260.53,+48.69] | -4.53 [-159.37,+145.44] | +93.82 [-56.53,+240.40] |
| pre | 0.10 | -116.15 [-236.55,+1.26] | -16.51 [-133.01,+97.43] | +83.39 [-30.50,+194.27] |
| post | 0.08 | -87.89 [-224.48,+50.05] | +11.00 [-125.75,+147.27] | +111.84 [-25.07,+248.19] |
| post | 0.10 | -89.88 [-195.39,+15.61] | +10.51 [-95.62,+115.09] | +108.89 [+5.75,+211.70] |

图中所有中心均为 chain posterior median，上下 errorbar 分别由第 16 和第 84 百分位直接计算。有限 fNL 二次项会让 MAP、中位数及上下宽度不必对称；因此不能再将 Delta-chi2=1 的 profile 区间当作 MCMC credible interval。低 k 的 post input=0 中位数仍为约 +11，与前面的 Gaussian 基线诊断一致。

24 个 posterior 全部通过预注册式门槛：最大 rank/folded split-Rhat=1.00628，最小 bulk ESS=15167，最小 5%/95% tail ESS=28947，最小 retained length / tau=263.68。另检查半链 median 漂移、四 ensemble median spread、tail occupancy R-hat、接受率和 prior 外侧各 2% 的质量；所有 case 均通过。这里四个独立 ensemble 是 chain 单位，同一 ensemble 内的 walkers 没有被错误当成独立 chains。

因为后验只有一维，另在同一 prior 上用 200001 点梯形积分独立计算 2.5/16/50/84/97.5 百分位。MCMC 与确定性积分的最大分位数差为 0.01988 个 posterior sigma；所有后验只有一个相对峰高大于 1e-4 的显著 mode，prior 边界质量低于 1%。从保存的 float32 retained chains 再计算全部 24 组分位数，与 JSON 的最大绝对差为 1.44e-5。

可审计产物为 `quijote_dm_z1_r15_1loop_b000_fnl_mcmc_v4.json` 和 `quijote_dm_z1_r15_1loop_b000_fnl_mcmc_v4_chains.npz`。JSON 保存每个 case 的 quantiles、MAP 对照、精确积分结果、全部 convergence gate、输入及输出 SHA256；NPZ 保存最多 2 x 4 x 3 x 4 x 8000 x 16 的 retained chain 数组和每个 case 的实际长度。原 profile-errorbar PDF 已可恢复地移动到 `old_doc_codes/archive_20260908_dm_z1_fnl_mcmc_profile_replacement_U1r25c/`，当前路径只指向 MCMC 版本；无需后续读取归档目录。

复现使用仓库已有、已确认包含 emcee 3.1.6 的 `desilike` 环境，不安装新包：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/global/homes/l/lzy/anaconda3/envs/desilike/bin/python \
  codes/run_quijote_dm_z1_r15_1loop_fnl_mcmc.py
```
