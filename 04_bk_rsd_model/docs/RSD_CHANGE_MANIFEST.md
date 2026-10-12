# RSD 新增/迁移清单（2026-10-01）

旧 `/pscratch/sd/l/lzy/reconstruction-png` 只读；只复制下列小型共享数值模块/测试，不搬运历史批量结果、manifest 或废弃 runner。

| 来源 | 第四部分目标 | 改动 |
|---|---|---|
| 用户提供 marisa-b-tree-png-rsd 文件夹 | work/reference_package/ | 完整保留设计和 oracle，不改方程 |
| 03_bk_model/marisa_b_core/src/marisa_b/marisa_b_native.h | 编译时 include | 复用 bias/triangle/config 类型；源文件不改 |
| 原仓库 codes/vendor/readfof.py | code/vendor/readfof.py | 原样复制；读取 FoF 速度，不调用写入功能 |
| 原仓库 codes/quijote_strict_triangle_fft_estimator.py | code/vendor/quijote_strict_triangle_fft_estimator.py | 原样复制；唯一 FFT 数值实现 |
| 原仓库 codes/test_quijote_strict_triangle_fft_estimator.py | code/test_quijote_strict_triangle_fft_estimator.py | 仅添加 vendor import path；9 个独立测试 |
| 原仓库 shared linear P/M table | 只读输入 | 内容 SHA256 固定并在运行时断言 |
| 第三部分 libpaware_ir.so source ABI | 审计时只读调用 | f_growth=0 的 pre/post G/L/Q 独立回归 |

新代码：`native/finite_png_poly2.h`、`native/tree_png_rsd.{h,cpp}`、`tree_png_rsd_native.py`、`audit_tree_rsd.py`、`export_pre_rsd.py`、`periodic_rsd_measure.py`、`test_periodic_rsd.py`、`run_pre_rsd_model_test.py`、`plot_pre_rsd_diagnostic.py`、`record_rsd_provenance.py`、`pre_rsd_model_test.slurm`。

未修改：第三部分旧实空间模型/API、所有旧 no-IR/IR MCMC、旧 post-recon 目录、旧 GitHub 仓库。本次没有提交/push GitHub。

验收：用户参考包 22 tests；native ≥1024 随机有向三角形逐项 G/L/Q/39 columns，100 squeezed，旋转/LOS/置换/parity，旧 native pre/post 实空间极限，连续 B0/B2/B4 quadrature 和 B1 parity，输入报错；FFT estimator 9 tests；速度/CIC 3 tests；实际三角形 exact counts 和 FFT/direct Braw 一致；实际 128/256 网格收敛。

机器可读产物的最终状态以 `outputs/TREE_RSD_NATIVE_AUDIT.json` 和 `outputs/pre_rsd_single_box/SUMMARY.json` 为准；数据目录 metadata 包含输入与输出哈希。单盒数据诊断不是 ensemble fNL recovery 验收。

`outputs/DELIVERY_AUDIT.json` 记录交付代码/模型输出哈希，核对实际样本哈希、库哈希、exact counts、FFT/direct closure 和网格收敛；Python 语法检查全部通过，outputs 中无 PNG。

## B-only 草稿与实空间 binning 核查（后续修订）

新增 `rsd_bonly_common.py`、`run_rsd_bonly_workflow.py`、`rsd_bonly_job.sh`：原 strict 三腿 B-only 草稿，尚未完成验收；用户要求数据点密度与 real-space 对齐后，已标记 superseded 并增加拒绝运行保护，不用于正式模型结论。

原 `run_pre_rsd_model_test.py` 的 PBS/spectrum helper 移入 `rsd_bonly_common.py`，原单盒公式和 EDGES 不变；其历史 DELIVERY_AUDIT 是此前交付快照，不代表后续草稿已验证。

新增只读审计 `audit_realspace_binning.py`：从现行实空间矩阵/scan AST/10 组实际冻结似然提取完整边缘、两腿 shell pairs、认证索引和切割点数。产物 `outputs/realspace_binning_reference/{AUDIT.json,REALSPACE_BINNING.npz,REPORT.md}`；不改实空间数据、模型或结果。

## 正式两腿 B000-only 流程

新增 `rsd_b000_contract.py`、`measure_rsd_b000.py`、`rsd_b000_fft_projection.py`、`build_rsd_b000_templates.py`、`run_rsd_b000_inference.py`、`run_rsd_b000_mcmc.py`、`report_rsd_b000.py` 及两个 Slurm job 文件。新增 native/FFT 独立回归、B-only likelihood/中断恢复回归、float32 子域边界回归；理论与协方差输出只在第四部分目录内。

复用但不修改：第一部分 `common/halo_b000_pipeline.py` 的 catalog/随机种子/测量/shot-noise 路径和 `quijote_dm_2d_paperbin_runner_tools.py` 的随机生成/周期边界 guard；第二部分 `run_five_method_p0only_mcmc.py` 的独立链诊断；第四部分既有源 catalog/header/PBS/spectrum helper。导入第二部分诊断不加载其 P 数据、P 模型或 P 先验。

理论 44-bin 模板及 3 个第二网格检查已完成，第五 b1 非训练点最大相对误差约 `1.72e-13`。初始真实样本已验证新理论/数据 44 个 denominator 一致，但 pilot 发现 device-domain float32 边界舍入；随即补上与 real-space 相同的共享 guard。旧早期 checkpoint 移到 `audit_early_checkpoint/` 可恢复保存，不参与新似然。此处记录的是数值测试与执行状态，不是 fNL recovery 或有效 kmax 的结论。

所有新数据使用同一冻结 contract；每份 raw catalog 哈希、测量入口哈希、共享测量源哈希和边界保护计数记入 metadata。每组 likelihood、sampler、共享 diagnostics 的 hash 也记录并用于恢复防护。作业依赖与失败/取消历史保存在新输出的 `PIPELINE.json`；未修改其他部分的旧 no-IR/IR 链、旧 post-recon 或 GitHub。

## 历史 V=100 完成后核验（2026-10-02）

全流程已完成：800 测量、500 盒协方差、21 基线 + 4 去重后先验 MCMC、8 页 PDF，25 组全部采样收敛；但在该精度/先验下没有科学认证的有效 kmax。具体 fNL 数值与边界解释见第四部分 README 和 `outputs/pre_rsd_b000_bonly_v2/RESULTS.json`。

新增 `analyze_rsd_b000_results.py` 与其隔离合成 profile 回归测试：复用 `Case.objective` 固定已知 mock 真值做事后 profile/共同残差诊断；只写 `FINAL_INTERPRETATION.json`，不改旧 SUMMARY、MAP、chain、likelihood、数据、先验、p 或 covariance。此次仅读取既有 PDF，未重新导出、未生成 PNG，未加入 IR/FoG/loop 或新 reconstruction。

## 用户纠正 covariance 口径后的重跑（2026-10-02）

当前主输出改为 `outputs/pre_rsd_b000_bonly_mean100_v1/`。只平均 data vector；covariance 保留 Fid500 单 realization 的样本散布，不按 validation/covariance realization 数缩小。18 组主扫描重新采样，保留原 7 参数模型、data、p、growth、先验、bins、Hartlap 规则；不恢复旧链，也不重测 800 份样本。

新增 `correct_covariance` 冻结入口与 `test_rsd_b000_covariance_correction.py`：逐项检查单盒 covariance、均值 data、其余模型输入与旧输出不可变性。新增 `submit_rsd_b000_covariance_fix.py` 将 18 基线、边界选择、最多 6 个 nuisance 对照、PDF/JSON/profile 报告串成 Slurm 依赖，并保存旧冻结文件 hashes。既有共享 sampler 与 diagnostics 不改；新 PDF/报告使用实际合同中的体积和 baseline 数，不写死 21 或 V=100。

报告中的平坦弱约束参数约 4% boundary mass 保留为诊断，不再直接用于否决 fNL 的先验稳健性；采用原定 nuisance 扩宽后的实际 fNL 中位数/宽度变化。PDF 技能用于新报告生成与矢量/文本 QA，遵守用户 PDF-only 约束，不生成 PNG。旧 V=100 结论只留作历史，不作为当前 V=1 的有效范围。

按用户后续指示新增 `run_rsd_b000_login.py`：仅取消本次仍排队的四个 Slurm 作业，将相同的 MCMC 入口移到登录节点；整个进程树硬绑定最多 8 个逻辑 CPU，每个并行 worker 单 CPU，各数值库单线程。保留 Slurm 提交与取消记录，实时核验 worker 的实际 CPU affinity；顺序执行 18 基线、最多 6 先验对照、新 PDF、truth-profile 和 `audit_rsd_b000_completion.py`。未修改别的任务或使用未授权核心。

初始预算诊断为 15/18 收敛、三个 `.08` 的 length/tau 不足；后续只显式增加未收敛链长，不修改诊断门。新增 `test_rsd_b000_budget_extension.py`、`recover_rsd_b000_login.py`，唯一数值 sampler 入口增加显式 budget 参数及历史来源防覆盖；旧源码只作为不可执行的来源快照保存在 `work/source_snapshots/`，SHA 为 `6be2c9d740a6a0a9bcaf79702a01df7ddf0813ee3228605add8f262a1676c135`。AST 校验 Case、诊断、checkpoint 和 proposal/RNG/backend 内循环不变，13 项独立回归 PASS。增加的步数预算不是模型自由参数；所有 physical 参数、prior、likelihood 和每组原始 RUN_PROVENANCE 均保留。

`rebalance_rsd_login_worker.py` 只允许核验 host/UID/PID/父进程/命令后，将单核 worker 迁移到原八核集合内的空闲核；记录 `CPU_REBALANCE_LOG.json`。没有增加 CPU、暂停或重启链，也不改 RNG。此调整仅用于减少登录节点核心争用。

## 当前 V=1 最终交付核验（2026-10-02 12:39 北京时间）

18 基线 + 6 nuisance 对照全部采样收敛，最大 rank Rhat 约 1.00160；低 k 宽先验链最长 44000 步，仅扩展计算预算，不改物理参数或诊断门。原八核 affinity 池、各库单线程的登录执行限制全程保留，LOGIN_STATUS 与 BUDGET_RECOVERY_STATUS 均 COMPLETE、没有活动 worker。

`COMPLETION_AUDIT.json` PASS，逐组验明 data mean100、Csingle、V=1、Hartlap 与来源；13 回归 PASS；旧输出 95 个冻结文件哈希未变。当前 PDF SHA256 为 `a4a2af1cd53733918869018e6e2a2fa7fdabdf891b6492e135997b83c82f366c`，RESULTS.json SHA256 为 `56f99654de8caeaecc9a31f8a7ca58d13828da9f5ae832edbb826bfa47d35335`。最终 8 页 PDF 文本/边界与哈希独立核对，未生成 PNG、未作栅格视觉检查；PDF 内的热图嵌入图像不等于输出 PNG 文件。

主扫描 `.08-.20` 的 18 个实际中央 68% 区间均包含真值；`.20` 三种真值 nuisance 扩宽后的中位数移动 <.02 sigma、宽度变化 <2%。`.08` 宽度改变 +12.3% 至 +13.4%，仍敏感。`.20` 只保留为本次单盒精度和先验条件下的上边界候选，不作高精度模型或完整区间认证；高 k MAP 正偏未消失。新结果和解释已加入 README，保留旧 V=100 结果的历史标签。

24 个 truth-fixed profile 均有效、未触发 free-MAP 优化问题阈值。平均 target + Csingle 的 GOF 数值不可解释为均值 likelihood 的校准 p-value；未增加 IR/FoG/loop、RSD reconstruction、P likelihood 或新 GitHub 写入。

## 新增 P0+B000 控制实验（2026-10-02）

新增：`rsd_pb_common.py`、`build_rsd_p0_window.py`、`measure_rsd_p0.py`、`run_rsd_p0_measure.sh`、`run_rsd_pb_inference.py`、`run_rsd_pb_mcmc.py`、`test_rsd_pb.py`、`run_rsd_pb_login.py`、`report_rsd_pb.py`、`audit_rsd_pb_completion.py`。所有新测量、似然、链和诊断仅写 `outputs/pre_rsd_pb_mean100_v1/`。旧 B ensemble、18 组 V=1 似然和最终 SUMMARY/RUN/RESULTS/PDF 全部哈希冻结，只读复用；不重测 B、不更新 GitHub。

数值复用映射：P catalog/随机/periodic guard/estimator 直接调用第一部分共享 helper 与 `measure_pk`；真实 P window 从同一生产 `jaxpower.BinMesh2SpectrumPoles` 导出；采样器通过显式 JointCase 继承原 Case 的 residual/objective/MAP/坐标变换/log_prob/初始化方法，复用原 `run` 和原独立链 diagnostics。原 B sampler SHA 固定为 `018d3d4bb2db0dfdd739b7933472716bdb1c866d2f7d9aa57dca3f463456614d`，不复制提案/RNG/backend 循环。

P 使用与实空间相同的 12 edges/数据点，kmax_P=.08；生产 CPU/GPU shell-boundary ULP 差异显式冻结，不混用旧 GPU window。9170 个低 k 格点的六个理论谱函数与原生产 binner 直接求和完全一致，12 个 mode counts 与实际 P 测量一致。完整 finite-PNG P0 模型与 Chebyshev 压缩在随机探针回归；初始 5 项回归 PASS，冻结联合案例回归在 aggregate 后再执行。

新 likelihood 使用 matched Fid500 `[P0,B000]` 的完整 `np.cov(..., ddof=1)`，不平均 covariance；Hartlap 使用 `12+NB` 的联合维数。B 目标、模型、噪声、bounds、C_BB 子块逐项与旧冻结 V=1 基线断言一致，只添加一个独立 Pshot 残余常数，物理 flat bounds 为 ±1e6 `(Mpc/h)^3`；b1/fNL 的旧 flat bounds 不变，p/growth 固定，不增 IR/FoG/loop 参数。

连续登录流水线于北京时间 13:39 在 login12 启动，最多八逻辑 CPU，每 worker 硬限一个池内 CPU；测量复用已有 cosmodesi，MCMC 复用 desilike，不安装环境。60 pilot/3 fine 的数值门通过才放行 800 测量和 18 MCMC；不以真值恢复筛选输入。输出比较 fNL/b1 的后验宽度和中心、fNL-b1 退化、原 B-only MAP、完整 covariance 的 `P + (B|P)` Schur 残差与 truth-fixed profile，不能用 mean100+Csingle 的 chi2 p-value 宣称高精度科学通过。

## P0+B000 最终核验（2026-10-02 15:09 北京时间）

原测量控制器触及用户 cgroup 的 30 GiB OOM 上限，一个 bulk worker 被 SIGKILL；保留原失败状态、启动合同和原控制器源码快照。仅作资源恢复：复用已存 793 份 P，补 LC_m real093..099 七份；每批 10 份、测量最多 4 workers、MCMC 仍最多 8 workers，原八核池 `[7,8,9,10,11,12,13,14]` 不变。独立 `LOGIN_RESOURCE_RECOVERY_LAUNCH.json` 与 `MEMORY_RECOVERY_NOTE.json` 记录，不覆盖原 LOGIN_LAUNCH；测量入口、模型、covariance、bounds、采样器和诊断门不变。

800 P 测量齐全、6 项联合回归 PASS、18/18 MCMC 收敛，全流程 COMPLETE、无活动 worker。最大 rank Rhat 1.0013055，fNL 最小 bulk/tail ESS 约 5636/9161，全部 8000–12000 步结束。COMPLETION_AUDIT PASS，原 B-only 所有冻结来源/结果 SHA 未变；交付独立重算 matched Fid500 Csingle，核对 800 个 raw SHA/IC/随机种子和 18 个 B target/coefficients/noise/bounds/C_BB/联合 Hartlap/Cholesky，全部通过。

同 V=1 fiducial/.10 的 fNL 从 32.4±163.7 变为 0.8±94.3，/.14 从 71.1±148.3 变为 1.9±87.4；b1 的宽度显著收紧。18 个实际中央 68% 区间均包含真值，但 /.18 与 /.20 仍有残余共同正偏。只增加独立 Pshot 一个参数，共 8 参数；未做联合宽先验对照，不认证高精度 kmax，不把 P 的改善唯一归因于 b1。RESULTS.json/PDF SHA 分别为 `8abb934915798f1f84f90692240e4990127f1e181670089f7027ab1f90863d0c` / `bea4411b57596fc68c492f4adc1d71d498185a215eb96ed58dc688e7b95c2b33`；6 页 PDF 结构、文本、页面边界和哈希核对，无 PNG，不声称栅格视觉验收。

新增只读 `diagnose_rsd_closing_leg.py`、`diagnose_rsd_zero_growth.py`，不改变 production 公式或数据：前者量化两 measured legs cutoff 下无约束 closing k3；后者对照 f_growth→0 Gaussian exact-lattice FFT 和实空间压缩格点矩基底。6.91 ppm 的非逐位差异保留原严格状态，另存 `ZERO_GROWTH_NUMERICAL_IMPACT.json`，用 Csingle 量化最大白化列差约 9.50e-5。记录 PBS bphidelta recipe 差异 -0.603588，未在本受控 P 实验中修复。未修改旧实空间、IR/no-IR/post-recon、旧 B-only 链或 GitHub。

## 补齐 RSD P0-only 三方对照（2026-10-08）

新增 `run_rsd_p0only.py`、`test_rsd_p0only.py`、`run_rsd_p0only_login.py`、`report_rsd_p0only.py`。新输出仅写 `outputs/pre_rsd_p0only_mean100_v1/`。同一 pre-RSD mean100 P0、12 点/.08 cutoff、Fid500 未缩放 C_PP、固定 p/growth；三参数 fNL/b1/Pshot 的先验直接抽取自联合案例。Hartlap 重新按 P0 的 12 维计算，不抽取联合逆协方差的子块。只运行真值 0/+100/-100 三组，表格沿六档 kmax_B 重复固定 P0-only 参照。

模型调用现有 `rsd_pb_common.p_prediction`；三参数残差、MAP 起点和输出标签作必要适配，复用唯一原 `run_rsd_b000_mcmc.run`、proposal/RNG/backend、Case 的 objective/physical/coordinates/log_prob/initial、独立链诊断。原 sampler SHA 仍为 `018d3d4bb2db0dfdd739b7933472716bdb1c866d2f7d9aa57dca3f463456614d`。三个回归核对完整协方差、P+B 的 P 子模型/数据/bounds 以及共享数值流程；报告另用有限 flat Pshot 的解析积分和加密 fNL/b1 网格核验后验。

10 月 8 日审计发现外部 live jaxpower/mesh2.py 的 SHA 改变，其他原冻结输入不变。新实验逐项核验 P0 ensemble/window/template 的冻结 SHA，不调用更新后的测量 binner；例外范围和旧/现 SHA 记录在 UPSTREAM_REUSE_AUDIT，未改变旧 P+B parent guard 或重新测量数据。控制器 taskset 启动竞态修正只涉及资源绑定顺序；原源码/失败状态另存，恢复仍为 login06、3 个逻辑 CPU、每个 worker 单核。旧 B-only/P+B 模型、数据、链和结果均只读。

北京时间 2026-10-08 19:25:20 完成；3/3 新案例均在 6000 步达到原收敛门，最大 rank Rhat 1.0004107，LOGIN_STATUS=COMPLETE、无活动 worker。真值 0/+100/-100 的 fNL 后验中位数±标准差为 1.5±100.2、104.4±98.9、-106.8±101.4。三项回归和 COMPLETION_AUDIT PASS；有限先验内 Pshot 解析积分加二维网格与 MCMC 的中位数差 <0.01 sigma、标准差相对差 <1.1%，POSTERIOR_GRID_AUDIT PASS。全部 800 份已冻结 P 测量的 metadata 确认 RSD 和 P/B 同源；没有重新测量。

交付新增三方对照 JSON/CSV/报告与两页 PDF；PDF 只做文本、页面边界、结构和哈希检查，未生成 PNG、未声称栅格视觉验收。RESULTS.json SHA256 为 `308122b08d10ab85ee45f5124c484b69b22302cd8388b6e3f1a177a8f67cd55f`，PDF SHA256 为 `a637e1a2485ab330ea72acbf67e228952280122698cea873d566fbb8910790fb`；下载副本哈希一致。P+B 相对 P0-only 的标准差在 kmax_B=.10/.14/.20 缩小约 6-7%/11-13%/14-16%，不混淆与 B-only 的改善比例；没有增加 IR/FoG/loop 或新的重构样本，也不将单盒精度检查升级为高精度 kmax 认证。

## STO Marisa post RSD 固定算子阶段一 2026-10-08

用户仅授权 STO+Marisa 一支及必要 pre 对照，已建立 active goal。扩展 `native/tree_png_rsd.h/.cpp` 的同一数值入口，增加固定非零 fNL_rec 和显式冻结 shift response 两个 C ABI；旧 C ABI 默认含义不变。源码的原始 SHA 快照保存在 `work/source_snapshots/tree_png_rsd_before_fixedrec_8ea43b7c.cpp` 与 `tree_png_rsd_before_fixedrec_800219a3.h`。新增 `build_post_rsd.sh` 只构建新命名库，不覆盖旧 pre 库。

新增 `post_rsd_fixedrec.py`、`post_rsd_fft_projection.py`；后者直接复用旧 Fields/FFTProjection/convolution 和 pre source，仅显式补入不同 p_halo 的 PBS 常数差及固定 shift 的 Delta B。原 `rsd_b000_fft_projection.py` SHA 不变。三个 `test_post_rsd_*.py` 共 15 项测试验证旧 ABI 逐位回归、完整四次多项式后截断、直接指数坐标映射二阶导数、LOS/置换/关闭位移极限、五节点 b1 插值、点级 weighted factorial identity 与 FFT 对 native 闭合枚举。

`audit_post_rsd_source.py` 的 SOURCE_AUDIT 状态为 SOURCE_TESTS_PASS_PRODUCTION_NOT_RELEASED。新版库 SHA256 为 `263778aa37531ac96ddd7b0297d460d78c0d353dfaf8351dbcf1908fead66953`；旧库仍为 `9a3ced1c69a0224e1738a2d7e4ea3a841b8533ec4a3b4a816d755a9051a0efea`，三套 pre RESULTS SHA 未变。`audit_post_rsd_operator.py` 比较实际 Marisa/CIC 算子与独立 numpy paint/FFT/readout，双精度约 2e-15、单精度约 9.1e-7 相对位移误差，均过预定门。

`audit_post_rsd_spectrum.py` 发现实际重构 alpha_rec 与旧物理 1/M 有约 0.929% 差异（并非加密插值可消除）；源码显式分离重构 alpha 与物理 PNG transfer，未改旧谱、未确定差异的唯一归一化原因。`inventory_sto_post_inputs.py` 逐样本盘点并记录缺少协方差 IC1000..1499 的同配方 DM/权重输入；halo-only lam3 的旧 fid4000 缓存明确排除。`audit_sto_post_cache.py` 用现有原始 FoF 和缓存检查 pilot 的身份与权重。所有验证使用单 CPU、数值库单线程，无新生产 MCMC、无 GitHub 写入、无 PNG。

推导、参数路径和未完成门见 `STO_MARISA_POST_RSD_DERIVATION.md`。目标的 39 组主 MCMC 和 IR 控制尚未启动；代码/代数通过不等于模拟恢复或完整 stochastic/IR 验证。

## STO Marisa post RSD 阶段二：真实 pilot、estimator 与 IR oracle（2026-10-08）

新增 `run_sto_post_rsd_pilot.py/.sh`，复用原 Marisa runner、STO 数据源、随机生成和测量。完成三种真值各一个 rec256 pilot，以及 fiducial 的 rec125、rec512、rec256_selfmatched，总计 6 个；不生产 STD/Marisa-only。realspace STO 权重随 halo 身份带入 RSD，不在 RSD 上重新分类环境。b_rec/p_rec/fNL_rec 对所有真值同样冻结；样本真值不进入算法分母。实际 cellsize8 为125³而非128³，jaxdecomp 非 Hermitian，未声称存在奇数 Nyquist bug。

LC_m000 缺缓存时调用第一部分原 `sto_recon/run_halo_sto_variants.py`；其路径要求输出在第一部分，因此中间源保存在 `01_reconstruction_methods/outputs/sto_marisa_post_rsd_source/`。同一入口还重生成 fiducial000 作为环境回归，数组和压缩文件 SHA 均与旧 matter-assisted 缓存一致。原 desilike 环境缺 pmesh 的失败在数值生产前发生，改用原 production cosmodesi 环境成功，无安装或算法替换。一次未加 taskset 的 pilot 启动被原 affinity guard 在计算前拒绝；随后所有 pilot 明确单核启动。

新增 `audit_post_rsd_weighted_estimator.py`、`audit_post_rsd_s113.py`、`audit_post_rsd_matched_self.py`：原 shot helper 默认 CIC/noninterlaced，与 TSC/interlacing3 raw 不同；S113 使用 bin-center 球贝塞尔窗。新 `post_rsd_estimator.py` 仅显式进程内适配原 helper，退出后恢复，不改安装文件或共享旧入口。完整保留原失败 JSON；实际离散 shell S113 加匹配 paint 后，独立 signed-point 含 diagonal/zero-leg 测试通过。真实 `_selfmatched` 目录不覆盖 legacy；同一 post catalog/P 完全一致。

`audit_post_rsd_pilots.py` 核验所有源和测量 hash、120-bin registry、frozen algorithm 与单核记录，报告同 IC 数值差异而不使用 pre covariance 代替 post covariance。rec256→512 的44点 B 相对 L2差2.62%，未过协方差加权数值门；不能静默选择较细网格后宣布已消除模型误差。

新增 `post_rsd_displacement.py` 和 `post_rsd_ir_oracle.py`：从 RSD physical displacement 加实际 shift 推导 A_ab(r)；逐腿 IR 源通过同一 native source 的精确单腿二次多项式响应得到，不复制 native 方程，不改原库。新增两个六项测试模块；SOURCE_AUDIT 当前27/27 PASS，原15项快照保留为 `SOURCE_AUDIT_stage1_15tests.json`。IR oracle 余项测试最初用随 wiggle 幅度变化的误差分母，失败记录和原测试源码保留；固定范数后的二次余项比约4，阈值不变，模型不改。

正式 IR 模板/参数插值、重构后的 stochastic、独立 Fid500 同配方输入和 mock/MCMC 仍未完成。没有新 MCMC、协方差替换、GitHub 写入或图像输出；所有旧 pre 最终结果哈希保持不变。

2026-10-09 凌晨继续增加 `audit_post_rsd_poisson_response.py` 和 `audit_post_rsd_poisson_coordinate.py`：前者用有限 N 的集合分割精确计算单正弦强度场的 Fourier moments，后者对真实点坐标指数映射作独立微分。四组坐标响应最大相对误差1.08e-10，且大 N 响应支持带显式外部指标标签的有向噪声候选；它与直接套入对称 R2 式的结果不同。区别及尚需核对的 estimator/随机场约定保留在 `STO_POST_RSD_STOCHASTIC_WORKING_NOTE.md`，没有宣称文献或实际 STO 的完整随机项已得到解决，也没有将候选改动加入 production 模型。

## STO Marisa post RSD 阶段三：IR 投影、独立输入与网格/随机压力测试（2026-10-09）

新增 `post_rsd_ir_harmonics.py`、`post_rsd_ir_projection.py`、`post_rsd_ir_grid.py`、`post_rsd_ir_inputs.py`：
唯一 long-tensor/source/FFT 继续复用；球谐展开、逐腿 tagged factors 和精确网格对称仅改变数值表示，
不额外角平均 source 或 LOS。读取第三部分冻结 Pnw split，物理 P/M 继续使用原 RSD snapshot；
重构 alpha2048表固定、不随 physical fNL 或 source wiggle 改变。
新增4项投影/3项对称回归及实际谱两对 bins 的双-grid审计；`IR_ACTUAL_SPECTRUM_AUDIT.json`、
`fast_ir_projection_audit/IR_ACTUAL_SPECTRUM_AUDIT.json`、`IR_GRID_ACCELERATION_AUDIT.json` 均通过对应数值门。
未加速实际四模板总耗时约45分钟，加速约5分钟，结果最大缩放差5.86e-15。
原始实际审计与比较源码快照保留，不将两点验证称为全44点/全参数生产放行。

新增 `post_rsd_marked_moments.py`、`audit_post_rsd_marked_response.py`，区分纯 data shift 源 D1、
signed F1 与平方权重 F2；两种 marked population 和有限 random 精确矩通过，坐标导数差2.36e-10。
新增 `post_rsd_poisson_projection.py` 及三项回归，将明确的 leading PN2 候选显式投影为 G/L/Q，
复用原 FFT；默认没有改动旧 likelihood。actual STO non-Poisson/finite-shift/IR stochastic 仍未验证。

扩展 matched-self context，仅新增零腿贡献记录，不改变返回的主 B 或原 self terms；旧 source/JSON
快照为 `post_rsd_estimator_before_zero_atoms_65d0f8d9.py`、`MATCHED_SELF_AUDIT_before_zero_atoms.json`。
`ZERO_ATOM_AUDIT.json` 的点枚举和 `ZERO_ATOM_REAL_PILOT_AUDIT.json` 的旧22数组逐位回归通过。
新 pilot 参数仅添加显式 `--zero-atom-diagnostics`、`--grid-origin-fraction .5`、`--nested-randoms`，
各自新目录保留旧合同与原 pilot source；完成总数9。
原点压力测试保持 rec256 与 CIC 响应，44点 B差 .0655%；nested nran30→60 保持 post halos逐位相同，
单盒B差3.36%。不借用 pre covariance，不从未白化单盒差异推断 fNL 偏差或生产 kmax。
`PILOT_NUMERICS_stage2.json` 保留旧结果；新报告将误导性的 `max_Braw_reconstructed_roundoff_difference`
更名为 `max_Braw_difference`，因为网格差不全是舍入误差，数值没有重算或选择性改变。

`audit_sto_post_calibration_inputs.py` 定位旧 p 校准 GSR1/current GSR2 的 tracer 不同，旧校准文件/结果不改。
新增 `prepare_sto_post_validation_sources.py` 串行调用第一部分现有源生成器补齐 LC_m0..99；
原 producer/entrypoint SHA 固定，不生成别的方法或改主 IC。v1在real015唯一坐标1000↔0的严格身份检查
处停止；全部失败状态/日志、已完成源保留。新增 `post_rsd_catalog_identity.py` 三项测试只允许精确周期端点等价，
不允许其他 ULP误差或排序变化。v2队列复用已完成0..15，不覆盖或重算它们。

新增 `measure_sto_post_calibration_p0.py`、`build_sto_post_p0_window.py`、`run_sto_post_calibration_p0_queue.py`，
为现成 GSR2 LC_p100..249 准备 independent physical-p 输入；主mean100/Fid500/V1不变。
首轮尚未测量即发现tail只存position cache、缺可选mass cache，失败目录可恢复保留为
`calibration_holdout_p0_initial_missing_mass_20261009`，原入口快照 SHA6ab9cf61...。
新增 `post_rsd_catalog.py` 只在单进程内临时解除原 reader 的可选 cache 路径，仍调用原质量阈值/排序/速度/RSD
数值函数，随后独立逐行核对原position cache；不伪造mass、不修改raw或protected reader。两项回归通过。
首盒LC_p100 P0已完成，actual binner9170modes、六谱函数门PASS，window另存新目录，旧 pre window未改。
校准队列最多三核，每worker最多5盒退出以限制JAX内存；不拟合p或创建协方差/MCMC。
`audit_sto_post_calibration_p0.py` 于UTC18:49:38完成150盒哈希/来源/窗口验收及P0均值汇总，
HOLDOUT_COMPLETION_AUDIT通过，aggregate SHA为149a276c1870cafd1e6719dbbf4648020ca201d1144eaaaec4fdb5973f8497d0；
入口不计算协方差、不替代主Fid500、不输出p或fNL后验。三核队列于UTC18:47:36完成、无活动worker。

新增 `audit_sto_post_validation_sources.py`；源队列100/100完成后，主三真值300个STO源及FoF行身份
全部验收通过（UTC18:34:57），不等于post P/B ensemble已测完。独立协方差1000..1499仍按原500盒要求
检查，配置路径仍缺500份DM/同配方缓存，不将任何holdout或halo-only cache替换进去。
综合 `SOURCE_AUDIT` 当前42/42通过，15/27/37/40项阶段审计均保留；旧native/pre-FFT和三套最终RESULTS SHA未变。
没有更新GitHub、没有提交Slurm、没有post MCMC，没有科学PNG/PDF输出。

继续新增 `audit_post_rsd_ir_registry.py`，以每bin满足原no-wrap条件的两套theory grids验证全部44个
实际拟合bins，复用原FastIRFFTProjection和source，逐格checkpoint保护；只覆盖一个诊断参数点。
`audit_post_rsd_ir_png_truncation.py` 检查32个有向三角形、三个b1探针和fNL宽先验端点：
明确发现b1=4/fNL=500处二次阻尼的一个微小负值，记录为高阶截断敏感性，不改主模型或先验。
完整指数只为同Gaussian tensor的部分高阶参照，不能宣称是完整PNG高阶理论。

UTC19:29:49，全部44-bin双grid入口正常结束；新增 `audit_sto_post_input_dependency_gate.py`
独立读取88份grid checkpoint复算误差，最大plain/PNG-range-weighted分别为1.44e-12/8.64e-13，
并重验原pre库/FFT/三套RESULTS和当前source hashes，保护检查通过。
新增 `audit_sto_post_covariance_inputs.py` 逐盒检查指定500份：raw FoF500、正确DM/缓存0、
已知替代缓存500但相同配方0，`COVARIANCE_INPUT_RECHECK.json` 固定此证据。
`INPUT_DEPENDENCY_GATE_AUDIT.json` 明确goal未完成、39组主MCMC完成数0；本输入问题已跨至少三个goal回合重复。
无活动计算队列后，goal工具于UTC19:32:43返回blocked；目标范围保持，等待输入目录或明确的新数据/样本授权。
这里的blocked不是科学验证通过或complete，也未修改任何covariance体积、IC、计数、方法或先验。
