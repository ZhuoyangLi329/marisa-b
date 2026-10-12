# 第四部分：Bk RSD model

本部分用于整理 redshift-space bispectrum（\(B_k\) / RSD）模型、代码、结果和验证记录。

## 当前状态

2026-10-11 **实空间p回代RSD的九组敏感性拟合已全部完成**。依用户要求，物理p改用实空间p-aware拟合实际采用的1.2822；来源为Part3 `paware_pb_ir_brecfix_k012_k014_v2/SUMMARY.json`，原校准来源Part2 `quijote_halo_z0p5_sto_marisa_fnlfixed100_freep_calib_v1/FREEP_RESULTS.json`。只替换本轮模型的固定物理p；重构算法和已测数据不重新生成，主data/C/先验/窗口逐位复用上一轮。

入口 `code/run_realspace_p_rsd_sensitivity_v1.py`，输出 `outputs/sto_marisa_realspace_p_rsd_20261011_v1/`；login26 controller1239694于04:32:17 UTC启动，现已实际结束，四单核worker使用CPU7/12/13/14。三组P0-only和六组4+4 P+B全部通过原门与完整HDF回放；四ensemble各32walkers，6000–14000步，最大rank Rhat1.001104、最小保留长度/tau50.5764。kB=.08/.14、kP=.08、mean100、C_single500/V1/full P-B cross/no-IR保持。九组每组131个独立预测探针通过，物理p实际进入P、B的p轴及原噪声项；旧链和pre对照保留。

三组新P0-only的fNL中位数/std（真值0/+100/-100）为14.34281/86.15498、106.89403/85.25430、-84.22291/86.94339。新4+4 P+B在kB=.08为8.88162/83.61614、101.38782/85.83691、-81.37043/83.52546；在.14为35.33040/70.68196、113.77679/71.36602、-38.97859/72.41035。换用实空间p只带来几单位的中位数变化，.14的残余中心偏移仍存在。同data/C/先验的独立Pshot积分给出P0由p变化导致的中位数变化-0.61813/-3.29364/+2.05444，标准差比0.96871/0.96726/0.97261；见本轮 `analytic_P0_grid_v1/AUDIT.json`，不能把有限MCMC差异全归因于p。

交付在本轮 `final_comparison_v1/`：`COMPARISON.json`含九组全部参数及两种p的逐组对比；`REALSPACE_P_RSD_COMPARISON.pdf`两页表格和68%区间图，仅包含P0-only和4+4。已核验全部18对median/std和九个median变化，并在内存逐页渲染检查，未保存图片或生成PNG。`NUMERICAL_COMPLETION_AUDIT.json`核对实际mean100/C_single500、原先验/窗口/噪声primitive及新p轴，`PDF_STRUCTURAL_QA.json`与`PDF_VISUAL_QA.json`保存图表检查，`REQUEST_COMPLETION_AUDIT.json`汇总本次请求证据。固定p的校准误差未传播；此次比较未认证完整STO噪声闭合或无偏kmax。

### 以下为上一轮post-RSD P0校准结果

2026-10-11 **新的post-P0重校准及全部九组主拟合已完成**，不是下方旧weighted-input校准结果的复用。按用户要求，校准数据改为真正Marisa位移后的STO+Marisa post-RSD P0：采用当前完整生产矩阵中的全部LC_p0..99共100份；额外150份只有weighted-input P0，不能冒充post数据。保持校准fNL固定100、原三参数先验、V1与未平均C_single500，继续取后验中位数固定进入主拟合。

本轮入口 `code/run_postP_calibrated_rsd_v1.py`，输出 `outputs/sto_marisa_postP_calibration_20261011_v1/`。login26 controller1009718于03:42:53 UTC启动，已在04:00:57 UTC前实际退出；四单核worker使用CPU7/12/13/14。按校准、三组P0-only、六组4+4 P+B（kB=.08/.14，各真值0/+100/-100）顺序全部完成，四ensemble各32walkers、6000–14000步，主拟合最大rank Rhat1.001211、最小保留长度/tau50.2910。数据mean100、post C_single500及完整P-B cross、kP=.08、原八参数/先验/no-IR/重构参数不变；旧结果保留，pre只复用对照，不重跑。

已完成post-P校准：p中位数1.3079259942，68%区间[0.6430745,2.0391199]，std0.6596333；这确实是post数据重新拟合，数值接近旧值不能被解释为假设位移前后p不变。三组新P0-only均6000步通过原采样门与完整链回放，fNL中位数/std（真值0/+100/-100）为11.33395/88.35566、110.19830/87.95123、-86.94073/89.21377。新4+4 P+B在kB=.08为9.49653/85.19362、100.65088/87.46858、-83.04278/87.68259；在.14为35.63966/72.22934、115.22232/73.15136、-37.39077/73.97602。后者收紧误差但残余中心偏移仍在，不能认证为无偏kmax。

独立有界Pshot解析积分记录在本轮 `analytic_grid_v1/AUDIT.json`：新旧固定p导致P0边缘中位数变化仅-0.00399/-0.02106/+0.01308；新MCMC与积分中位数有-2.02973/-0.34804/-0.81001的有限抽样差，未改p、先验或C来追求中心回归。校准/目标重叠及fid目标/C重叠继续明确记录；p不确定度不传播，保留条件诊断性质。

交付在本轮 `final_comparison_v1/`：`COMPARISON.json`含全部参数、九组新结果、旧post及不重跑的pre对照；`POST_P0_RECALIBRATED_RSD_COMPARISON.pdf`两页仅展示4+4及P0-only，已核验27对median/std并逐页在内存渲染检查，未保存图片或生成PNG。`NUMERICAL_COMPLETION_AUDIT.json`逐项重建mean100、未平均C500和完整P-B cross，证明原数据/先验/模板/窗口不变、p轴真正更新；每case完整HDF/先验/原诊断/分位数/std/thin回放及512个独立log-prob抽查通过。`PDF_STRUCTURAL_QA.json`与`PDF_VISUAL_QA.json`分别保存数值/页边界和视觉检查，`GOAL_COMPLETION_AUDIT.json`汇总完整目标证据。

### 以下为上一轮weighted-input p校准及比较的完成记录

2026-10-11 01:50（北京时间）**本轮用户完整目标完成：p处理审查和实现修复、全24组4+3、随后全24组4+4、完整比较及两页PDF QA均已完成**。实际controller2069901于17:46 UTC前退出；所有48组各四ensemble×32walker，6000–16000步，最大rank Rhat1.001571、最小保留长度/tau50.0215，原诊断和独立全链回放通过。六组新P0-only对照也已完成；旧链和旧结果保留，不作替代。

最终结果在 `outputs/rsd_all_available_p_recalibration_20261010_v2/pb_4plus3_then4plus4_all_p_v1/final_comparison_v1/`：`COMPARISON.json`含48组、全部参数分位数、六组P0-only、p误差及12行配对比较；`RSD_PB_4PLUS3_VS_4PLUS4_ALLP.pdf`为两页矢量图表；`PDF_QA.json`证明全部48个median/std数值对、24个比值、字体和页边界及逐页视觉检查通过。Mac锁屏导致原生Preview不可用，替代为PDFium在内存中渲染JPEG字节并检查，没有保存图片，也没有生成PNG。`GOAL_COMPLETION_AUDIT.json`逐项证明完整目标及实际48组当前source/HDF哈希，没有缩减范围或放宽采样门。

新p：pre=1.1551560180（全500正样本，std0.630499482），post weighted input=1.3080857860（全250正样本，std0.667493980）。修复了pre沿用real-space1.179、pre B对新p不能显式更新及旧摘要硬编码150等实现/元数据问题。实空间旧校准精度缩放42.875、使用post P0；本轮保持C_single/V1、post校准使用weighted input P0，这些差异已记录并诊断，不隐称相同，不反调p强迫恢复真值。

所有主拟合保持mean100、未平均C_single500、V1、no-IR、P0/.08及B两腿原分箱.08/.10/.12/.14；pre C来自1000–1499，post C来自0–499，已接受的目标/协方差及校准/目标重叠明确记录。p固定且误差未传播，低cut的A先验敏感和新增R近均匀先验仍在元数据中；采样收敛不等于完整STO stochastic闭合或无偏kmax认证。

在kB=.14，pre的4+3 median/std为2.40/86.31、107.16/84.90、-98.67/90.09，4+4为1.87/85.76、105.20/84.62、-97.20/88.78；post分别36.66/72.81、114.23/72.41、-40.96/74.08，以及35.75/72.77、114.38/73.40、-38.87/74.71。顺序为真值0/+100/-100。R效应很小，当前中心化角度R的连续B000平均为零，实际全先验端点白化响应最大0.0374669；不能把本轮结果解释为已测得独立各向异性STO噪声，也没有证明post残余fNL偏移已解决。

### 以下为本轮保留的运行检查点，不代表当前状态

2026-10-11 01:26:33（北京时间）**全24组新p的4+3 P+B完成并逐case全链独立核验；4+4完成15/24，合计39/48**。实际controller2069901在login26存活，当前LC_p-pre/kB=.10；四writer使用CPU7/12/13/14。原source/模型/先验/moves/RNG/收敛门未改变，阶段顺序是真实24组4+3完整回放之后才开启24组4+4，不用旧p或旧12参数链替代。这是带时间的运行快照，实时状态仍读STATUS并核对/proc。

新增只读 `ANISOTROPIC_R_INFORMATION_AUDIT.json` 核验全部24组冻结输入、不读活动HDF、不改模型：当前中心化R项的连续各向同性B000投影严格为零；实际离散窗口下R=±10两端的完整joint白化向量间距最大0.0374669，故R后验近似均匀先验，约4%的先验边界质量本身不能称为fNL先验依赖。全部参数分位数、六组新P0对照及p校准误差将纳入最终比较JSON；最终48组报告和PDF QA尚未完成，goal保持active。

当前执行入口 `code/run_recalibrated_prepost_noise_comparison_v1.py`，输出 `outputs/rsd_all_available_p_recalibration_20261010_v2/pb_4plus3_then4plus4_all_p_v1/`；48组原始输入共6288点独立预测回放，4+4与对应4+3的data/C/信号/p/原七先验逐位相同，唯一新增R_stoch_anisotropic[-10,10]，R不进入P0。正式进度读 `STATUS.json` 并核对/proc；24组完成实证为 `PHASE_4PLUS3_COMPLETION.json` 与各case的 `COMPLETION_AUDIT.json`。每case回放完整四HDF、全部物理先验、原门、分位数/标准差/thin RNG，并另独立抽查512个存储log_prob；不是只看SUMMARY中的converged。

全部pre正样本0..499的P0已于23:37:21测完、原worker/owner实际退出；pre新p校准四ensemble×32×6000步及完整HDF通过，p中位数1.1551560180、std0.630499482，raw摘要正确记录500。STO输入正样本0..249全250的p中位数1.3080857860、std0.667493980，旧raw摘要硬编码150的历史错误由修正伴随metadata保留实证；当前pre实际保存回调和后续通用 `rsd_physical_p_sampler_v2.py` 均从实际case取stage/count，避免复发。p_rec仍是固定算法值1.2822，未充当物理p。统计精度保持此前C_single/V1；未获直接答复前不采用仅校准的Cmean。

pre新 `PRE_B_PAWARE.npz` 全27分箱已于00:10完成：严格原pre源、b1三次/p二次，p=1.179回放旧系数，独立native有向源和每bin三个非节点FFT通过；完整joint C500白化最大差6.86314e-12。只生成可显式代入新p的系数，不增加p拟合参数，不更改旧数据或原噪声配方。已完成六组P0-only，各6000步且全HDF/先验/原门/分位数/std/thin RNG通过：pre的fNL中位数/std为0.0118/95.5418、101.7806/96.3689、-102.1076/97.1980；post为13.8275/88.4445、110.0552/88.7005、-86.2955/88.6567。这些都是固定p、mean100配C_single500/V1/no-IR的条件诊断，p校准不确定度未传播。

额外p处理核验：`analytic_pcal_p0_grid_v1/AUDIT.json` 用独立有界Pshot解析积分与二维格点重建同一先验/模型后验，pre/post的p中位数与MCMC差分别0.0021103/-0.0006473；P0后验也相符。无噪声模型自身在输入fNL=0/+100/-100时的边缘中位数约1.5111/103.9169/-102.3132，证明中位数并不严格等于MAP/真值，但不足以解释全部实际post偏移。`P_CALIBRATION_SIGN_RANGE_DIAGNOSTIC.json` 不改主p，独立matched100 profile显示weighted input p_MAP(+/-)=1.31620/1.33658，而post-P0有效值为1.24941/1.45366；范围.04/.06/.08也有变化。这些是有限样本/谱形/model范围的诊断，不给每个真值另调p，不宣称显著偏差或模型无偏。FoF速度转换另与Quijote官方Reading_FoF文档核对，未发现额外h或GroupVel转换错误。

重审校准必须用 `rsd_pcal_readonly_recheck.py` 写独立快照；原verify会刷新已被P0冻结引用的COMPLETION_AUDIT时间，不能覆盖旧证据。真实post全链重审验证旧audit/raw/corrected摘要SHA全部不变，新增链SHA回执。相关精确多项式、原RNG/断点、解析积分和通用摘要回调共12项NERSC测试通过，见 `PRE_PAWARE_FULL_COMPARISON_AND_PCAL_GRID_TEST_AUDIT.json`。比较报告入口 `report_recalibrated_noise_comparison_v1.py` 仅在真实48组结束后运行；单一两页PDF和比较JSON尚未生成，仍需PDF-only QA及最终交付。当前goal active，不以24组4+3完成缩小用户完整目标。

2026-10-10 23:10（北京时间）用户编辑goal为：仔细对照real-space的p校准并修复错误，完成全部4+3后再只加回一个独立各向异性噪声，重跑同口径4+4。主拟合继续mean100/C_single500/V1/no-IR；4+3与4+4均覆盖pre与STO+Marisa、三真值、kB=.08/.10/.12/.14（每版本24组P+B，.14优先），P0/.08共用六组对照。4+4尚未启动MCMC，不能用旧12参数链替代。

`REALSPACE_RSD_PCAL_PROCESS_COMPARISON.json` 已独立核验：实空间旧profile的实际协方差缩放是42.875而非由盒数推断，RSD当前依用户要求为1；同三参数/先验/fNL固定100/12个P边界，但各自实际离散nmodes不同，旧real-space的L/G资产保留reference，Q来自active table，而RSD六谱模板来自同一PL/M源。相同window的27个f_growth=0代数对照最大scaled误差2.02605e-15。旧real-space STO校准用post P0，当前RSD候选用加权输入P0，两者同100正样本差的单盒白化范数5.30190（包括可由残余常数吸收的变化）；独立固定fNL100 profile的p_MAP分别为input100=1.316197、input250=1.241278、post100=1.249408。这些是process诊断MAP，未替换主p中位数，也不将差异自动称错误。校准精度选择已单独询问用户：是否仅p校准使用全部均值的C_single/Ncal，主fNL拟合永远保持V1/C_single；未获直接答复前不改变此前C_single要求。

旧摘要回调写死150的错误已在实际原sampler保存链路中复现，两项NERSC回归通过后使用 `continue_all_available_pre_pcal_v2.py` 修复实际shared.save_json回调；只替换尚在等待的CPU13进程1886272（实际退出、迁移收据保留），新owner1946804等全部pre测量完成。原测量与post链不改。23:08检查已新增153/400 pre行；post新的三个P0候选链各6000步原门通过，controller1901413正等待pre校准，不将它们作为最终全方法更新结果。

新增 `post_rsd_legacy4_model.py` 只在原4+3上增加 `R_stoch_anisotropic`：N2*f_growth²*sum[P_i*(mu_i^4-1/5)]，用各自实际B window，不按cut重新正交化/改归一化；R默认先验保持原[-10,10]，P0模型不变。四项NERSC回归通过：R=0逐位还原七参数预测、独立有向腿投影、完整joint白化及f_growth=0极限。模型+回调共六项正式数值回归见 `FOUR_NOISE_AND_PCAL_CALLBACK_TEST_AUDIT.json`；初次 `-m unittest` 未设置code搜索路径导致零项导入失败，后改为直接执行测试脚本，未改变数值代码或放宽阈值。后续需完成p审查、所有新p的4+3链，再进行4+4采样与完整对照；goal保持active。

2026-10-10 22:54:50（北京时间）用户最新要求覆盖之前的旧p扫描：“所有p重新校准以后再给我结果”，并要求优先排查P0-only偏移的p处理。当前目录 `outputs/rsd_all_available_p_recalibration_20261010_v2/`；旧0.08/0.14结果与链全部保留，不是本轮更新后的结果，旧p的0.10/0.12未启动。

原700份测量已全部完成，`sto_marisa_post_rsd_mean100_v1/ensemble_measurements/AUDIT.json` 及实际 `MATCHED_ENSEMBLE.npz` 给出完整post C_single500（fid0..499），V=1/ddof1，不除目标盒数或协方差盒数。pre保留其原RSD C_single500（fid1000..1499）；主拟合数据仍三种真值各0..99的mean100，post fid目标和协方差重叠、两组calibration与主目标0..99重叠均按用户授权明确记录，不称独立mock recovery。

物理p按原fNL固定+100、[p,b1,Pshot]三参数模型分别重校准：post/GSR2 carried-weight RSD input使用全部已测LC_p0..249共250；pre使用全部完备LC_p0..499共500，原100测量只读复用，补测400份P0（不测B）。pre真实real000重放的逐mode labels/weights/vectors和nmodes与原生产window逐位相同，单盒白化数据差2.2878349872e-5；float32展示k坐标约3.0269e-6的差异完整保留，不参与模型。四worker使用CPU8..11，controller12，已新增40/400；其余完成后由CPU13自动接续pre校准。

post新校准已完成四ensemble×32walker×6000步，全部原门通过；实际owner1877267已退出，独立 `post_physical_p_all_available/COMPLETION_AUDIT.json` 重放全部HDF/先验/分位数/标准差/thin RNG，并证明250条实际输入及原均值。独立256点Z1平方回放白化scaled误差1.12856e-15，显式物理p路径一致，没有发现把p_rec当物理p。原adapter摘要保存回调写死“150”这一报告层bug被保留在原 `SUMMARY.json`，修正另存 `SUMMARY_METADATA_CORRECTED.json`，不改链、原摘要或已冻结采样代码；后续读实际case/矩阵和修正元数据，不能按原摘要硬编码计数。

新的三参数P0-only自动入口 `code/run_all_available_p_p0only_v1.py` 已在CPU7实际进入fiducial-post采样（owner1901413）；先post三种真值，等全500 pre校准完成后自动pre三种真值。各case用自己的新p中位数和C500，保留原Pshot先验、mean100/.08/no-IR/V1；冻结时每case128点独立回放新p注入与白化残差。P0-only仅含进入P的fNL/b1/Pshot；后续P+B继续用户指定同一4+3参数及kB=.08/.10/.12/.14，优先.14。pre旧B系数固定p=1.179，必须重建后才能用新p，不允许只改metadata；post既有p-aware源可按新p折叠。两种p校准和全部新拟合完成前，不发布旧p的比较表为更新结果，goal保持active。

物理提醒：模型P0的PNG依赖为fNL*(b1-p)，在fNL=0处P0预测不依赖p；校准p本身并不保证边缘后验中位数精确等真值。p校准仍用C_single500、固定新p中位数且不传播其不确定度；MAP p与边缘中位数的差异被记录，不反调p以强迫fNL恢复。

### 以下为保留的旧p历史检查点，不代表当前运行状态

2026-10-10 22:23:59（北京时间）0.14 fiducial post/pre已各四ensemble×32walkers×8000步全部通过原采样门，独立完整HDF/先验/分位数/标准差/thin RNG回放及每case512个存储log_prob抽查均通过。pre fNL中位数4.58345、std88.76525、68%[-80.18628,93.30109]；post中位数32.88629、std69.48318、68%[-31.83436,102.84019]。post std比pre小21.72%，比同阶段P0-only84.94815小18.21%，比0.08 post82.78739小16.07%；这只是当前C/p不同的诊断配置对照，不是正式纯重构改善。post中心向正值移约32.9，不能因真值0落在单盒forecast68%区间就称为已认证无偏。两组MAP valid，最大rank Rhat1.000772，最小保留长度/tau57.161；无参数边界质量。证据 `diagnostic_three_noise_prepost_k014_first_v1/COMPLETION_AUDIT.json`。

原owner1755901仍实际存活，LC_p-post已自动继续，0.14的+/-100余四组及0.10/0.12仍未完成；当前完整四cut目标active。下方为运行起始快照。

2026-10-10 22:20:34（北京时间）按用户最新“直接先算一个kmax0.14，先看看，快点”，已优先启动 `code/run_sto_three_noise_k014_first_v1.py` / `diagnostic_three_noise_prepost_k014_first_v1/`，实际login26 owner1755901/starttime152671861存活，四个单CPU writer使用7/12/13/14；fiducial-post先跑、紧接fiducial-pre，再自动+100/-100。首个真实进度为四ensemble保存6000/5000/5000/6000步，尚未称为收敛结果。0.10/0.12延后，不用先等它们完成才给0.14预览；完整四cut目标仍active。

0.14仍使用0.08同一 `LegacyThreeNoiseModel`、4+3参数、先验、固定p及逐目标N2/N3，不增加任何noise系数，不加IR；只有B mask增至27点，P仍12点/.08。post保留同402份测量及完整P–B cross，pre保留原C500，V=1/C_single/mean100；Hartlap按39点重新计算，分别361/401和459/499。原0.08的data/C子块逐位不变。17项NERSC原生模型/原RNG/IO/配置记录测试通过，六组各131个真实输入独立预测检查最大白化scaled差9.0433e-16。只是数值一致性，不是STO物理认证。

按用户反馈，`legacy3_scan_contract.py` 已去掉写死参数名/先验/盒数/截断的永久硬门，改为本次设置记录和实际数组维度诊断；授权科学调整可以另开新版，不拒绝用户修改。运行中的链仍不混入另一似然，旧结果只读保留。0.14完成case可用 `audit_sto_three_noise_k014_first_v1.py` 独立读完整HDF复核，其他case运行时不会读取它们的活动HDF；结果仍明确diagnostic。

2026-10-10 21:59:34（北京时间）实时复核：用户指定的三噪声参数 pre/post RSD P+B **六组重新拟合已全部完成**，owner1509596及24个已知最后writer实际ABSENT，54项冻结来源SHA逐一不变。`diagnostic_three_noise_prepost_k008_v1/MCMC_COMPLETION_AUDIT.json` 已在真实owner/writer终止后回放六组完整HDF、所有物理样本先验、全部原采样门、分位数/标准差与30000点thin RNG；六组全部通过。实际最高rank Rhat=1.001594，最小保留长度/tau=50.0809；fiducial-post每ensemble12000步，其余五组10000步，每组四ensemble、各32walker。六组Minuit MAP均valid，但mean100配C_single的chi2不作为校准GOF。新goal的重拟合与pre/post比较已具备实证，交付前还须核验镜像与完成回执；原700测量继续，当前685/700，未改动当前链冻结C402。

这一轮严格7个自由参数，其中3个noise：`fNL,b1,b2_native,bK2,Ashot_residual,Bshot_residual,Pshot_residual_nbar`。继承旧实际real-space兼容的常数噪声与G/L/Q功率形状，并进行显式RSD角度提升；不是旧12参数的子链或旧8参数pre的复用。**只覆盖kmax_P=kmax_B=.08**，均为no-IR、100盒均值数据、21点（12P0+9B）、完整P–B cross、C_single/ddof1/V=1，不除以100或Ncov，p不确定度未传播。pre沿用C500（fid1000–1499）、p=1.179；post沿用C402（fid0–193及292–499）、p=1.2637656599262268，post fid目标0–99与协方差重叠按用户许可保留。当前RSD物理先验和残余幅度范围保持，未导入旧real-space弱Gaussian nuisance prior。

下表的sigma是后验标准差，不是将非对称68%区间写成对称误差：

| 输入fNL | pre中位数 | pre sigma(fNL) | STO+Marisa post中位数 | post sigma(fNL) | sigma相对降低 |
|---|---:|---:|---:|---:|---:|
| 0 | -7.00 | 99.92 | 7.39 | 82.79 | 17.15% |
| +100 | 97.34 | 101.89 | 96.58 | 82.53 | 19.00% |
| -100 | -107.15 | 99.20 | -79.26 | 82.90 | 16.43% |

三种输入真值均落在相应六组边缘68%区间内，但这只是当前有限验证样本的结果；pre/post协方差样本及tracer p不同，上表不是严格同C/同tracer的纯重构效应。Ashot常数幅度仍弱、接近先验主导（原boundary_mass检验约4%，与近均匀先验相容），不能据此认证三噪声闭合物理充分。模型明确为legacy-compatible现象闭合，未认证完整marked-STO有限PNG/shift/IR噪声，未给出正式fNL或有效kmax结论。其他五个独立B截断 .10/.12/.14/.18/.20 本轮尚未采样；.15仅与.14同分箱的历史别名。原700/C500/39主链与IR任务仍保留，不因这次三噪声诊断完成而宣称全部结束。

已核验产物：`diagnostic_three_noise_prepost_k008_v1/RESULTS.json`、`INDEPENDENT_INPUT_AUDIT.json`、`MCMC_COMPLETION_AUDIT.json`，以及六组`cases/*/{contract.json,SUMMARY.json,MAP.json}`。独立输入验收覆盖786个全域/边界点，最大白化scaled差7.3755e-16<原1e-9；target/C/chol与各自原输入逐位一致。22:01:52实际补充验收 `SAVED_LOGPROB_SPOT_AUDIT.json`：11项原生回归再次通过，六组24个ensemble合计3120个首末/固定随机状态的存储log_prob与独立full-cross似然相符，最大绝对误差1.2790e-13；这是抽查，不冒充全链log_prob核验。`REQUEST_COMPLETION_AUDIT.json` 将本次用户目标的四项要求逐一关联实证，原全链/输入报告与本地镜像SHA相符。

### 保留的历史检查点

2026-10-10 21:40:55（北京时间）用户改变目标为“按旧realspace只保留3个噪声相关参数，重新拟合并与pre-recon比较”。旧12参数owner1209663、自动128k入口1414034和旧writer已于21:29:07按PID/starttime/cmdline确认终止；原链、最后同长摘要（fid/LC_p各64000、LC_m50000）及在途checkpoint保留，停止是用户改方向而非新的科学失败，证据 `user_three_noise_redirect_20261010_v1/STOP_RECEIPT.json`。原700测量继续，检查点670/700；没有安装软件、改原测量或删除文件。

新入口 `code/run_sto_three_noise_prepost_v1.py` 与目录 `diagnostic_three_noise_prepost_k008_v1/` 已于21:36:45启动，实际owner1509596/starttime152415312、login26 CPU12；六组依次为每个0/+100/-100的post与pre，全部重新采样。**严格7个自由参数：fNL,b1,b2_native,bK2,Ashot_residual,Bshot_residual,Pshot_residual_nbar；只有3个noise**，不释放raw/epsilon_phi/w²-bias/J21/R_anisotropic/shift-noise系数。旧8参数pre只作来源与历史对照，不直接冒充新的7参数pre。当前fiducial-post四ensemble/32walkers各12000步已通过原7个采样门，fiducial-pre正在6000→8000；其他四组会自动继续。四writer单CPU7/12/13/14，测量8–11继续，总CPU union仍7–14共8核。

本轮复用各自原mean100、原21个数据点（12P0+9B）、kmax_P=kmax_B=.08、完整P–B cross及C_single/V1/no-IR：post仍C402、p=1.2637656599262268，pre仍原C500、p=1.179；不除以100或协方差盒数，不动态换C，不传播p校准不确定度。两组C的sample IDs和p不同，因此sigma对照不是严格同C/同tracer的“纯重构效应”。保留当前RSD native core bounds；A对应旧S的[-20,20]、B对应旧A的[-10,10]，Pshot保持物理+/-1e6；没有导入旧real-space弱Gaussian nuisance prior。

`post_rsd_legacy3_model.py` 继承旧实际兼容G+lambda L+lambda²Q闭合：常数A*N3、功率B*N2；旧real-space Bshot的线性系数为b1*bphi（不是把完整Z1²的factor2交叉项搬过来）。新RSD有效提升用Z_G=b1+f_g*mu²的Gaussian/L项，Q保持bphi²，并投影到原非零腿/full-shell window；这是明确的legacy-compatible现象闭合假设，不是完整marked-STO、有限shift或IR物理认证。NERSC实际11项模型/独立有向legs/原RNG/JSON/lock/init回归通过。独立 `INDEPENDENT_INPUT_AUDIT.json` 对六组各131个全域/边界点回放，最大白化scaled差7.3755e-16<原1e-9门，并逐位验证target/C/chol仍与各自原输入一致。完成验收器 `audit_sto_three_noise_prepost_v1.py --completed` 还须在真实owner/writer终止后回放全部六套完整HDF、原门、分位数及thin RNG，再输出pre/post比较；当前新goal保持active。以下为历史检查点。

2026-10-10 21:07:58（北京时间）实际PID1209663/starttime151936506仍在login26存活；fiducial四ensemble各64000步结束，但只剩length/tau未过，不能写成收敛。LC_p同长检查点60000，四writer保存进度为62000/61000/61000/61000，尚在完成62000检查点；此组也只剩length/tau未过，LC_m还未开始。原700测量继续，已649/700，当前计算作业59629958为PENDING。当前推断仍冻结C402，不追随新增盒动态换C。

新增 `code/extend_sto_latest_pb_budget_v1.py`、`post_rsd_budget_only.py` 与 `audit_sto_latest_pb_extension_v1.py`；**仅准备并测试，尚未启动**。它必须等原402 owner和所有writer实际终止、三组真实产物齐全后才运行：原64k目录完整只读保留，逐字节复制未收敛组的HDF到独立 `diagnostic_latest_cov_pb_20261010_v1/budget_extension_128k_v1/`，续用原RNG/断点，每2000步检查，最多128000，已收敛组不重复采样。直接复用原parallel_case函数和writer，只路由预算/目录/入口；模型、mean100、C_single、p、先验、moves和原诊断门不改。新增预算/复制续跑/原文件保留等共12项在NERSC实际通过，包含chain/log_prob/accepted/RNG逐位比较。新完成审计入口在原owner活跃时实际返回NOT_READY，未读取活动HDF。最终还须全链原门、物理先验、分位数、thin RNG及原64k完整前缀回放；当前goal继续active。以下为历史检查点。

2026-10-10 20:30（北京时间）实际初步P+B已进入采样，不再停在P0阶段。仍为冻结C402（0–193、292–499）/mean100/V1/no-IR、12P+9B、kmax_P=kmax_B=.08；完整P–B cross、原EFT12模型/先验/seed/moves/收敛门/64k预算不变。新p校准与三组P0各四ensemble/32walkers/6000步全部通过原门，p=1.2637656599262268（std .6702758、不传播其不确定度）；post-P0的fNL中位数/标准差为12.1730/84.9482、105.4528/85.0203、−83.3012/86.6209。

原402 controller1098445于20:05首次派发四worker后退出：每个HDF已经实际保存1步，随即因progress中的numpy.int64不可JSON序列化而失败。全部原owner/worker退出事实已核验，失败状态与原日志保留。独立 `code/resume_sto_latest_pb_progress_v1.py` 只将进度步数转Python int并重定向本轮日志；复用原parallel_case/advance_ensemble和四个已有HDF，从原step1/RNG续跑，未重做p/P0或改冻结代码。新controller1209663/starttime151936506、CPU12，恢复时间20:17；新增实际HDF/JSON/断点逐位回放等7项在NERSC通过，来源见 `diagnostic_latest_cov_pb_20261010_v1/progress_io_recovery_v1/RUNTIME_CONTRACT.json`。运行中只看writer-owned进度和SUMMARY，不读取活动HDF。

20:30核验：四ensemble同长36000，fiducial尚未收敛；Rhat/fNL ESS/其他ESS/ensemble spread/acceptance通过，length/tau与half-shift未过，原程序继续加长，随后自动LC_p/LC_m。CPU7/12/13/14四writer实际存活、单线程，登录测量8–11继续。独立 `INDEPENDENT_INPUT_AUDIT.json` 从402实际盒重新算C_single、mean100、两个own-Hartlap和逐盒noise矩通过，平均矩回放最大白化scaled差4.8579e-15；PB/P0 Hartlap分别379/401、388/401，cross最大相关.2632614、joint correlation最小特征值.4713312。该PASS只认证输入/数值，不认证STO物理闭合。

`MAP_PROFILE_REVIEW_v1/REPORT.json` 独立blind least-squares与固定真值profile已完成；fiducial原Minuit的valid=false保留，其目标函数比独立free-LS高.0050804，未重写MAP或链。三种真值固定profile的单盒forecast加权chi2约.22324/.22307/.24573，不是mean100的校准GOF或正式fNL恢复。完成验收器 `code/audit_sto_latest_pb_inference_v1.py` 已准备；真正完成后还须在owner和所有writer终止后回放四套完整HDF、原门、分位数和thin RNG。当前goal保持active而非complete；原700及正式C500/39主链/IR流程不替换，完整STO noise/IR物理闭合仍未认证。以下为保留的历史检查点。

2026-10-10 19:50（北京时间）用户明确要求“重新跑吧，用最新的realization数目”。旧205盒诊断链的writer1046481已按PID/starttime核验后用SIGINT终止，旧链、MAP、失败与恢复记录全部保留；旧目录STATUS中的RUNNING是终止前快照，不是当前运行证据。新入口 `code/run_sto_post_latest_cov_pb_v1.py`、新目录 `outputs/sto_marisa_post_rsd_mean100_v1/diagnostic_latest_cov_pb_20261010_v1/` 已于19:49启动，PID1098445、login26 CPU12。

新协方差冻结于19:49:34，实际402个fid：0–193与292–499，C_single/ddof1/V=1，既不除以目标盒数也不除以协方差盒数；后续新增DONE不动态更换当前链。先在同快照下重跑LC_p100–249 mean150物理p校准，再更新三组mean100的post-P0对照，之后自动跑三组post-P+B（原12P+9B，kmax_P=kmax_B=.08，完整同盒P–B cross）。p固定为新校准中位数，不传播其不确定度；no-IR、原EFT12源与物理先验不变，旧8参数模型和正式C500/700流程不替换。

19:55检查：新物理p校准四ensemble/32walkers各6000步已通过全部原门，p中位数1.2637656599262268；当前为fiducial post-P0对照采样，三个P+B尚待自动衔接，不把准备完成说成已完成P+B采样。原700测量已到606/700；60项原科学来源与新P控制冻结来源全部SHA不变。实际任务CPU union为7–12；P+B四writer启动后为7–14，不增加第九个逻辑CPU。

四个独立ensemble改为并行，每个32walkers、独占一份HDF；CPU7/12/13/14，原登录测量CPU8–11继续，合并CPU集合仍7–14共8核。原Stretch80%/DE20%、seed、6000起步/每2000检查/最高64000、原burn/ESS/Rhat/tau/稳定性门全部保持。NERSC实际6项测试通过，包括分文件/断点与原串行sampler的chain、log_prob、accepted与RNG逐位一致。只在所有writer退出后读取HDF，运行时只看writer-owned进度JSON；HDF打开仅等待EAGAIN锁冲突，不关闭锁。未通过严格误差门的本地多项式cache候选没有部署或用于采样。

本轮诊断授权来自 `PROVISIONAL_PB_APPROVAL_20261010.json` 的直接用户答复“先跑诊断 MCMC，同时核验”，优先于下方16:08的旧顺序。完整STO stochastic/finite-shift/IR物理闭合仍未认证，不生成正式noise PASS或fNL/kmax结论。全44 intrinsic及三种zero-leg数值primitive已独立通过，最大scaled误差8.1122e-14；300目标finite-random marked响应独立回放误差6.7234e-15。新 `noise_eft_unfitted_bins_v1/REPORT.json` 已用固定.08/train50参数检验odd50的未拟合B分箱及完整cross条件残差：高k外推误差明显增大，不能将低k可拟合直接外推为完整噪声闭合或更高kmax认证。以下均为保留的历史检查点。

2026-10-10 16:08（北京时间）用户要求优先完成完整STO噪声核验，不先用实空间白噪声转移近似启动P+B。目标仍是实际初步P+B MCMC，不改成只完成P0或单元测试；顺序与缺项见 `docs/STO_NOISE_VALIDATION_SEQUENCE.md`。新增 `sto_rsd_noise_fields.py`、`test_sto_rsd_noise_fields.py`、`measure_sto_rsd_input_noise.py` 推进真实输入场这一层：原DM快照实际含位置和速度，Gadget sqrt(a)速度转换与FoF GroupVel转换分开；三真值real000均使用全量DM，保存matter real/RSD及halo w real/RSD、w²/w³ RSD的全部mixed spectra和mode，并检查128/256、TSC/interlacing3。该诊断不复制/修改production重构或P/B算法、不截断DM、不减假定Poisson matter shot。

四项独立测试在NERSC实际通过，包括原生pmesh流式/整批paint与直接低mode DFT；报告 `NOISE_RSD_FIELDS_TEST_AUDIT.json`。16:08:28检查，实际PID269059/starttime150443407在login26 CPU14、单线程R态，RSS约390MiB，fid000/mesh128已读完第1/8个DM分片、16994457粒子。状态以 `noise_rsd_input_spectra_v1/STATUS.json` 和真实PID核对；第一份启动连接曾短暂认证失败，没有创建测量进程，恢复连接后单独确认启动。当前只是第一层实际样本实测，不将halo-matter投影残差自动认作pure epsilon，不认证connected3/4、PNG、有限shift或IR噪声，也没有生成正式noise PASS或启动P+B。原700测量及C500/p/P0衔接不变，登录任务仍不超过8核。下方为保留的早期检查点。

2026-10-10 14:43（北京时间）用户批准使用粗协方差先试物理p校准和P0-only，并明确B/P+B等真实STO噪声核验。独立诊断入口 `run_sto_post_partial_p_mcmc_v2.py` 已在login26 CPU13完成四组MCMC：校准与三种真值各四独立ensemble、32walkers、每ensemble6000步，全部通过原收敛门。协方差冻结于14:28:46，使用当时实际完成的205个fid（具体编号在 `diagnostic_partial_cov_p_mcmc_20261010_v2/CONTRACT.json`），保留C_single、ddof1、V=1，不除以目标盒数。物理p校准仍是原LC_p100–249的mean150、fNL固定100和原三参数先验；p中位数1.288740，68%区间[0.605341,2.043685]，宽且属于粗协方差诊断。三组post-P0使用固定此p、mean100、原12 bins/kmax_P=.08、b1自由和原Pshot物理先验，no-IR。fNL中位数/后验标准差分别为fid14.003/88.487、LC_p108.395/88.273、LC_m−83.216/91.100。链收敛不代表B噪声或模型物理认证，正式500盒p/P0流程仍原路保留。

噪声核验与上述采样并行运行于CPU14，`audit_sto_post_noise_partial_cov.py` 已于14:36完成同一205盒的实际POST权重N2/N3/source验收、P子块一致性、两个B定义的边缘协方差及零腿白化检查。18个真值/cut组合的零腿改动单盒白化范数最大0.0057445；不是fNL偏差或正式显著性。复用已独立核验的44-bin三个leading transport算子，另对Nraw/N2=.5/1/1.5/2作幅度假设敏感度检查，并保留逐盒平均N2²；探针p=1.31没有代替MCMC校准值。见 `noise_actual_partial_cov_v1/DIAGNOSTICS.json`。真实STO N(k)、相关随机项、PNG依赖和IR dressing仍需核验，未生成 `POST_WEIGHTED_NOIR_NOISE_AUDIT.json` 或放行B/P+B。旧源real-space STO spectra包含halo–matter残差可作对照，但不是post-RSD噪声校准。新诊断v1在线性谱来源引用检查失败、未进入采样，全部失败记录保留；v2的三项模型/协方差回归在NERSC通过。整个并行阶段login任务CPU集合为7–14，最多8逻辑核，单CPU14检查峰值84MiB。

2026-10-10 12:33（北京时间）用户要求登录节点与计算节点共同测量，当前入口改为 `run_sto_post_simultaneous_v3.py`，状态以 `outputs/sto_marisa_post_rsd_mean100_v1/simultaneous_measurements_v3/STATUS.json` 和 `login/STATUS.json` 为准。登录节点 login26 已启动四个单核 worker（CPU8–11），正在 fiducial496–499；控制程序使用 CPU12，既有 p/P0 等待程序使用 CPU7，本任务登录节点 CPU 集合为7–12，最多六个逻辑核。计算节点新作业59620965仍申请8CPU/14G/30分钟，快照时PENDING/Priority，后续实际状态需实时查询。

两边按原700身份共同领取未完成样本，login从高编号开始、compute从低编号开始；每份样本的共享flock由调度器和数值worker共同持有，完整测量经原来源/三产物SHA验收后原子发布。计算节点启动还会实际检验跨节点协调器锁排他性，不通过则拒绝派发。原重构/P/B/随机种子/网格及全部冻结科学来源保持原版本，60项冻结来源已在12:33逐项复核。700完成后等待全部worker和计算作业结束，再恢复原login parent1139375按原REUSE/hash门补齐记账，原C500→physical-p→三P0流程继续；39组主MCMC、必要对照与IR目标均未减少，当前仍未放行正式B/P+B MCMC。

转换记录见 `simultaneous_measurements_v3/MIGRATION.json`。旧排队作业59619838经hold确认后取消，Slurm历史记录证明AllocCPUS=0、Elapsed=0、未分配节点；旧compute协调器1208792已终止。切换v1的退出竞态和v2的scontrol终止记录过期失败均保留在独立目录，均未启动测量worker或新compute作业。成功v3通过sacct验证实际终止事实，不将实时记录缺失当作新提交理由。旧compute8 series的343及旧login parent的291均属于历史计数，后续完成数需统计canonical700目录，不能使用历史最大编号推断连续性。

2026-10-10 09:06实际更新：P+B的P侧已新增`post_rsd_ir_p0_projection.py`，复用B的tagged D因子、同一谱分解和长位移参数，不重新造P壳层，不增加物理拟合参数。五项独立测试本地/NERSC通过；原12个P分箱、9170实际非零模式的真实谱检查已完成63个bias参数点，独立手写Z1/t导数及逐mode求和最大scaled差7.9922e-16，三组uncached长位移投影回放通过。9/17/33节点最大scaled误差3.1478e-10/1.5188e-14/3.5594e-14，原1e-9门未改，用时54.74秒、单CPU13、峰值93.60MiB，实际PID492652已退出；34/15/17项活跃控制源再次逐项SHA不变。报告`ir_p0_signal_actual_v1/AUDIT.json`，SHA`0bf6833612d26931233d86db9a9e08f0c02d4f3caf5a2ebe9f68c23412fe5e1a`。

此入口仍是确定性Gaussian-long、strict-O(Pw)、fNL²的数值探针p=1.31，不是physical-p或STO noise校准。新检查显式量化了“零阻尼single-wiggle源”与原完整tree的O(Pw²) matching余项，最大coefficient-scaled量1.2679e-4，尚未由C_single500白化，不是fNL偏差或IR提升。正式IR对照前需将matching与阻尼效果分开核验，旧no-IR/IR模型、数据和先验完全不变，未提前放行任何主MCMC。细节见`docs/IR_SIGNAL_MATCHING_NOTE.md`。

09:02:56调度核验：canonical321/700、目标300已齐、Fid协方差测量121/500；第五批59609202实际COMPLETED/0:0，同一controller自动提交下一份59609743，8CPU/14G/30分钟，当时PENDING/Priority。原owner仍实际T态、controller仍同一PID/starttime存活，下方08:38为保留的历史检查点。

2026-10-10 08:38实际更新：三真值各0..99的300份目标测量已由`audit_sto_post_target300_completion.py`逐份通过原load_pilot的三产物SHA、原STO/FoF来源、固定b_rec/p_rec/fNL_rec、原12+12+44几何与权重验收。实际POST_HALOS重新计算N2/N3与source admission一致，另逐份确认300套权重均为float32，进入共享Catalog的float32转换逐位不改变权重。三组mean100、两种B定义及全部行保存在`target300_completion_v1/TARGETS_ONLY.npz`，SHA`ce26c0775adb0c0719d48885ad3a586ce25ba36af303854e84ce47871a5a11ff`；报告`AUDIT.json`，SHA`2d7488b384cbf2c971abb24cfe45f8b6a0b8db31d7e43c16314050981550e942`。核验实际耗时7.80秒、单CPU13、峰值52MiB，实际PID398463已退出，未重新测量数据、创建任何部分/均值协方差或选择正式B定义。

三个目标的mean(N2²)相对mean(N2)²分别多5.3750e-6/5.6747e-6/4.2510e-6，保留实际均值而不把乘积均值简化；这些self矩不是实际STO raw随机功率。原full-shell分母不变的非零腿counterfactual，对mean100的kmax_B=.20 relative-L2改变量分别0.07887%/0.07262%/0.08569%，尚未由原C_single500白化，不能据此认证零腿可忽略。两种数据定义都保留，没有用nuisance吸收差异。

08:38:36调度核验：job59609202已实际在nid004077 RUNNING，仍8CPU/14G/30分钟，两路worker正在fid115/116；canonical315/700对应目标300已齐、Fid协方差测量115/500，剩余385份正在原短批次队列继续。physical-p校准、主MCMC及IR模型均未提前放行，下方08:29为上一检查点。

08:29:11最新调度核验：第四批59608119已实际COMPLETED/0:0，canonical完成315/700；同一controller1208792自动提交下一份59609202，仍8CPU/14G/30分钟，当前PENDING/Priority。原parent仍实际T态、没有重复producer或人工重提，证据`compute8_transition_20261010/LIVE_CHECKPOINT_0829.json`；下方08:25为保留的上一检查点。

2026-10-10 08:25实际更新：当前测量job59608119在nid004122实际RUNNING（batch正在DRAINING并自然完成最后两盒），8CPU/14G/30分钟，canonical完成313/700；前三批59605393、59606644、59607580均Slurm COMPLETED/0:0，实际耗时15:42、20:10、18:23。原parent1139375仍同一PID/starttime的T态，controller1208792以及原physical-p/P0衔接仍实际存活；冻结控制来源34/15/17项逐一SHA未变。旧login parent的291计数是暂停检查点，不是当前已完成总数，不能用它宣称测量停滞。

新增`post_rsd_ir_signal_interpolation.py`只对直接IR信号的b1依赖作可加密Chebyshev表示，节点数不是物理拟合参数；保留原b1=[.5,4]、fNL=[-500,500]、严格G/L/Q及native b2/bK2的affine结构，不能把IR指数当成no-IR精确四次多项式。五项独立算法测试在本地和NERSC通过。真实原谱/冻结重构核的最低原bin1已完成76次IR投影：完整b1区间10个非节点及端点、32/48双网格、9/17/33节点，原1e-9门不改；候选最大scaled误差分别3.3279e-10、1.3237e-13、3.2593e-13，耗时117.44秒，单CPU13、峰值408.68MiB，实际PID343492已退出。独立节点积barycentric及手写Chebyshev回放全部真实节点/探针也通过，最大表示差3.7681e-13。见`ir_bias_interpolation_probe_v1/AUDIT.json`和`INDEPENDENT_INTERPOLATION_AUDIT.json`。

以上是单bin、固定数值探针p=1.31的IR信号表示检查，不是physical-p校准，也不认证全44参数域、实际STO随机项、正式零腿选择、C_single白化误差或fNL恢复；没有修改任何旧冻结源、production likelihood或先验，没有增加物理拟合参数、创建正式noise PASS或启动主MCMC。后续仍需用实际校准p和原C_single500验证全44-bin后才可用于IR开关拟合。

2026-10-10 06:37实际更新：首个v3计算节点批次59605393已在nid004080成功结束，Slurm确认`COMPLETED|0:0|00:15:42|8`。06:15:50完成实际暂停交接，四个原在途worker均自然结束（Z态，不称为不存在），来源和三产物SHA通过。两份新增测量LC_m real093、fiducial real094已以原字节原子发布；再次由原`verified_finished`独立核验通过，staging路径已因rename消失，无覆盖旧结果。单盒分别403.74/410.06秒，完成记录分别单CPU121/122；冻结34项源再次通过。总canonical完成297/700，旧login队列计数仍291是因为原parent1139375实际T态，不能把这个冻结计数当作现有产物总数。下一批59606644已由同一controller1208792自动提交，仍8CPU/14G/30分钟，检查点PENDING/Priority。证据`compute8_transition_20261010/V3_FIRST_BATCH_COMPLETION.json`。

PN_cross全44原双网格88次独立卷积检查已于06:28:30完成，最大fNL-scaled差2.5429e-15<原1e-9门，用时745.03秒；实际audit PID1877859已退出。报告`noise_cross_all44_v1/AUDIT.json`，SHA`5d370be503b4aa4e89736d3c227731ffb5f718c93625be65c24ba0bf3b2522b0`。结合之前的PN_same/NN全44 scalar/integer检查，三个unit-amplitude算子的数值几何均已有独立全分箱证据；这仍不校准actual STO N(k)/physical p、不认证connected/PNG噪声或正式零腿选择、不放行39主MCMC/IR。

辅助并发节点诊断因Slurm Required ports are in use未获新step，已只对身份和启动时间核验过的本地辅助SSH PID51972发TERM，未给主作业或任何measurement发信号。之后直接节点只读诊断因原job已结束被pam拒绝，不绕过此限制，也不把它当作主测量失败；主作业成功和两份canonical产物有独立证据。避免NERSC上宽泛pgrep/proc扫描；当前后续检查只针对已知PID。以下均为保留的历史检查点。

2026-10-10 05:25实际更新：v2 job59603739于05:12:43启动，但10秒后在环境加载阶段因可选DESI_ROOT未定义且nounset开启而`FAILED|1:0`；未启动Python batch、没有job目录或测量worker。旧v2控制器已实际退出，原login parent仍同一PID/starttime且未暂停，旧源码/日志/失败状态保留。v3仅在source原nersc_env期间临时关闭nounset并随后恢复，复用v2 STOP/USR1保护和原数值。取消DESI_ROOT的环境smoke（MPI world1、单核）及13项控制测试通过，无包安装、无数据/协方差/先验变化。

新v3短批次job59605393已于05:25:29释放，实际NumCPUs=8、14G、30分钟、CPU shared、scratch/u2，当前PENDING/Reason=None；计算节点测量尚未开始。新controller在login26 CPU12、PID1208792/starttime146584417实际存活，34项冻结源全部SHA一致。原login测量271/700、parent1139375实际S态，C500/physical-p/P0原衔接存活并保留；只有新job真正RUNNING且在途盒自然完成并通过哈希门才交接。证据`compute8_transition_20261010/V3_SUBMISSION.json`。当前内存预算允许计算节点两路单核测量，8核是allocation上限而非声称8路worker；完整39主MCMC、必要控制和IR仍未放行。以下为保留的历史检查点。

2026-10-10 04:27实际更新：v1测量job59597886曾在nid004140启动，但没有完成交接或派发数据。控制器在SIGSTOP后立即检查T态而失败；旧parent实际已暂停，四个在途盒自然完成。其后Slurm预警USR1在尚未安装handler的等待阶段结束了空批次，实际`FAILED|138:0|00:13:38`及srun信号日志保留。核验旧控制器PID不存在、批次无worker且logs/publications/staging均空、四盒DONE/全部来源与三产物SHA通过后，于04:04:21只SIGCONT恢复同一原v3 parent1139375，不重建队列或伪造完成；见`compute8_transition_20261010/SAFE_ROLLBACK_59597886.json`。

v2使用新控制目录与bootstrap，复用原协调器/batch/atomic box及全部数值：STOP须轮询确认同一PID/starttime/cmd达到T；交接前即安装预警handler，worker只忽略USR1而正常遵守TERM/KILL。Slurm预警可提前60秒且默认发送到job steps，故预警后不再启动新盒。13项控制/路由/失败保护测试及真实临时process STOP、跨exec的USR1保护通过；首次临时child未exec前的身份误拒绝与license数量1显示省略导致的预检拒绝均保留，未因此重复提交或动生产输入。v2 controller实际login26 CPU12、PID356144/starttime146231731，29项冻结源一致；新job59603739实际8核/14G/30分钟，声明scratch,u2，04:26:36已释放，同一时点仍PENDING/Reason=None。原login测量继续，最新247/700，未暂停；原C500/physical-p/三P0衔接仍绑定同一原parent。v2源码冻结后不得修改，旧v1源码与失败记录保留。证据`compute8_transition_20261010/V2_SUBMISSION.json`。

新增全44的PN_same/NN独立整数闭合校验：原双网格88探针全部通过，最大缩放差2.4549e-14<原1e-9门，用时255.74秒、单核，见`noise_scalar_all44_v1/AUDIT.json`。复用既有scalar/integer oracle，不重写源核；这不认证PN_cross的全44独立grid oracle、不校准actual STO N(k)/physical p、connected/PNG噪声或正式零腿选择，不放行主MCMC/IR。以下均为历史检查点。

2026-10-10 03:20实际更新：8核/14G/20分钟的理论job59597420已由Slurm确认`COMPLETED|0:0|00:12:45|8`，实际运行节点nid004123，44个bin完整聚合且active_workers为空。随后在login26 CPU12独立回放全部396节点及88细网格probe通过，最大缩放差分别6.7423e-15/2.1312e-15，原1e-9门未改；系数SHA为`1dca92763a4e56f5d1bda52a14f5e7628ca0ae59cd859ddc90772cfa4d569fe1`，见`noise_transport_templates_v1/INDEPENDENT_COEFFICIENT_AUDIT.json`。这只认证有向噪声transport的数值表示，不校准实际STO N(k)、physical p、PNG/connected随机响应或正式零腿定义，不放行主MCMC/IR。

03:19:38实际/proc核验：原测量parent1139375、physical-p衔接1139384、P0衔接1304572及compute8 series1417040均仍是同一PID/starttime且存活，测量227/700，原parent尚未暂停。首个测量job59597886的理论依赖已解除，仍PENDING、未见计算节点测量执行；只有它真正RUNNING后才做原队列安全交接。冻结源码及完整39主链/必要控制/IR目标保持不变。单核计时覆盖所有原网格类别，纯投影两路理想预算17.3分钟不是运行保证；实际本次12:45已正常完成，计时证据保存在`compute8_transition_20261010/THEORY_TIMING_PROBE_20261010_0304.json`。以下均为保留的历史检查点。

2026-10-10 02:31实际更新：原login26测量完成207/700，parent1139375及p/P0衔接、compute8 series均按实际/proc同一starttime核验仍活着；四个测量worker继续运行。冻结源码分别17/15/17/24项SHA不变。8核理论job59597420仍PENDING/Licenses，测量59597886仍PENDING/Dependency，没有重提或暂停旧parent。

随机项新增实质推导：区分actual raw data noise N(k)与粒子self N2，在Gaussian pair/tree计数及F2零模均值下得到colored transport；white极限为Nraw*Kcross+(Nraw-N2)*Ksame+Nraw*(Nraw-N2)*KNN，Nraw=N2才退回原iid候选。独立Wick/F2标签检验显示独立uniform random在这一leading阶相消；colored/signed相关child精确矩的Nparent=10^6最大差7.8089e-6<原1e-4门，N=1000的有限N差保留且按1/N收敛。源码`post_rsd_general_noise_transport.py`仍复用唯一FFT、原full-shell/zero规则及物理fNL²，六项在本地/NERSC通过。不是实际STO误差测量或完整noise closure；actual N(k)、PNG/connected随机响应、finite shift/IR与零腿定义仍未验收，production模型/先验未改、主MCMC仍0/39。详细推导及边界见随机项working note，证据`GENERAL_COLORED_NOISE_TRANSPORT_AUDIT.json`。

2026-10-10 01:44实际更新：原login26测量PID1139375仍实际活着且RUNNING，完成183/700；原p衔接、三P0衔接与新compute8 series PID均按同一starttime核验存活。8核理论job59597420仍PENDING/Licenses，测量job59597886仍PENDING/Dependency；没有重提、暂停原parent或声称compute已开跑。

等待期间补齐IR运行环境兼容与严格协方差缓存。生产SciPy1.17.1不再提供原`sph_harm`；新增显式进程内桥接，按官方顺序映射到`sph_harm_y`，不安装包或改冻结旧源码。六项解析相位/角度、prior probe、PSD、pre-limit、显式noise offset、不可变数组检验在本地/NERSC通过。使用同一实际冻结谱/q513/rBAO110/lmax10，旧SciPy1.16.3参考与新生产环境的球谐/系数/协方差/阻尼最大缩放差5.13e-15。

缓存利用d0=U+b1*V、d1=bphi*W，仅原长位移积分五次提取六个Gram核，严格重构协方差而非把IR-B当四次多项式。实际九个b1/p probe的系数/阻尼最大误差1.3111e-13<原2e-10门，所有full Gaussian covariance PSD通过、不截断特征值；没有固定实际物理p或加拟合参数。五次cache准备约4.55秒，单参数系数组装约60微秒对原约0.91秒，但仅此子模块，不代表完整IR-B/MCMC加速。证据`ir_covariance_cache_validation/PRODUCTION_COMPAT_AND_CACHE_AUDIT.json`。宽prior下有限PNG截断阻尼负值、全44 IR-B投影、STO随机项与主MCMC仍未验收，本轮未改变任何IR公式或放行主链。

2026-10-10 01:19实际更新：用户将大部分计算改为compute节点、每份申请8核、尽量缩短walltime。原32核理论job59594709仅在hold并确认PENDING/零runtime/无执行产物后取消，sacct实际`CANCELLED by112513|0:0|0`；首次即时accounting未确认的观察失败记录保留，稍后对同一job确认后释放新job，没有重复提交。新理论job**59597420**实际NumCPUs=8、14G、20分钟；首个测量短批次job**59597886**实际NumCPUs=8、14G、30分钟，afterany依赖理论job，二者串行分配而不叠加申请16核。当前两者PENDING/Reason=None，尚无计算节点执行证据；原login26测量实际171/700、四worker继续运行。

现有shared分区MaxMemPerCPU=1905MiB，所以8核申请14GiB，不能保留60G使scheduler补到33核。测量单盒原131盒观测峰值约5.02GiB、最长627秒；按6GiB/worker+2GiB共享余量，两盒原单核worker并行，每批剩余不足16分钟不再派发并自然排空。短批次之间只有前一份作业确切终止且产物hash通过才提交下一份；观察缺失/查询超时不是终止。只有Slurm实际终态TIMEOUT/原allocation已结束，才重做尚未发布盒，独立staging仍保留，canonical从不接收半成品；数值worker失败不静默重跑。

`continue_sto_post_compute8_series.py`已实际启动login26 CPU12，PID1417040/starttime145097761，只等待确切首个job。仅在该测量job实际RUNNING后SIGSTOP暂停同一v3 parent1139375（不结束/改源），核对真实所有children自然完成及DONE/来源/三产物SHA，再授权compute测量；每批从计算节点通过原host的实际/proc验证父进程仍T态、同PID/starttime/cmd，另有series独占lock。完整700后只在实际计算job成功退出/无活worker后SIGCONT恢复v3，原单盒入口REUSE已完成结果、按原hash门补齐记账并自然退出；原C500→physical-p→三P0衔接都保留，不伪造旧COMPLETE状态。现在尚未暂停旧parent，也没有compute测量产物。

新增输出路由只隔离`ROOT/pilots`；全部alpha、self/zero审计和数据仍从原ROOT读，原recon/P/B/seed/mesh源码未改，完整盒才以原字节发布到canonical。NERSC九项预算/时间/路由/失败文件/accounting/PID测试全通过；自建短命process的STOP/CONT实际身份smoke通过；真实fid000完整数据仅只读通过publication hash门，没有移动旧数据或把小范围检查当成模型认证。证据`compute8_transition_20261010/LIVE_CHECKPOINT.json`，已启动的series及其immutable_sources在结束前不得修改。MCMC仍按原要求在登录节点，完整39主链、必要控制及IR都未放行。下方01:19之前均为历史检查点。

2026-10-10 00:41实际更新：测量queue PID1139375与physical-p衔接PID1139384均在login26核验仍活着，测量151/700。新增`continue_sto_post_p0only.py`实际parent PID1304572固定CPU7、独立session；精确绑定原p衔接PID+starttime，只有完整700/C_single500以及原same-tracer physical-p校准收敛、原parent实际退出且child为空后，才自动冻结并依次跑原39中的三组post-P0-only。现在它仅等待，没有P0输入冻结或任何新主MCMC。它与已有管理进程共享CPU7，不新增核心配额；四测量worker仍各单核，主数值流程/旧冻结源未改。

新增P0统计/采样适配相关11项在现有NERSC环境单核全部通过（本地8项通过、3项因缺NERSC采样环境跳过）。独立重算全部500真实行的C_single和每真值100行均值，逐位核对matched/nonzero两种B定义中的P数据与P-P边缘C相同，使用本身12维Hartlap而非full precision切片；接口测试与原39的P0组装数组完全相同。实际六个P-window模板、物理p校准、原三参数/物理Pshot[-1e6,1e6]先验、原P-only MAP及MCMC实现复用。N2仅作为残余Pshot的参数单位，不是宣称STO噪声Poisson闭合。详情`P0_CONTINUATION_LIVE_CHECKPOINT.json`。

后三组P0不依赖B随机项的认证，不创建假B audit、不选择正式B零腿定义；剩余18B+18PB、必要先验/共同P控制以及同基线IR均继续必需。未来联合病例冻结后需核对同一P数据/窗口/先验并在最终39组路由中复用这三组，不重复采样或把总scope降为P-only。理论job59594709同一时点仍PENDING/Licenses、实际CPU使用为0；它冻结的21项输入/源码、测量17项及p衔接15项的SHA全未变。下方00:09及更早均为保留的历史检查点。

2026-10-10 00:09实际更新：用户要求样本测量恢复到登录节点。只在hold/复核未运行、无节点、零runtime后取消测量job59592166，并核对终止旧p等待parent728654；理论job59594709保留。第一版login恢复v2未加载原NERSC环境，四worker在MPI导入前失败，没有新测量输出；v2源码/合同/日志/FAILED状态及失败的衔接控制均保留，已验证parent和workers退出。新v3只为原数值worker恢复source nersc_env步骤、禁用user site后reset数值库1线程，MPI smoke world-size1、JAX0.9.1/CPU通过，不改重构/P/B/样本/先验。

**实际已经测起来：**login26 parent1139375，四worker固定CPU8/9/10/11，parent CPU12，自动衔接parent1139384固定CPU7；所有native线程逐个核对仍只允许各自的一个CPU，线程数16不是16核。实际user cgroup上限30GiB，检查点总内存约12.52GiB；按6GiB/worker+4GiB共享余量最多4并发。新完成四盒且DONE/来源/结果SHA通过，完成数由131增至135/700，下一波四盒实际活跃并已进入B_post。详情`LOGIN_V3_LIVE_CHECKPOINT.json`，后台独立会话不依赖聊天或SSH窗口保持打开。

`continue_sto_post_login_pipeline_v3.py`绑定真实queue PID+starttime，待全700完成且原进程实际退出后自动执行严格汇总/C_single500/V1→原physical-p校准MCMC。完整主39组和IR仍待真实STO模型验收，不假称已开主链。assembler现在识别v3实际队列，旧历史状态不覆盖，仍拒绝活着的queue、缺行、平均cov或正则化。所有该continuation冻结源码在它结束前不得修改。4盒并行按旧单核中位8分钟估计剩余约19小时；9.5小时只适用于无法在30GiB内存下安全运行的8盒并发，不是保证完成时间。

同轮独立噪声数值进展：新增6项回放/整数闭合/零腿/NN结构保护在本地及NERSC通过。真实bin1与bin99各用原coarse/fine grid，共4探针；vector contractions、标量恒等式、独立整数pair-count结果最大缩放差8.9733e-15，原门1e-9。`NOISE_SCALAR_TWO_BIN_PROBE_AUDIT.json`明确仅two-bin探针，不是全44或真实STO闭合；完整396-node/88-probe独立回放入口已补齐，但job59594709仍PENDING时不写PASS。

22:58更新：已按原CPU/shared、32核、60G提交独立的44-bin noise-transport理论模板作业59594709；实际Slurm PENDING/Priority、尚无节点，原700盒测量作业59592166仍PENDING且未改动。新作业按4GiB/单核worker加4GiB余量，最多14并发（实际内存环境若为60960MiB则13）；396节点/88细网格probe全部通过后才聚合。它计算噪声算子模板，不重测样本，也不放行STO噪声或主MCMC。新作业stored batch script与当前源SHA完全相同。

精确模块测试发现纯几何helper不必要地导入旧IR sph_harm，而现有生产SciPy无此函数；不装包、不改旧IR，直接复用已独立验收no-IR合同的原网格映射与geometry SHA修复。新增compute预算两项在NERSC通过，连同前28项共30项；计划合同SHA为ad54307a36882b9eee7b148c4979ae89c8ac7cb212fef00412373e770a0bc027。目录扫描I/O等待已进一步定位到旧audit_post_rsd_weighted_estimator.py的stat；其预检parent实际已退出，测试child仍有终止信号pending，不假称已退出。所有失败/处置与最终作业信息保存在`NOISE_TRANSPORT_PREFLIGHT_RECORD.json`。模板合同中的源码在该作业结束前不得改动。完整数据、实际STO噪声验收、39主MCMC、必要控制和IR仍未完成，goal保持active。

22:50阶段更新：补充了相关cluster/signed-marks的独立一至四点矩与真实坐标响应检验，`CLUSTERED_NONPOISSON_RESPONSE_AUDIT.json`在NERSC通过（5测试，坐标误差4.5095e-10，独立闭式误差6.2767e-16）。指定cluster反例的iid候选信号响应最大缩放差0.3045；不是测得实际STO有30%偏差，不能据此宣布真实STO闭合。详见`docs/STO_POST_RSD_STOCHASTIC_WORKING_NOTE.md`新增两节。

post-noIR似然与Case支持原nuisance系数乘随b1/fNL变化的多项式噪声形状，参数数目/原bounds/mean100/Csingle/V1不变，仍在完整P+B空间白化。NERSC的28项moment/似然/Case/冻结保护/算子测试通过，本地不含NERSC专用sampler的24项通过；新compute内存/核心预算两项已在本地通过。有向PN-cross、PN-same、NN三个unit-amplitude transport算子已实现并通过点级物理式、非节点PN以及闭合shell枚举，三个算子不等于三个已认证nuisance。44-bin模板入口需396节点/88细网格probe，通过前不放行实际STO噪声模型。

新增模板提交前的广义unittest discover扫描停在共享filesystem I/O等待，PID975337实际存在、wchan=cl_sync_io_wait、CPU11；不是计算完成或停止。其仅负责预检/后续提交的parent975242经命令/starttime核验后终止且确认不存在，子进程终止信号pending，原测量作业59592166未动。改用确切test模块复核后再提交，未因观察超时重复作业。22:45实际Slurm仍只有测量作业59592166、PENDING、无节点；p校准控制器728654仍实际存在、CPU7单线程。主MCMC仍0/39，IR仍未完成。

22:13阶段更新：22:10只读复核作业59592166仍PENDING、尚无计算节点，post完成131/700（fiducial53、LC_p39、LC_m39）。全部131份完成记录均为login26单核；单盒总耗时中位480.580秒，重构24.497秒、输入P23.444秒、post-P23.700秒、B及matched self/zero诊断402.101秒；实际JAX0.9.1/CPU后端。详见`CPU_MEASUREMENT_TIMING_SNAPSHOT_20261009.json`。它不是CPU/GPU同条件benchmark，也不是MCMC或IR完成时间承诺；8盒并行约需40GiB，不能忽略登录用户总内存限制。

39组no-IR主case及18组同weighted-tracer输入P控制的数组组装接口已补齐，保持mean100/C_single500/V1及完整P-B交叉子矩阵。旧9项接口测试NERSC通过；冻结入口按实际44-bin独立回放报告的状态名称核验，并绑定同一系数和合同SHA、660节点及88细网格探针。新增4项保护测试连同原接口测试在NERSC共13项通过（单核CPU11、现有生产环境）；本地不含NERSC专用采样器的8项case/冻结保护测试通过。缺完整700/C500、收敛physical-p或真实post-weighted噪声审计时不创建主MCMC目录；这些接口测试不能替代STO有限PNG随机项的科学验收。当前没有正式case冻结、主MCMC或IR结果。

20:56阶段更新：完整Fid0..499输入核验在20:42:43完成，500/500通过、0失败，原核验PID683117已实际退出。real471使用原producer/既有DM的同配方重放metadata，16数组及整STO文件SHA完全相同，实际weights仍用原缓存；原v2等待记录保留且PID536207已确认退出。独立准入核验于20:54:04完成，报告`FID500_ADMISSION_COMPLETION_AUDIT.json`逐行核对500份实际measurement admission、冻结源码/原FoF/STO/metadata哈希及目标0..99回归；425无标记字节、246同源FoF副本、127/293/471 metadata路由均显式记录。旧target的N2/N3派生报告最大相对末位差4.44e-16，文件哈希完全相同；保留首次逐位比较的误拒绝记录，仅跨旧报告的派生矩使用8个float64 epsilon舍入门，全部原始字节SHA仍精确。没有新权重、较少covariance盒数、covariance平均或MCMC放行。

`continue_sto_post_physical_p.py`已在login26 CPU7启动（PID728654），等待确切作业59592166；只有实际Slurm成功完成、700份及C500审计/矩阵哈希通过后，才自动冻结并运行必要的physical-p校准，复用原采样器/先验/收敛门，数值库单线程。它不是正在运行的MCMC，不替代39组主链或IR控制。四项准入/衔接保护测试在本地与NERSC通过。32核/60G/shared CPU测量作业仍待调度，已完成post测量仍131/700，主MCMC仍0/39。

20:32阶段更新：用户明确将重构和P/B测量迁到计算节点，并指定32核、shared、60G；MCMC仍在登录节点且最多8核。沿用既有STO+Marisa的`desi/cpu/shared`提交配置，作业59592166由Slurm路由到`shared_milan_ss11`。请求为CPU32/mem60G；因该partition每核内存上限1905MiB，调度器将最低CPU预留调整到33，程序affinity仍最多32逻辑核。单盒实测VmHWM约5.02GiB，所以保守保留6GiB/盒和4GiB余量，最多9个原单核测量worker；没有改变原数值库线程、网格或测量源码。

原target/covariance两个dispatcher已停止派发，六个在途worker自然完成后，逐盒DONE/contract/measurement/halo哈希核验，再终止旧parent；已在原login26核实PID135656/457808均不存在，迁移记录为`outputs/sto_marisa_post_rsd_mean100_v1/compute32_migration/MIGRATION.json`。当前131/700个DONE产物，不创建部分C500。作业已于20:31:32释放挂起进入调度队列；20:31:58状态PENDING、Reason=None，尚未核实RUNNING，同时Slurm的scratch/SCRATCH许可证当前全部被集群预留。新队列逐一验收已完成盒，再复用原target/covariance单盒入口；完整700份后同作业自动汇总matched与nonzero诊断两套C_single，MCMC仍未放行。迁移/汇总/统计/p校准20项相关测试在最终配置下本地与NERSC通过；首次迁移watcher因误用系统Python3.6失败，日志和失败状态保留，改用已有production Python后释放作业通过，不安装包。下方登录节点RUNNING记录均为迁移前历史快照。

19:42阶段更新：用户已更新现有SSH证书，有效至10月10日19:07:16；实际SSH及login26访问成功，未重启原target队列。最新确认target97/300盒完成。正式剩余Fid100..499协方差测量队列已于19:10:44启动，login26 parent PID457808，当前两个单核worker固定CPU12/13，已完成6/400；Fid0..99复用target。同一目标队列确认结束且parent进程不存在后，协方差队列才使用空出的CPU7/8/9/10扩到六核；整个NERSc任务仍保留输入审计和旧reader预算、最多8核。

real293重放结果现已核验：16个数组与整STO文件SHA`8b5b6a6af473ee51e2662de9f3a7fcadf9f2380e1ae729ec3e5a00f6144faf49`全部相同，原权重不替换。routed_v1来源核验已到425份通过，随后在real425的冗余DONE标记读取等待；不是FoF/STO/diagnostics数据缺失。该旧reader按命令身份确认后终止、PID已确认不存在。routed_v2在CPU11重新验收全部必需数值输入，仅425不使用DONE字符串，SHA显式null；不伪造标记字节、保留全部recipe/FoF/halo身份/RSD门。新的通过行只追加到measurement准入目录，不改已被测量引用的旧行。

`assemble_sto_post_measurements.py`已部署，严格要求700个唯一catalog全齐，再复用`post_rsd_ensemble`生成Fid500未平均C_single。matched和nonzero诊断B分别汇总，不擅自混成一个likelihood；P-only/B-only/P+B按各自协方差子块和维数作Hartlap。8项汇总/统计相关检验在本地和NERSC通过，实时check-ready确认不齐时不创建covariance。新主MCMC仍0/39，物理p、噪声基、同基线IR验收及后续全部推断尚未完成，goal保持active。

19:00阶段更新（本地记录，SSH恢复后同步）：用户已明确批准正式covariance采用原实空间fiducial0..499，接受与fiducial target0..99重叠；`ENSEMBLE_SELECTION.json`记录显式overlap、700个唯一catalog、V=1、C_single/ddof1、不除以100/500。旧pre不改。新post no-IR采样Case接口复用原采样器/HDF/RNG/全部收敛门；五节点B与完整joint白化三项检验在本地和NERSC通过，原stochastic未审计时禁止静默复用。协方差剩余400盒的逐盒PASS准入/同字节IO/CPU租约队列已在本地完成，四项调度测试和另外五项相关测试共9/9通过；尚未成功上传或启动，不把本地准备误报为远端运行。

最新已核验的target状态为17:16:05时28/300盒完成且队列继续运行；不能当作现在的实时计数。完整来源v4已验收到0..292共293份，real293的旧diagnostics读取等待。其原producer重放首次因未加载现有MPI环境失败，日志/部分产物/源码快照保留；加载原NERSC环境的v2尝试已发起，但其执行结果目前未核验。追加metadata路由入口已本地准备，只有v2全数组及整STO文件SHA一致才准入，不替换原weights。

当前新SSH连接被拒绝：现有NERSC SSH证书于2026-10-09北京时间17:38:19到期。Globus更新与此是不同认证；没有因此宣称旧远端进程终止或重启它们。现有sshproxy已在本机3000端口发起FedID续签，输出写独立`.ssh`临时目录，不保存密码/OTP到仓库；等待用户在浏览器完成登录。新主MCMC仍0/39，C500/物理p/噪声及IR全域验收仍未完成。

17:02阶段更新：Globus认证已恢复，real246的原FoF同源单文件恢复完成；独立checksum同步比较读取25,985,172字节、文件因checksum相同而跳过、没有二次传输。原FoF质量阈值reader验收，309347个halo与原STO位置逐行精确相同、无重排或容差，原速度/RSD转换不变；见`input_repair_fid246/AUDIT.json`与`TRANSFER.json`。原文件未覆盖、无新N-body或DM。完整Fid500输入核验v4已在login26 CPU11启动，仅246使用验证副本、127延续同payload的metadata证据；旧v3等待记录保留。

目标批量测量已于16:57:43实际启动：`target300_measurement_queue`，login26 parent PID135656、四个单核worker固定CPU7/8/9/10，数值库单线程、独立session/服务器文件日志。复用校验通过的20盒，排队三种真值0/+100/-100各100盒，保持rec256/meas256/nran30/matched-self及零腿诊断；现场日志确认首批±100盒已到B_post。新的同字节暂存IO适配本地与NERSC两项检验通过，不改原重构或P/B数值入口。此为MCMC前置的完整数据测量，不代表新post主MCMC已启动；主链仍0/39，mean100和C500未完成。协方差仍为V=1单盒散布，不除以100/500。

16:35阶段更新：完整44-bin no-IR确定性p-aware模板于15:41完成，16:10独立验收回放全部660节点和88个细网格探针通过；最大误差分别4.925e-14和4.226e-13，原门2e-10保持。汇总系数SHA为`a8aea52605660b4fb3a0fccf69452f57964508d3fa3d0a47bc3808de368695f2`，见`noir_paware_templates/INDEPENDENT_COEFFICIENT_AUDIT.json`。物理p校准接口及三项独立检验已完成，真实p尚未拟合，post C500及主MCMC仍未完成。

匹配pre单盒已于15:09验收通过：分箱边界、计数和样本指纹相同，raw B回放最大白化误差4.730e-6，小于原数值尺度1e-5；浮点k均值标签差8.285e-6在原5e-5范围内。首次过严的逐位标签检查失败产物保留，零腿及self诊断已保存于`matched_pre_pilots/fiducial_real000_meas256_ran30_v2/`，旧pre不覆盖。real127的原STO在已有DM及原producer下重放后16个数组和整文件SHA全部相同，新的metadata证据仅用于real127、权重继续读原缓存。

完整输入检查当前为`fid0_499_raw_source_audit_v3`：0..245共246份通过、0失败；real246原FoF文件在Lustre OST61的读取仍等待，检查过的既有位置修复目录没有其FoF副本。已有原Globus下载脚本明确给出同一realization/snapshot的源地址，正在核对单文件恢复方式；没有提交批量下载、更换realization或把246份当作Fid500。全部完成状态和后续MCMC仍以实时证据为准。

14:45阶段更新：20盒nested random30/60配对控制于14:16全部完成，14:25整批哈希、样本和统计验收通过；见`pilot_fid20_random60_diagnostics/AUDIT.json`。20盒post halo坐标和权重全部相同。以原20盒的逐点单盒std归一，配对平均差最大为B(kmax=.20)的0.04389、P0(kmax=.08)的0.02505。这里只报告数值控制的均值和诊断误差，不是全C500白化、fNL偏差或随机点收敛认证；没有平均主covariance。新矩阵SHA为`882c449df59538e6fd4cbb018a904ea06d06482b71e66c28ebd8944a5f605b85`。

同一阶段，`noir_paware_templates`在login26 CPU10持续构建44-bin确定性模板，14:43观察到24/44完成；五个b1节点、三个物理p节点与两个非节点/finer-grid探针使用同一个原FFT源，误差门2e-10。物理p尚未校准，数值p轴不是新增先验或自由参数，模板也不替代stochastic验收。`matched_pre_pilots/fiducial_real000_meas256_ran30`在CPU11测量新的pre控制：复用原pre入口和同一post self adapter，单独保存零腿反事实，并要求与冻结旧pre的raw B核对。旧pre数据和结果不覆盖；主MCMC仍0/39。

候选0..499输入审计的最新入口为`fid0_499_raw_source_audit_v2`：截至14:37，0..126共127份全部通过、0失败，real127的原`diagnostics.json`读取仍未返回，尚无500份完成报告。此前两次payload审计额外发现核验器误拒绝精确周期端点L；独立重读real46/66/82/86/110确认每份仅一个L端点、没有真正越界。已核对PID身份并终止有此错误的两个审计，原日志和源码快照保留；修正仅允许精确[0,L]输入，由原生产wrapper处理周期像，没有扩大浮点容差或修改数据。原FoF质量选择/RSD转换和STO行身份核验全部保留；可选旧位置/质量导出不再作为必要输入。12项相关本地回归通过，新审计前100份也重现了先前source哈希和RSD检查。

2026-10-09（北京时间）goal 仅推进 **STO+Marisa post-recon RSD**，pre 保留作对照，不生产其他方法。当前goal已恢复为 **active（复查原实空间0..499输入，尚未改变样本合同）**；下文03:32的blocked记录是历史检查点，不是当前工具状态。推导在 `docs/STO_MARISA_POST_RSD_DERIVATION.md`；阶段产物位于 `outputs/sto_marisa_post_rsd_mean100_v1/`。已补固定非零 fNL_rec 的 native 接口和复用原 pre FFT 引擎的 post 投影；42 项 source/FFT/坐标映射/weighted factorial/IR/噪声候选投影/缓存适配回归通过。新库为 `libmarisa_tree_rsd_fixedrec.so`，原 pre 库和三套 pre 最终结果 SHA 未变。no-IR post 位移项最高含 b1^4，模板需五节点四次插值；IR 指数依赖 b1 后不再是精确四次多项式，不能继承这个插值阶数的认证。

10:09输入口径澄清：RSD不要求新N-body realizations；本流程复用原始FoF位置/速度与原实空间STO权重，在RSD加权密度场上重算Marisa，再重新测P/B和covariance。1000..1499是助手为与验证集不重叠而选的范围，不是RSD理论要求。原实空间使用fiducial0..499协方差与0..99均值target；若恢复这组编号，需明确其验证样本重叠，且LC_p100..249物理p校准也不能继续称为与协方差相位保证独立。候选输入只读核验正在运行，已到前50份，后续观察到Lustre I/O等待；real499的原FoF完整header/length和对应STO文件已确认，但不能据此声称全部500份已验收。未更改生产样本、V=1或不平均协方差的约定，未下载新数据或启动主MCMC。

独立小型加权粒子的实际 Marisa paint/smoothing/FFT/readout 审计通过：双精度位移相对最大误差约 2e-15，单精度约 9.1e-7。这是离散算子数学核验，不是生产分辨率 alias 或完整随机项验收。实际重构 alpha_rec 与物理源 1/M 相差约 0.929%；加密 alpha 表后仍存在，已单独实现冻结 W_rec/D_rec 响应接口，物理 Z1/Z2/primordial 继续使用原 M，不改旧谱表或 PNG convention。

10:19补充统计与P0接口验收：`post_rsd_ensemble.py`将样本实体与统计角色分离，支持显式选择700个唯一实体的重叠方案或800个不重叠实体；不会自行授权选择。P/B按同盒key组装全交叉协方差，P-only/B-only从C取边缘子块再按各自维数做Hartlap，不截取joint precision。`post_rsd_p0_model.py`复用原线性P0源并显式传物理p，不将p_rec当作校准。新增7项回归通过；`POST_STATISTICS_AUDIT.json`用哈希核对的旧800盒重现21份P/PB似然数组，另核验18个B边缘子块。没有重放本地缓存不全的18份独立B-only似然，亦不把旧pre当作新post数据；未测新post covariance或运行主MCMC。

10:28补充实际测量：`pilot_fid20_queue`在login26启动fiducial0..19的20盒population pilot，parent PID1256396，两个worker分别硬绑定CPU10/11，各库单线程。它只使用已验收的原target范围输入，不决定主covariance编号；调用原pilot数值入口而不复制/修改重构算法。设置rec256/meas256/nran30、matched self和单独零腿反事实，nested随机命名供后续30/60配对。10:30实际两个worker均已到B_post，RSS各约3.3GiB；队列状态和已完成数以其STATUS.json及对应DONE为准。这20盒仅用于pilot散布/数值控制，不能替代Fid500协方差或当作主MCMC结果。

11:00中断恢复：聊天中断关闭SSH stdout，原controller在打印完成通知时触发BrokenPipeError；原parent和最后worker均经进程检查确认已退出，实际0..5共6盒已完成。原controller、失败STATUS和日志全部保留。`pilot_fid20_recovery_20261009`在login26以独立session恢复（parent PID1368864、PPID1），stdout/stderr写服务器文件；校验并复用前6盒，只调度剩余6..19。原数值入口与固定参数未改，仍两个单核worker。当前状态应读取新队列STATUS，而非旧失败队列。新增`audit_sto_post_fid20_pilots.py`要求20盒全部完成后才汇总来源、分箱、扣除关系、均值和单盒std；已实际验证6/20时返回NOT_READY且不创建产物。它不生成主协方差或放行MCMC。

11:46恢复队列20/20正常完成；12:30整批验收通过，见`pilot_fid20_diagnostics/AUDIT.json`，20盒矩阵SHA为`12ac41e98fec87dc9050151b9061ecf0f12c7c4205d90d063ac889e2e407b003`。单个fid000配对、以这20盒的单盒std归一，kmax=.20时半格原点平移最大差0.0113，nested random30→60最大差0.2833。这不是fNL偏差、全协方差白化或均值收敛证明。因此12:47启动`pilot_fid20_random60_queue`补同20盒的配对随机点控制；复用已有real000，只新增1..19，保持数据坐标、STO权重和固定重构参数，两个单核worker、独立session及文件日志。主样本和Fid500未更改。

13:08实际B窗口验收：`estimator_B_window/AUDIT.json`从原粒子准备与生产BinMesh3SpectrumPoles导出123875个拟合shell格点。全部格点归属与现有FFT理论相同（0个差异）；全部120个实际pair counts重现测量，44个拟合bin的理论counts全部匹配。旧参考文件在17个拟合bin的counts与当前CPU测量不同、最大相对差0.004448，但这不表示当前FFT有该偏差；未修改理论、测量或旧参考。新window SHA为`a56a4ca6384e5835689ff8266bb45c3425891f9b16f05a56f55dd6470663ee2a`。

输入I/O补充：原login23 reader定位到旧source real065/post_sto_halo.npz的stat等待；同一文件在login28正常stat且文件存在。使用未修改的盘点实现，在独立目录`fid0_499_login28_recheck`作节点交叉复核；该读者已走到100行后也出现Lustre I/O等待。尚无全部500份最终报告，不能将这些等待标成文件缺失或全部验收通过；两个旧读者均不因观察超时而被宣称终止。

现已完成 9 个单盒 pilot：0/+100/-100 各一个 rec256；fiducial 额外 rec125、rec512、matched-self、零腿诊断、半格原点平移与 nested nran60。均固定 b_rec=2.2280546353635335、p_rec=1.2822、fNL_rec=100、R=10，data/random 使用同一 disp，实际 LOS=z 的物理 RSD 不去除。当前环境重生成 fiducial000 的 matter-assisted STO 文件与旧文件 SHA 完全相同。源生成调用第一部分现有入口，不改原缓存或算法；LC_m 100/100 权重已补齐。`VALIDATION_SOURCE_AUDIT.json` 于北京时间02:34:57完成三真值合计300份源的配方、权重和原始FoF行身份验收，但不等于300份post P/B已测完。

`PILOT_NUMERICS.json` 记录：同一 fiducial000 的 rec256→512、kmax_B=.20 的 B 向量相对 L2 差为 2.62%，所以不能宣布网格收敛；它同时包含 CIC 窗口变化，不能全部称为 alias。matched-self 的同项差约 .114%，post halo 坐标 SHA 与 legacy 完全相同。这里是未白化单盒差异，不是 fNL 偏差或 sigma。原 S113 的 bin-center j0 与 raw/shot paint 不一致已通过独立点过程测试定位；显式 context 使用实际离散 shell windows 和 matched TSC/interlacing3，在含对角/零模的三对 toy bins 中最大归一化误差 6.79e-7 < 5e-6。旧失败测试和 legacy 数据保留；若正式采用修正，需匹配的新 pre 对照。

`PILOT_STRESS_AUDIT.json` 在固定 rec256/CIC 响应下量化网格原点：半格平移后44点 B 相对 L2 差 .0655%，halo 位移差 RMS 每轴约 .022–.023 Mpc/h，而坐标平移往返舍入仅约 1–2e-6 Mpc/h。相同随机种子、30倍随机点为60倍的前缀，nran30→60 的 post halo 坐标完全相同，但单盒 B 向量差3.36%、P0差 .342%。这区分了网格相位和有限随机采样的影响，不等于均值偏差或 fNL 偏差，仍没有 covariance whitening。

`ZERO_ATOM_AUDIT.json` 独立枚举验证 factorial B(0,k,-k) 不必为零；新增诊断没有改变旧 matched pilot 的任何22个原数组或 post halo SHA。若将零腿 numerator 排除但保持 full-shell denominator，真实 fid000 的44点 B 相对 L2改变量约 .0803%，最大单 bin 改变量约其 B 的1.05%。该反事实单独保存，正式 estimator 选择尚未冻结，不能混用旧 pre。

确定性 LO IR 已从小批量 oracle 推进到复用原 FFT 的 tagged-factor 投影。`IR_ACTUAL_SPECTRUM_AUDIT.json` 使用原 P/M 和已冻结 smooth split，实际重构 alpha 独立固定：harmonic/direct tensor 误差4.80e-12，径向513对1025节点的有向源误差2.65e-8；首轮低/高两对 bins 在 theory mesh144/160 上误差小于7.2e-14。`IR_GRID_ACCELERATION_AUDIT.json` 的精确网格对称加速约9–10倍，与未加速投影差小于5.86e-15。后续已扩展至同一数值探针的全部44 bins（见下）；这里的 theory FFT grid 不是 catalog reconstruction grid，尚未认证全参数插值、post stochastic 或科学 IR 效果。范围为 Gaussian long modes、strict O(Pw)、fNL²，不包含高阶 PNG 累积量。

带权有限 random 的独立坐标/集合分割审计支持一阶 shift 的有向 PN2 候选，明确仅用 data 的 N2；新增 `post_rsd_poisson_projection.py` 显式复用同一 FFT 将候选投影成 G/L/Q，不修改确定性源或旧 likelihood。它不是实际 STO 非 Poisson/有限位移的完整随机项。理论代码和点过程通过不能标成 mock recovery。

`CALIBRATION_INPUT_AUDIT.json` 查明旧物理 p 校准使用 GSR1，而当前 target source 为 GSR2；fid000/LC_p000 的权重相对 L2分别差约7.52%/7.98%，因此不直接沿用旧 p=1.33946 或旧 post p=1.28276。算法 p_rec=1.2822仍冻结。相同 GSR2 配方的独立 LC_p real100..249 的 weighted-input RSD P0 已150/150测完，作为物理 p 校准输入；它们不属于主 mean100 或 Fid500，也不是新增 STO-only 科学对比。`calibration_holdout_p0/HOLDOUT_COMPLETION_AUDIT.json` 于北京时间02:49:38验收通过，核对全部raw/source/测量哈希、12个P bins与窗口；保存150×12矩阵及均值，没有生成协方差或拟合新p。该队列已COMPLETE、无活动worker。

全部44个实际B bins的确定性IR双theory-grid审计于北京时间03:29:49完成，`ir_full_registry_audit/REGISTRY_AUDIT.json` 通过；独立复算最大缩放误差1.44e-12，按|fNL|<=500缩放各阶系数后的最大误差8.64e-13，均低于预设1e-9门。仍只检查一个数值探针，不等于全b1参数空间或科学IR有效性。每个bin使用满足原no-wrap不等式的两个网格，保留完整shell分母和无第三腿cut，每格checkpoint受source/contract哈希保护；计算进程已正常结束。

`IR_PNG_TRUNCATION_DIAGNOSTIC.json` 额外比较strict fNL² dressing与同一Gaussian tensor的未展开指数（后者仅为部分高阶敏感性参照，不替换当前模型）。32个有向闭合三角形、p=1.31探针中，b1=2.228、fNL=±100的最大总B差/绝对source尺度约5.75e-5；先验角落b1=4、fNL=500有一条二次阻尼因子为-0.00476，总B差/该source尺度最大约3.39%。该尺度不是B自身或sigma，未作实际estimator投影/协方差白化；不能据中心探针放行全先验，也未裁剪负值、缩窄先验或增加高阶模型。

所有数值 worker 单核、各库单线程；校准队列最多三核，与源队列合计最多四核，任务总限仍八核。没有启动 post MCMC，没有生成 PNG/PDF 科学图。

**正式 post ensemble 生产和 MCMC 尚未放行。** 独立协方差 fiducial real1000..1499 的 500 个 halo 存在，但当前配置路径的 DM 和正确配方缓存均缺失，已向用户请求其路径。另见的 fid4000 缓存是 halo-only lam3，不能替代 matter-assisted env_lam1_gsr2_thr15；没有更换 IC、平均 covariance、减少 Fid500 或下载新数据。主合同仍为每真值 mean100、独立 Fid500 Csingle、V=1、六档 B cutoff 和 Pmax=.08，计划18 B-only +3 P0-only +18 P+B。后续仍需实际生产网格/alias、shift-noise、独立 p 校准、mock/MCMC 与 RSD IR 生产验证。

`COVARIANCE_INPUT_RECHECK.json` 于北京时间03:15逐盒复查：raw FoF500/500，正确配方输入0/500；另500份缓存全部为不同配方（halo/halo-split、GSR1、lambda3）。`INPUT_DEPENDENCY_GATE_AUDIT.json` 于03:31独立复算44-bin误差、重验源/旧库/旧结果保护与统计合同，明确完整目标未完成、主MCMC0/39。该输入缺口已跨多个goal回合确认；goal工具于03:32:43返回blocked。需用户提供指定DM/同配方缓存目录，或明确授权新的数据/样本方案，再重新核验后续门；没有把局部通过缩写为任务成功。

2026-10-08 补充 **RSD P0-only**，与现有 RSD B-only、P0+B000 构成三方对照。新输出为 `outputs/pre_rsd_p0only_mean100_v1/`。这里的 P0 是 pre-recon 红移空间单极矩（z=0.5、LOS=z），模型仍包含 Kaiser 与 finite local-PNG 项；kmax_P 固定 .08 h/Mpc、12 点，每种真值 0/+100/-100 只新增一组 MCMC。相同 P0-only 结果在六档 kmax_B 的表格中重复用作固定参照，不算六组独立拟合。

P0-only 直接复用联合实验冻结的 P0 测量、actual CPU window、理论模板和对应先验。仅拟合 fNL、b1、Pshot_residual_nbar 三参数；与 P+B 相应参数 bounds 完全一致，B 专属 flat nuisance 参数在 P-only 中积分为常数。data 为原 mean100；covariance 为 Fid500 的未缩放 C_PP 子矩阵，Hartlap 按 12 维计算为 486/499，未使用联合 precision 的子块或条件协方差。三项预检验证了真实 covariance、全部六档联合 P 输入/先验与随机参数点的模型 parity。

新增 `run_rsd_p0only.py` 适配三参数 MAP/残差，复用原 `run_rsd_b000_mcmc.run`、提案、RNG、checkpoint 与独立 ensemble 诊断；`run_rsd_p0only_login.py` 在 login06 的 CPU [7,8,9] 上自动运行三组，数值库均单线程。初次启动遇到 taskset 尚未执行就检查 affinity 的竞态，采样前停止；父进程显式完成单核绑定后恢复。原控制器源码、失败状态和启动记录可恢复保留，采样入口与冻结输入不变。

共享 jaxpower 的 mesh2.py 在 10 月 2 日后更新；`UPSTREAM_REUSE_AUDIT.json` 明确记录旧/现 SHA。本次不执行该 live binner，复用的 P0 ensemble、window、模板均与原冻结 SHA 一致，其他原输入/源码未变；旧 P+B 的严格 parent guard 保留原样。`report_rsd_p0only.py` 对 Pshot 在原 bounds 内解析积分，再做 fNL/b1 网格积分，与三组 MCMC 独立核对，输出 JSON、三方对照表与 PDF。实际完成状态以新目录 LOGIN_STATUS、COMPLETION_AUDIT 与 RESULTS 为准。

北京时间 2026-10-08 19:25:20 全流程 COMPLETE，3/3 新案例均在 6000 步通过原收敛门，最大 rank Rhat 为 1.0004107，无活动 worker。固定 p=1.179、growth=0.7625790037720044，真值 0/+100/-100 的 RSD P0-only fNL 分别为 **1.5±100.2、104.4±98.9、-106.8±101.4**（后验中位数±标准差，不是对称 68% 区间；实际分位数见 RESULTS）。Pshot 解析边缘化加二维网格的独立核验 PASS，MCMC 中位数差小于 0.01 sigma，标准差相对差小于 1.1%。COMPLETION_AUDIT、三项回归与两页 PDF 的文本/页面边界/哈希检查均通过，未生成 PNG，未做栅格视觉验收。

三方对照见 `outputs/pre_rsd_p0only_mean100_v1/REPORT.md` 和 `RSD_P0_B_PB_COMPARISON.pdf`。P+B 相对 P0-only 的 fNL 后验标准差，在 kmax_B=.10/.14/.20 时分别缩小约 6-7%/11-13%/14-16%；先前 P+B 相对 B-only 的约 40% 改善不能解释为相对 P0-only 的同等增益。高 k 的共同正向偏移、联合宽先验对照尚缺等限制仍保留，不据此认证高精度 kmax；旧 B-only/P+B 和实空间/post-recon/IR 结果均未覆盖。

2026-10-02 用户新增 RSD `P0+B000` 控制实验。独立目录为 `outputs/pre_rsd_pb_mean100_v1/`，不覆盖下述 B-only 结果。P0 沿用实空间基准的 12 个 edges/数据点，`kmax_P=.08`；B 保留 `.08/.10/.12/.14/.18/.20` 六档、三种真值的 18 组主扫描。P 与 B 必须来自同一份 FoF、相同随机种子，补测同一 800 个 RSD 样本的 P0，以 Fid500 建完整 P-B 交叉 covariance。仍为 mean100 target + Csingle、V=1，绝不除以 100/500。

联合实验保留原 B 的全部 7 个参数、bounds、p、growth、数据、系数和 stochastic 形状，只增加独立 `Pshot_residual_nbar` 一个残余常数参数。P 模型是离散格点的 `(b1+f mu²+fNL bphi/M)² P_lin + Pshot`，完整保留 finite-PNG 二次项；不把 P-derived b1 prior 替代完整联合 likelihood，也不丢交叉 covariance。IR/FoG/loops/counterterms、B 的 bphidelta recipe 与 finite-PNG stochastic closure 均不改，后两者仍是需要另查的模型系统性。

预检实际发现 CPU float32 的 P shell 边界与旧 GPU mode counts 不同。未按旧 bin center 近似，也未修改共享 estimator：`build_rsd_p0_window.py` 从实际生产 `BinMesh2SpectrumPoles` 导出 9170 个有效格点标签、整数坐标和权重，六个谱模板直接通过原 binner 回归，实测 12 个 mode counts 全部一致。`P0_WINDOW_AUDIT.json` 独立记录该数值门，`P0_WINDOW.npz/P0_LATTICE.npz` 与输入/源码 SHA 一起冻结。失败的启动前合同与一个 P checkpoint 可恢复保存在新输出的 `preflight_versions/`，不参与似然。

北京时间 2026-10-02 13:39，`run_rsd_pb_login.py` 在 login12 启动（原 PID 2124717），硬 affinity 为 `[7,8,9,10,11,12,13,14]`，每个 worker 单 CPU、各数值库单线程。依次自动执行 60 P pilot → 3 个 mesh512 → 数值门 → 800 P 测量 → 联合冻结/回归 → 18 MCMC → JSON/CSV/PDF 与完成审计。测量阶段遇到该登录用户 cgroup 的 30 GiB OOM 上限；保留已完成的 793 份测量、原启动/失败记录和原控制器源码快照，仅以每批 10 份、最多 4 个测量 worker 补齐缺少的 LC_m real093..099，模型、数据选择、协方差、先验及采样器均不改。资源恢复控制器 PID 156688 于 14:52 启动，MCMC 仍最多 8 workers；15:09:03 全流程 COMPLETE、18/18 收敛、无活动 worker，完成审计 PASS。采样入口 `run_rsd_pb_mcmc.py` 仅作显式 Case/合同/输出标签适配，复用不修改的 B-only sampler 的提案、RNG、checkpoint 和全部诊断；预算上限 64000，实际全部在 8000–12000 步达到原门。OOM 来源已核验，内存增长的具体机制没有唯一定位。

2026-10-02 用户明确纠正统计口径：**只对 data vector 做 realization 平均，likelihood covariance 使用 Fid500 单盒样本散布，不除以 100 或 500**。B-only 主结果目录为 `outputs/pre_rsd_b000_bonly_mean100_v1/`，联合实验另存 `outputs/pre_rsd_pb_mean100_v1/`；100 份 validation 的均值中心与目标体积 `V=1 (Gpc/h)^3` 分开记录。Hartlap 修正是有限协方差样本数的修正，不是体积缩放。

旧 `outputs/pre_rsd_b000_bonly_v2/` 的 mean100 拟合实际使用 `Csingle/100`，其 V=100 结果和旧 single000 对照只读保留，不再作为用户要求的主扫描或 V=1 的有效 kmax 结论。新入口 `run_rsd_b000_inference.py correct_covariance` 复用已完成的 800 份测量和旧均值/理论模板；18 组新基线仅改 covariance/Hartlap Cholesky/体积 metadata，保留全部 7 个参数、先验、p、growth、bins。新 MCMC 从头采样，不恢复旧链。

`COVARIANCE_CORRECTION_AUDIT.json` 逐组断言 covariance 等于 `np.cov(Fid500单盒数据, rowvar=False, ddof=1)` 的对应子矩阵，且 data、coefficients、noise、bounds、indices 与旧均值基线完全一致。全档位为 `.08/.10/.12/.14/.18/.20`，`.15` 仍为 `.14` 的同索引别名；每种真值 `.08` 与新候选边界的 nuisance 扩宽测试仍按原规则执行，最多 6 组。

用户随后明确要求在登录节点运行、不得超过 8 核。`run_rsd_b000_login.py` 在确认本次四个 Slurm 作业仍 PENDING 后取消它们，保留原提交来源；整个新进程树绑定到最多 8 个逻辑 CPU，每个 MCMC worker 绑定其中一个 CPU，BLAS/OpenMP/NumExpr 全为单线程。最多 8 个案例同时采样，基线→先验对照→PDF/profile/完成审计串行依赖不变。实际 host、PID、CPU affinity 和每个活动 worker 的实际允许 CPU 集合记录在 `LOGIN_LAUNCH.json/LOGIN_STATUS.json`；不修改其他任务的作业。

初始 24000 步预算结束时，18 组中 15 组通过；三个 `.08` 案例仍未满足 `retained length / max(tau)>50`，因此明确停止后续步骤，未放宽诊断门。新增显式 `--max-steps/--extend-budget`：只给未收敛链增加 48000、必要时 64000 的总步数上限，续用原 HDF RNG/checkpoint、MAP、data/covariance/先验。原始 sampler 源码快照 SHA 固定；AST 逐项证明 Case、diagnose、checkpoint 逻辑及 proposal/RNG/backend 内循环完全相同。原 `RUN_PROVENANCE.json` 不覆盖，扩展来源另存 `BUDGET_EXTENSION_PROVENANCE.json`，初始未收敛诊断也单独保留。`recover_rsd_b000_login.py` 仍只在原八核池内执行，延长完成后复用原流水线，不重复采样已收敛组。

2026-10-01 已添加独立 C++ plane-parallel tree RSD 核、Python 批量接口、逐项原生核验收，以及真实 Quijote halo 的 pre-RSD 样本和单盒模型测试入口。复用第三部分的 bias-v1 类型，保持原有 real-space 核/API 不变。

当前科学范围是严格 tree deterministic local-PNG，保留 `G + fNL L + fNL² Q` 和完整 RSD–PNG 交叉项。数据演示从 pre-recon 开始，不能据此声称已经验证 RSD reconstruction 或完整 RSD IR resummation。

## 建议目录

- `code/`：RSD 模型、生产入口和审计脚本。
- `docs/`：理论背景、推导、任务书和结果分析。
- `outputs/`：遵守根目录约定，存放新样本、测量、审计 JSON、NPZ、日志和 PDF 图。
- `results/`、`figures/`：原预留目录；本次新示例产物统一写入 `outputs/`。
- `work/`：中间缓存、临时日志和可恢复的工作产物。

## 研究边界

RSD 版本应明确区分 tree-level、one-loop、IR resummation 和 reconstruction response，不能把 real-space 的结果直接当作 redshift-space 验证。每个结果需要记录红移、\(k_{\max}\)、triangle/bispectrum binning、速度或增长率约定、协方差体积、Hartlap 修正以及是否使用 CIC 或 adaptive reconstruction。

本部分的首次科学验收应先完成低 \(k\) tree-level RSD 闭合和 mock recovery，再进入更高阶模型或正式联合约束。

## 使用入口

- [实现与数据合同](docs/TREE_PNG_RSD_IMPLEMENTATION.md)：公式、接口、投影和限制。
- [迁移/改动清单](docs/RSD_CHANGE_MANIFEST.md)：来源与独立回归测试。
- [首轮实际样本测试报告](docs/PRE_RSD_FIRST_TEST_REPORT.md)：已完成的审计、数值结果和科学边界。
- `code/build_rsd.sh`：编译独立 native 库，不覆盖旧库。
- `code/audit_tree_rsd.py`：1024 个随机方向、100 个 squeezed、旧 native 实空间极限、投影验收。
- `code/export_pre_rsd.py`：读取原始 FoF 速度，核对旧 halo 行顺序，导出 LOS=z 的 pre-RSD 样本。
- `code/pre_rsd_model_test.slurm`：单 CPU 运行验证、128/256 interlaced-CIC 测量和 strict-three-leg B0 模型诊断。

样本位于 `outputs/pre_rsd_z0p5_losz/{fiducial,LC_p}/real000/`；固定 `z=0.5`、`Mmin=1e13 Msun/h`、`L=1000 Mpc/h`、`V=1 (Gpc/h)^3`。模型示例位于 `outputs/pre_rsd_single_box/`。

本次三边 shell `B0` 与既有两腿 Sugiyama/TriPoSH `B000` 不共用 binning/covariance；新入口明确记录这一点。正式 fNL 恢复还需 RSD mock ensemble、协方差与统计验收，不能沿用 real-space covariance。

## P0+B000 完成结果与诊断（2026-10-02 15:09 北京时间）

同一 800 个 raw FoF 的 P0 已测完；100 份 target/每种真值、独立 Fid500 covariance。完整 P-B 交叉 covariance 保留，最大绝对相关系数约 0.6978，联合 correlation matrix 最小特征值约 0.10167。6 项联合回归 PASS。原 B-only 的数据、确定性系数、噪声形状、bounds 与未缩放 C_BB 子块逐项不变；Hartlap 因联合维数 12+NB 而按原规则重新计算，并非继续使用 B-only 的因子。

18/18 MCMC 全部通过原门，最大 rank Rhat = 1.0013055，最小 fNL bulk/tail ESS 分别约 5636/9161；每组四套独立 ensemble、32 walkers。所有真值固定 profile 有效，最大 truth-profile Δχ² 约 0.3165。LC_p/.08 的 Δχ² 为约 -0.000510，是固定真值优化略优于存档 free MAP 的小数值差异，不作负物理显著性解释；未改写 MAP 或链来消去这一记录。

fNL 后验中位数 ± 后验标准差如下；±std 不是对称中央 68% 区间，实际分位数见 RESULTS.json：

| kmax_B [h/Mpc] | 真值 0 | 真值 +100 | 真值 -100 |
|---|---:|---:|---:|
| .08 | -3.3 ± 95.9 | 101.4 ± 97.5 | -107.9 ± 97.5 |
| .10 | 0.8 ± 94.3 | 107.4 ± 93.3 | -103.4 ± 94.3 |
| .12 | 5.8 ± 91.0 | 110.2 ± 92.1 | -98.1 ± 91.9 |
| .14 | 1.9 ± 87.4 | 104.7 ± 86.8 | -101.1 ± 89.9 |
| .18 | 27.8 ± 88.1 | 140.0 ± 86.5 | -79.3 ± 91.0 |
| .20 | 19.7 ± 84.6 | 127.7 ± 82.9 | -89.5 ± 87.1 |

18 个实际中央 68% 区间均包含真值。与同口径 B-only 相比，fiducial/.10 从 32.4±163.7 变为 0.8±94.3（std 缩小 42.4%），/.14 从 71.1±148.3 变为 1.9±87.4（缩小 41.0%）；/.20 从 85.1±113.8 变为 19.7±84.6（缩小 25.6%）。低至中 k 的三种真值 fNL std 缩小约 36%–52%，b1 std 缩小约 62%–68%。fiducial/.10 的 b1 中位数从 1.5833 变为 1.9175，其实际 68% 区间从 [1.3131,1.8754] 收紧为 [1.8171,2.0154]；/.14 为 1.9156 [1.8362,1.9940]。高 k 的 fiducial b1 仍下降到约 1.87，fNL 仍有共同正偏。

控制实验支持：B-only 的弱 bias/PNG/stochastic 约束与边缘化几何是偏移的重要贡献，加入低 k P 后中心与精度均改善。不能把全部改善唯一归因于 b1，因为 P 也直接约束 fNL，完整交叉 covariance 和新 Pshot 参数也参与联合推断；fNL-b1 的相关系数绝对值不一定下降，不宜仅用相关系数宣称退化已消失。联合似然仍为 8 参数：旧 B 的 7 参数 + 一个独立常数 Pshot 残差，不增加 IR/FoG 参数。

尚未做联合模型的 nuisance 宽先验对照，不能继承 B-only 的宽先验结论。SUMMARY 的 prior_sensitive/boundary flags 仍原样保留，弱约束平坦参数两端各 2% 区域天然约有 4% mass；这些 flags 不是实际宽先验稳健性检验。mean100 + Csingle 仍是单盒精度的 model/forecast 检查，不是真实 mean100 的抽样 likelihood；覆盖真值不能认证高精度无偏或硬 kmax 范围。本轮 IR/FoG/loops/counterterms 全关，finite-PNG B stochastic closure 和 bphidelta recipe 未修。

只读背景诊断另存 `outputs/realspace_vs_rsd_diagnosis_20261002/`：

- 真正两腿 estimator 的 f_growth→0 Gaussian 基底对照实空间：b2_native=b2_EFT−4gamma2/3，bK2=gamma2。六个探针的最大相对差为约 6.91 ppm，严格 bitwise 级门未通过，原 REQUIRES_INVESTIGATION 状态不改。补充 `ZERO_GROWTH_NUMERICAL_IMPACT.json` 使用未平均的单盒 covariance 量化：最大白化列差约 9.50e-5，Gaussian base 的 δC⁻¹δ≈1.03e-8，远小于当前数据精度，不能解释观测到的漂移。实空间源码含 float32 径向表征和有限 r6/m120 格点矩压缩；浮点与有限 quadrature 的贡献尚未分别定位，不宣称精确逐位相等。
- 名义 kmax_B 只切两条 measured legs，不切 closing k3。/.10 的 k3 最大约 .2061 h/Mpc、约 57.81% 有效 mode-count 权重在 k3>.10；这不是 58% 信号/信息量。实空间也有相同几何，不能单独归为 RSD bug，但会让非线性速度物理的高 k 进入低名义 cutoff 成为可能。
- 对照现行实空间代码发现字面 PBS recipe 不同：其 bphidelta 的 universal anchor 使用 2δc(b1−1)，本 RSD helper 使用 2δc(b1−p)。匹配 bias 基底后，p=1.179 时 RSD−realspace 的 bphidelta = 2δc(1−p) = -0.603588；并非 native Z2 代数验收失败。本轮保持此 recipe 不变以隔离 P 的作用，后续需受控修正/比较；它不能在 fNL 真值 0 时单独生成 PNG 确定性项的真实残差。
- 实空间既有 B-only 使用固定 b1=1.93 和 Gaussian nuisance priors，本 RSD B-only 为自由 b1 和 flat priors，目标样本数/部分模型约定也不同。因此原 real/RSD 对比不是只改 RSD 的同条件实验，不能仅凭后验中心认定红移空间核写错。

最终 `COMPLETION_AUDIT.json` PASS；交付前独立从 500 个 matched 单盒向量重算完整 covariance，并核对 800 个 raw SHA/IC/random seed、18 个旧 B 子块与联合 Hartlap/Cholesky，均 PASS。RESULTS.json SHA256 为 `8abb934915798f1f84f90692240e4990127f1e181670089f7027ab1f90863d0c`，6 页 PDF SHA256 为 `bea4411b57596fc68c492f4adc1d71d498185a215eb96ed58dc688e7b95c2b33`；全部页面文本、页面边界和哈希核对，通过 PDF 技能的结构/文本 QA，遵守 PDF-only，未生成 PNG、未进行栅格视觉检查。原 B-only 冻结输入与最终结果均未修改。

## 当前 V=1 B-only 重跑最终结果（2026-10-02）

北京时间 2026-10-02 12:39:48，登录节点流水线与预算续跑控制器均 COMPLETE：18 组基线和 6 组 nuisance 扩宽对照全部通过原定采样门。基线三个 `.08` 分别在 32000/26000/30000 步通过；其宽先验对照分别在 44000/42000/44000 步通过，未使用 64000 上限，也未放宽门槛。每组四套独立 ensemble、32 walkers；24 组最终最大 rank Rhat 约 1.00160。

`COMPLETION_AUDIT.json` 为 PASS：逐组重验 100-box 均值 target、未平均的 Fid500 单盒 covariance、Hartlap Cholesky、冻结来源与预算续跑来源；旧输出 95 个冻结文件哈希未变。13 项回归 PASS，实际 host 为 login12，整个进程树绑定 CPU `[6,7,8,9,10,11,12,13]`，各数值库单线程，最大 8 个逻辑 CPU。

基线 fNL 后验中位数 ± 后验标准差如下；这里的 ±sigma 不是对称的 68% credible interval，实际分位数见 JSON/PDF：

| kmax_B [h/Mpc] | 真值 0 | 真值 +100 | 真值 -100 |
|---|---:|---:|---:|
| .08 | 20.1 ± 191.1 | 127.1 ± 170.3 | -87.6 ± 205.0 |
| .10 | 32.4 ± 163.7 | 155.0 ± 149.4 | -89.3 ± 170.4 |
| .12 | 65.9 ± 157.8 | 189.6 ± 144.1 | -60.6 ± 163.8 |
| .14 | 71.1 ± 148.3 | 180.3 ± 136.3 | -48.0 ± 152.8 |
| .18 | 91.8 ± 123.4 | 209.3 ± 120.0 | -22.2 ± 121.7 |
| .20 | 85.1 ± 113.8 | 194.3 ± 113.8 | -27.9 ± 113.1 |

18 组实际中央 68% 区间均包含各自真值；这只说明在单盒精度下未显著排除真值，不证明高精度 tree 模型偏差已消失。高 k MAP 与旧 V=100 的中心几乎相同，例如 `.20` 的 fNL MAP 约 `76.0/186.2/-33.4`，仍有共同正偏；本次没有加入 IR、FoG、loops 或改变模型。

直接先验对照只将 nuisance bounds 乘 2，fNL/b1/p、数据与 covariance 不变。通过标准仍为中位数移动 <0.1 个基线 sigma 且 sigma 改变 <10%，两条链都须收敛：

| 真值 | .08 中位数移动 / sigma | .08 sigma 改变 | .20 中位数移动 / sigma | .20 sigma 改变 |
|---|---:|---:|---:|---:|
| 0 | .0342 | +12.89% | .0059 | +1.92% |
| +100 | .1058 | +12.29% | .0184 | -0.72% |
| -100 | .0176 | +13.36% | .0093 | -0.03% |

因此 `.20` 的三组直接宽先验检查均稳定，`.08` 三组均因宽度变化超过 10% 而敏感，+100 的中位数移动也略超门槛。`RESULTS.json` 中三种真值的上边界候选均为扫描上限 `.20`，但 `scientifically_certified=false`：中间档位未逐一做宽先验扫描，有限 PNG stochastic closure、固定 p、协方差宇宙学依赖等限制仍在，不能把候选边界当作完整有效范围认证。

统计解释必须保留：平均 target + Csingle 是单盒精度下的模型/forecast 检查，不是 100-box 均值的抽样 likelihood。报告中的 chi2.sf 只能作为 forecast 权重的残差指标，不能当作均值数据的校准 GOF p-value；接近 1 并不意味着模型达到高精度。匹配 IC 的三个真值也不是三个独立显著性测试。

24 个真值固定 profile 均 Minuit valid，未触发 free-MAP 优化问题阈值。最终产物位于 `outputs/pre_rsd_b000_bonly_mean100_v1/`：`RESULTS.json/RESULTS.csv`、`FINAL_INTERPRETATION.json`、`COMPLETION_AUDIT.json`、`RSD_B000_VALIDATION.pdf`。全部 8 页文本和页面边界、PDF/JSON 哈希已核对；PDF 内的热图可含嵌入图像对象，但未生成或保留 PNG 文件，未作栅格视觉检查。

## 历史 V=100 B-only ensemble 流水线（2026-10-01，非当前主口径）

用户要求与 real-space 拟合数据点密度完全对齐后，正式观测量使用两腿 `Sugiyama B000(k1,k2)`，不再使用早期 strict-three-leg B0 草稿。原草稿只读保留并拒绝直接运行。完整边缘、shell-pair 顺序、44 个认证索引和实际旧似然合同保存在 `outputs/realspace_binning_reference/`。

| 扫描 kmax_B [h/Mpc] | .08 | .10 | .12 | .14 | .18 | .20 |
|---|---:|---:|---:|---:|---:|---:|
| B 数据点数 | 9 | 14 | 20 | 27 | 35 | 44 |

`.08` 是 kmax 扫描起点，不是删去 .08 以下数据；切割按两腿 measured shell-average k 的最大值，不切 closing third leg。`.15` 与 `.14` 索引相同，作为别名，不重复 MCMC。

新输出独立存放于 `outputs/pre_rsd_b000_bonly_v2/`。冻结 800 个源 FoF：fiducial/LC_p/LC_m 各 100 个 IC 0-99 为验证样本，另取 fiducial IC 1000-1499 的 500 个不重叠盒子建新 RSD 协方差。pilot 先跑每种真值 20 个，其输出计入正式样本，不重复测量。

测量直接调用第一部分 `common/halo_b000_pipeline.measure_b000`，保持 TSC、interlacing=3、compensation、30 倍 uniform randoms、生产随机 namespace 与既有 shot-noise 扣除；同样调用共享 `wrap_positions_to_periodic_box`，不另写重构/测量公式。不测量或拟合 P 数据。新 CPU 壳层与旧 GPU 少数边界 ULP 不完全相同，但所有新 RSD 样本、理论的完整两壳 denominator 必须逐点一致；数值门不允许混用旧协方差。

理论采用 exact-lattice separable FFT，closing third leg 无限制且防止 alias；核和分解由独立 native 三角形、直接 mode-pair 枚举、第五 b1 点与第二 theory mesh 验证。`fNL` 仍为二次多项式，`b1` 为三次，`b2_native/bK2` 为 affine。基线固定外部 `p=1.179` 与 growth，不使用 P-derived b1 prior。IR/FoG/loops/counterterms 均关闭。

随机项首先审计 native SP 的独立 lattice mu^4 形状；只有在真实 covariance 下可忽略才降为两项，否则明确增加第三个 Gaussian stochastic 形状。该审计不等于 finite-PNG stochastic closure，所有科学结论保留这个 provisional 限制。

流程：60 份 pilot + 3 份 mesh512 + 理论数值门 → 补齐 800 份测量 → full covariance/Hartlap + 冻结似然 → 18 组 mean100 与 3 组实际 single000（.12）MCMC → 每种真值 .08 与候选边界的 nuisance-prior 扩宽测试（合并重复，最多 6 组）→ JSON/CSV/单份多页 PDF 报告。mean100 的 covariance 为 `Csingle/100`，有效 V=100；single000 用 `Csingle`，V=1，不把样本数混成 forecast 体积。

MCMC 每组四套独立 ensemble、各 32 walkers，6000 起步、每次加 2000 至 24000；HDF checkpoint 可恢复，source/likelihood hash 变化时拒绝静默续跑。验收要求 rank Rhat、bulk/tail ESS、length/tau、前后半稳定性、ensemble 一致性和 acceptance。到预算未收敛也写明诊断，不算科学通过。

主要入口：`rsd_b000_contract.py`、`measure_rsd_b000.py`、`build_rsd_b000_templates.py`、`run_rsd_b000_inference.py`、`run_rsd_b000_mcmc.py`、`report_rsd_b000.py`。`PIPELINE.json` 记录已提交作业及依赖；提交成功不代表科学结果完成。数值审计失败会拦住 bulk/likelihood/MCMC，不会为通过测试修改数据或按真值挑输入。

启动时点快照（2026-10-01）：60 份 pilot、3 份 mesh512 对照及理论数值门已 PASS，三种真值的网格差异分别为 `.0187/.0285/.0103` 个 pilot mean100 sigma，配对相位相关最低 `.9910`。800 份补齐测量已自动放行。程序与数值回归 10 tests PASS，不能据此声明科学 fNL recovery。

旧 real-space 对照另行保存在 `LEGACY_BOUNDARY_AUDIT.json`：旧 GPU/new CPU 在 `k=.18607143` 的边界有 72 个模式从 shell9 移到 shell8，该边缘距 integer radius 的物理 k 仅约 `0.774` 个 float32 ULP。`.18/.20` 旧对照差异约 `.698` 个 pilot RSD mean100 sigma；未变壳层计数 bins 和 `.14` 以下差异小于 `.00022` sigma。这个审计单位不是旧 real-space likelihood 的正式显著性；没有重新跑 GPU 来证明 bitwise 来源。新数据与 double-lattice 模型计数一致，但不能把高 k 旧数据直接混入新协方差，或把此差异解释为 RSD/IR 效果。

## 历史 V=100 执行与科学结论（2026-10-02 核验，不可转移到 V=1）

Slurm 没有中途停止：800/800 正式 mesh256 测量、3/3 mesh512 对照、新 RSD 协方差、21 组基线 MCMC 与 4 组实际 nuisance 先验对照均已完成；全部 25 组通过预注册的采样收敛门。结果汇总时间为北京时间 2026-10-02 00:16，最终 8 页 PDF 于 00:17 左右完成。最多 6 组对照按边界去重后仅需 4 组：fiducial/.08，LC_p/.08，LC_m/.08 与 .10；剩余两个 array slots 明确 SKIP，不是缺失拟合。

native-SP lattice 秩审计保留 `R_stoch_anisotropic`，共 7 个参数；这是第三个 Gaussian stochastic 形状，不是 IR 引入的参数，IR/FoG/loops 仍关闭。外部 `p=1.179` 固定、b1 自由、无 P likelihood/P-derived b1 prior。

mean100（实际 `Csingle/100`、V=100）后验中位数与标准差：

| kmax_B | 真值 0 | 真值 +100 | 真值 -100 |
|---|---:|---:|---:|
| .08 | 4.1 ± 16.9 | 104.0 ± 23.3 | -90.1 ± 14.9 |
| .10 | -4.7 ± 12.4 | 83.5 ± 15.6 | -90.3 ± 10.1 |
| .12 | 41.3 ± 13.5 | 154.6 ± 16.1 | -63.3 ± 11.6 |
| .14 | 38.9 ± 12.0 | 148.8 ± 13.3 | -66.0 ± 11.2 |
| .18 | 80.5 ± 11.3 | 196.6 ± 12.1 | -31.3 ± 11.2 |
| .20 | 76.7 ± 10.9 | 186.8 ± 11.4 | -32.8 ± 10.6 |

`.08` 的基线后验包含三种真值，但不能认证为先验稳健的有效 kmax：S_stoch 的 MAP 贴基线上限 20，扩宽 nuisance 后又贴上限 40；fNL 中位数移动 `.51/.98/.38` 个基线 sigma（fid/+/-），宽度缩小约 14%。LC_m/.10 的额外先验检查移动 `.19` sigma。

`.12/.14` 三种真值都有约 `3.0-3.7` sigma 的共同正偏，Gaussian/Hartlap 描述性 GOF 明显变差（不是精确 posterior-predictive p-value）；`.18/.20` 更差。当前测试区间 `.08-.20` 没有找到同时满足精度与先验稳健性的科学认证范围。V=1 的实际 single000/.12 三组误差约 100，真值落在宽后验内，不能据此证明高精度模型通过。

特别区分：R_stoch 弱约束，均匀分布在两端各 2% 的区域自然有约 4% mass，所以其 boundary flag 本身不证明 fNL 先验敏感；低 k 的直接扩宽结果与 S_stoch 上限才是实际证据。没有按结果调 p、替换协方差、删 bins 或改变既有先验。

继续核验新增 `analyze_rsd_b000_results.py`，复用同一 B-only objective，固定每个 mock 的真实 fNL 后重新 profile 其余参数，不改链/likelihood。`FINAL_INTERPRETATION.json` 记录 25 个 profile、MAP 优化检查、参数边界与三种真值共同残差。未发现 fixed-truth profile 优于原 free MAP 的优化问题；`.12/.14` truth-profile whitened 残差 cosine 约 `.978-.986`。三种真值共享 IC，不能合并为三个独立显著性；相似残差提示共同模型/样本问题，但没有唯一定位为 IR、FoG、loops 或 PNG stochastic 问题。

最终产物：`RESULTS.json/RESULTS.csv`、`RSD_B000_VALIDATION.pdf`、`FINAL_INTERPRETATION.json`。PDF 仅读取并核对所有 8 页文本与矢量边界，不生成 PNG；未做栅格视觉 QA。本次已完成基线计划与结果分析，不自动扩展到新物理模型或 RSD reconstruction。
