# STO+Marisa post-RSD：正式噪声闭合与经授权的诊断 P+B

2026-10-10 19:50更新：用户直接批准“先跑诊断 MCMC，同时核验”，授权记录为 `outputs/sto_marisa_post_rsd_mean100_v1/PROVISIONAL_PB_APPROVAL_20261010.json`。因此完整物理闭合之前允许明确标记provisional的诊断P+B，但不得生成正式noise PASS、正式fNL或kmax结论。该新授权优先于本文件早先的“完整噪声先于所有初步链”顺序；正式推断的物理要求没有降低。

用户随后要求使用最新完成realization数量重跑。新 `diagnostic_latest_cov_pb_20261010_v1/` 冻结402个fid（0–193、292–499；19:49:34），先更新同一C_single快照的p/P0，再自动进行三种真值P+B。保持mean100 target、V1、ddof1和完整P–B cross；不动态换C、不将C缩放为mean covariance。原205链只读保留，700测量继续。

当前目标仍为实际初步P+B MCMC，同时完成随机模型验证；不是只完成P0或数值单元测试。下方执行顺序描述正式闭合的必需证据，不能把某一层数值PASS误记为完整物理闭合。

## 已有证据与不能跨越的边界

- 三个 leading shift transport 算子的原44-bin双网格、独立整数求和/卷积与物理 b/p 系数回放已经完成。
- colored Gaussian-pair 候选的独立 Wick/F2-zero-mean 测试通过；它只覆盖已指定的 Gaussian pair closure，不包含真实 marked 场的全部 connected 三点/四点。
- 与 p/P0 并行的205盒实际权重、部分 C_single、零腿及幅度假设诊断完成，但没有测定真实 N(k,mu)，不是 `POST_WEIGHTED_NOIR_NOISE_AUDIT.json`。
- 旧 STO `spectra.npz` 是实空间 halo–matter 操作性残差。它既不是 post-RSD 的纯 epsilon 校准，也不是已验证的 isotropic white-noise 输入。
- 同盒原始 DM HDF5 有 Coordinates 与 Velocities；可读取现有数据作真实 RSD 参照，无需新的 N-body realization 或下载。Gadget 粒子存储速度须乘 sqrt(a) 才是 peculiar km/s，不能套用 FoF GroupVel*(1+z)。

## 执行顺序

1. **实际输入 marked 场与数值窗口**：新 `measure_sto_rsd_input_noise.py` 先读取三真值 real000 的全量 DM（不能截断粒子），保存 DM real/RSD、halo w real/RSD、halo w²/w³ RSD 的原始 Fourier modes 与全部 mixed spectra。先跑128/256双网格、TSC/interlacing3，做流式paint、质量守恒、归一化、直接低mode与速度转换的独立检查。该阶段是新增诊断，不替换生产 P/B estimator，不生成正式模型 PASS。
2. **从操作性残差到模型随机场**：区分 matter 非线性、velocity bias、有限mode投影偏差、DM离散效应与真正的 stochastic N(k,mu)。不得直接把 `P_hh-|P_hm|²/P_mm` 命名为已校准纯噪声，不减假定 Poisson matter shot，不把 N2 当 Nraw。
3. **marked self 与 connected 响应闭合**：保留权重平方/立方场的实际 mixed correlators；检验源的 F2 fluctuation 与 intrinsic 三点、shift 四点在当前 tree 计数中的贡献。已有 F2 零模均值检查不能代替此项，三个 transport 算子也不是三个可任意独立浮动的 nuisance。
4. **真实有限位移、PNG 与 estimator 匹配**：固定 b_rec=2.2280546353635335、p_rec=1.2822、fNL_rec=100、rec256/R10/data-only shift；对0/+100/-100做匹配检查，保留完整两腿壳层分母和 unrestricted closing leg。实际源的物理 p 使用校准，不以 p_rec 或数值探针回退。零腿两种定义需显式一致选择，不能用噪声参数暗中吸收区别。
5. **协方差权重与放行**：在实际 post 同盒 C_single 与 P–B cross block 下评估遗漏项及先验域误差，V=1、ddof1，协方差不除以100或500。部分协方差可用于用户明确授权的provisional链，但不能替换正式 C500 流程。正式放行必须先形成逐项实际证据的 no-IR 噪声报告；上述用户授权只允许未认证的诊断链，不允许把它写成正式闭合。不通过则保留失败，补模型而非更换真值/阈值。

IR 噪声 dressing 是上述 no-IR 闭合之后的独立验证层；no-IR PASS 不等于 IR 完成。当前“完整”指当前建模阶数和分析窗口内的全部必需随机项与真实样本验证，不是声称任意高阶、任意 k 都完全正确。

## 当前新增入口的边界

16:39更新：三真值双网格六套全DM/mixed谱实测于16:23完成，原PID269059实际退出；noise_rsd_input_spectra_v1/GRID_AND_SPECTRA_AUDIT.json按全部134896个matching modes独立回放谱，最大auto-normalized差1.0714e-14<原3e-13门，128/256最大谱差4.2583e-5。每真值仍只有real000，没有将投影残差认作pure noise。新marked cumulant/particle接口9项测试在NERSC通过，真实300目标的finite-random条件矩和三/四点连通响应探针已在CPU13启动。恒等式与尚未完成的signal/noise、tree阶数分离见STO_POST_RSD_MARKED_CONNECTED_DERIVATION.md。

`sto_rsd_noise_fields.py`、`test_sto_rsd_noise_fields.py`、`measure_sto_rsd_input_noise.py` 仅推进第1项。四项独立测试在 NERSC login26 单核通过（原生 pmesh 实际运行），没有安装包、改变原生产源、创建噪声放行报告或启动 P+B。

新增输出：`outputs/sto_marisa_post_rsd_mean100_v1/noise_rsd_input_spectra_v1/`。状态与实际 PID 一起核对；观察超时不代表终止。该样本选择与原 target 重叠，明确只用于首轮数值/物理诊断，不声称 independent mock recovery。
