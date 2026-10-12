# 第二部分：recon 方法在 P(k) 上的应用

本部分只保留使用重构后的 halo P0(k) 做 local PNG fNL 限制所需的代码、结果摘要和 PDF 图。原仓库不修改，第一部分的单 realization B000/P0 测量入口仍位于 `../01_reconstruction_methods/common/halo_b000_pipeline.py`。

## 样本

主样本是 Quijote z=0.5 的 FoF halo，质量下限 `Mmin=1e13 Msun/h`，real-space positions，盒子边长 `1000 Mpc/h`。Fiducial 有 500 个 realization；local PNG 的 `LC_p`（`fNL=+100`）有 100 个 realization。P0-only 结果用 Fiducial real000--499 构造每种方法自己的 covariance，用 LC_p real000--099 作为 +100 target；fNL=0 target 是 Fiducial matched real000--099 mean。

## P0(k) 提取

每个 realization 使用 `jaxpower` 经 `clustering_statistics.spectrum2_tools.compute_box_mesh2_spectrum` 测量周期盒 monopole：`ells=(0,)`、LOS 为 z、meshsize=256、`k` 从 fundamental mode `2*pi/L` 开始到 `0.08 h/Mpc`，共 12 个 bin。runner 的 `--measure-pk` 先测 pre，再对 post 测 reconstructed data 加 shifted uniform randoms 的 overdensity；`pole.value()` 是 shot-noise corrected P0，同时保存 `k`、`edges`、`nmodes`、`norm` 和 `num_shotnoise` 到 `pre_pk.npz`/`post_pk.npz`。聚合器检查所有 realization 的几何一致后堆叠矩阵。

P/B 联合分析保留 realization-by-realization 的完整 P--B cross covariance；历史最大相关约为 pre=0.547、STO+Marisa=0.447，不能使用 block diagonal。

## P(k) 模型

P0-only MCMC 使用 finite local-PNG tree-level lattice model，参数为 `(fNL, b1, Pshot_residual)`，其中 `p_halo=1.12`、`delta_c=1.686`、`b_phi=2*delta_c*(b1-p_halo)`：

```text
P_hh(k) = b1^2 P_L(k)
        + 2 fNL b1 b_phi P_L(k)/M(k)
        + fNL^2 b_phi^2 P_phi(k)
        + Pshot_residual
```

P-only 不使用 P(k) bias prior，`b1` 是自由参数。Fiducial 平均 P(k) 的独立 real-space fit 为 `P_hh=b1^2 P_lin+c0`，得到 `b1=1.9326527 +/- 0.0018319`。

## Recon 方法与使用

P0-only 主结果自 2026-09-16 起为五方法：pre、standard post、Marisa post、STO-only、STO+Marisa。四方法 v1 按冻结预注册 `inputs/p0only_preregistration.md` 完成并保持不动；STO-only 行按修正案 `inputs/p0only_sto_only_amendment.md` 新增（同一套 matter-assisted N+Env STO 权重生产、位置不动、无 gra-recon），四方法行原样并入五方法表。五方法 Fisher 基线另包括 STO+standard。STO+Marisa 的顺序是先 matter-assisted N+Env STO weighting，再 Marisa gra-recon；standard 和 Marisa 都使用第一部分的 data/random 与 shifted-random 约定。

每种方法使用自己的 fNL=0 Fid500 P0 covariance，除以 `V=42.875 (Gpc/h)^3`，使用 12 维 Hartlap 因子 `0.9739479`；target mean 不再除以 100。P+B 扩展使用 P0 的 12 bins、B000 的 44 bins，并保留全 cross covariance。

## 文件

- `code/aggregate_four_method_p0.py`：四方法 P0 matrix 聚合（冻结 v1 输入）。
- `code/aggregate_five_method_p0.py`：在冻结四方法矩阵上扩展 STO-only 的五方法聚合。
- `code/fit_halo_pk_bias.py`：Fiducial P0 bias/shot-noise 拟合。
- `code/run_four_method_p0only_mcmc.py`：四方法 P0-only 三参数 MCMC（冻结 v1）。
- `code/run_five_method_p0only_mcmc.py`：五方法 P0-only MCMC runner；本轮只新跑 sto_only 两 case，四方法行自冻结 v1 继承。
- `code/audit_four_method_p0only_mcmc.py`：四方法独立复核。
- `code/audit_five_method_p0only_mcmc.py`：五方法独立复核（sto_only 自 600 个原始 post_pk.npz 重建）。
- `code/plot_p0only_contours.py`：四方法 P0-only 图。
- `code/plot_pre_vs_sto_only.py`：pre vs STO-only 专用对比图（后验 contour 与测量平均 P0 曲线/ratio）。
- `code/aggregate_pre_post_joint_p0.py`：pre+STO-only 联合 24-bin 矩阵聚合（逐 realization 堆叠、全 cross-covariance）。
- `code/run_pre_post_joint_p0only_mcmc.py`：pre+STO-only 联合五参数 MCMC（共享 fNL，通道独立 b1/sn0）。
- `code/audit_pre_post_joint_p0only_mcmc.py`：联合分析独立复核。
- `code/p0only_arviz_compat.py`：R-hat/ESS 的 NumPy/SciPy 兼容层（源自 marisa_b_core，见文件头）。
- `inputs/`：五方法与四方法统一矩阵、pre+STO-only 联合矩阵、finite local-PNG lattice、冻结预注册、STO-only 与 pre+STO-only joint 两份修正案、Fiducial P0 bias-fit 输入。
- `results/`：审计、Fisher、MCMC 和 bias 摘要（五方法 `five_method_p0only_mcmc_*`；联合 `pre_post_joint_p0only_mcmc_{results,audit}.*`）。
- `docs/`：报告与结论（五方法 `five_method_p0only_mcmc_report.md`；联合 `pre_post_joint_p0only_mcmc_report.md`）。
- `figures/`：只保留直接相关 PDF（五方法 `five_method_p0only_*`；两方法对比 `pre_vs_sto_only_*`，测量均值比全程约 1.29 与 STO 加权 b1^2 增幅一致；联合 `pre_post_joint_p0only_*`）。
- `work/`：五方法与联合分析的 profiles、链、posterior、任务记录与中间报告。
- `work/quijote_halo_z0p5_five_method_p0only_fid0_p100matched100_mcmc_v1/`：五方法 profiles、链、posterior、任务记录与中间报告。

大 runner 的旧结果矩阵、posterior chain 和原始模拟没有复制到 v2；四方法 v1 的 posterior 按只读链路（sha256 校验）引用旧仓库 NPZ。

Completeness review is documented in docs/COMPLETENESS_AUDIT.md. The direct input contracts needed by the P0-only runner are under inputs/: the unified four-/five-method matrices, finite local-PNG lattice, preregistration (含 STO-only 修正案), and Fiducial P0 bias-fit input.
