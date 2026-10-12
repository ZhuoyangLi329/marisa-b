# 第三部分：Bk model / MARISA-B

本部分整理 MARISA-B 的物理模型、native C++ 核心、Python 接口、Quijote 生产入口、测试、结果报告和 PDF 图。代码按模型对象和计算阶次分层，原仓库不修改。

## 目录

- marisa_b_core/: 独立 MARISA-B 源树，保留 src、python、production scripts、configs、schemas、provenance、tests 和 docs；不复制 build 二进制、Git 元数据、外部模拟和 Python 缓存。
- dm/: DM 的 tree-level、SPT one-loop、有限 fNL G/L/Q 模型、积分收敛、profile/MCMC 入口和结果。
- halo/: halo tree-level finite PNG、Gaussian halo one-loop EFT-v2、post-reconstruction EFT-v2 入口和验证结果。
- figures/: 第三部分直接相关的 PDF 图。

## DM 模型链

DM tree-level 使用标准 F2/F3/F4 kernel 和 primordial local PNG 项，投影成 real-space B000。SPT one-loop 使用 B222、B321I、B321II、B411 Gaussian diagrams，以及 B122I、B122II、B113I、B113II PNG linear loop 项和 B112II finite-fNL quadratic 项。最终 finite model 写成 B=G+fNL*L+fNL^2*Q，并在每个 kmax 重新计算 covariance/Hartlap。

主要入口是 marisa_b_core/scripts/production/quijote_dm_spt1loop_realspace.py、dm/prepare_quijote_dm_z1_r15_matched_fnl_model.py 和 dm/run_quijote_dm_z1_r15_1loop_fnl_profiles.py，角向/q 收敛检查位于 dm/profile_quijote_dm_spt1loop_* 和 dm/scan_quijote_dm_spt1loop_convergence.py。

当前最终 DM 结果显示 z=0.5、R=10 的 diagonal B000 在 kmax=0.08/0.10 可以 recovery；kmax=0.15/0.20 虽然误差变小，但出现共同 Gaussian baseline 偏移。z=1、R=15 的 diagonal one-loop profile 在 kmax=0.15 最稳妥，kmax=0.18 起出现明显失效。主要限制是 Gaussian one-loop baseline 与测量的高 k mismatch，不是遗漏已知 PNG loop 或 B112II。

## halo 模型链

halo tree-level 使用 finite local PNG halo tree，包含 b1、b2、tidal bias、reconstruction PNG bias 以及 fNL^2 项。Gaussian halo sector 由 EFT-v2 提供 B222、B321I、B321II、B411、counterterms 和 stochastic basis；production Gaussian tier 是 coevolution，full tier 是强制 cross-check。

当前 assembled halo finite-PNG prediction 是 Gaussian halo one-loop EFT baseline + finite halo-tree PNG G/L/Q + implemented matter PNG one-loop uplift。它必须准确描述为“halo Gaussian one-loop EFT 加完整 halo-tree local-PNG sector”，不能称为完整 halo local-PNG one-loop EFT。post-reconstruction 的 EFT-v2、PNG tree、matter uplift 和 shift-field response 已集成，但数值收敛和 PNG halo 输入验证仍是 provisional。

主要入口是 marisa_b_core/scripts/production/run_pre_recon_halo_tree_fnl2_v1.py 和 run_post_recon_halo_finite_png_v1.py，halo/ 中的脚本负责 finite-tree bases、direct finite-tree profile 和 kmax validity。

## 阅读和构建

先读 marisa_b_core/README.md、marisa_b_core/docs/halo_v1_derivation.md、marisa_b_core/docs/SCIENTIFIC_STATUS.md、marisa_b_core/docs/INFERENCE_CONTRACT.md，再读 DM 和 halo 的 production scripts。C++ 源码的 canonical core 在 marisa_b_core/src/marisa_b/marisa_b_native.cpp；halo EFT diagrams 在 marisa_b_core/src/eft_v2；DM shell projector 和 one-loop driver 在 marisa_b_core/src/marisa_b 与 scripts/production。

MARISA-B 是独立代码环境，使用 GNU Make 和 C++17/OpenMP 构建；本次迁移不复制已编译 build 目录，避免把机器相关二进制当作源码。大数据、缓存、MCMC chain 和完整生产矩阵仍应放在外部 data root。

## 结果与边界

DM 和 halo 的最终 PDF、审计 JSON、模型报告分别位于 dm/ 和 halo/。所有科学图只保留 PDF。文件 provenance 和排除项见 MIGRATION_MANIFEST.json。
