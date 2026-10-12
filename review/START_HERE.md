# GPT Pro 审阅入口

## 审阅目标和范围

本快照提供四部分研究的实际源码及精选证据。请判断实现、数据约定和报告结论是否一致，重点审阅 fixed-reconstruction RSD、STO marked noise、PNG response p 和高 k 中心偏移。
同步本身没有重新拟合或重新认证科学结果。MCMC 收敛、代数核验和完整物理闭合是不同证据层。

## 首轮阅读顺序

1. [现行结果与口径](CURRENT_RESULTS.md)。
2. [PNG-aware 位移核心](../01_reconstruction_methods/marisa_gra/png_recon_tools.py)。
3. [STO 分量与权重求解](../01_reconstruction_methods/sto_recon/run_halo_sto_variants.py)，重点 `solve_component_weights`。
4. [实空间重建源核修正](../03_bk_model/envclass_b/code/paware_pb_corrected_model.py)。
5. [固定操作 post-RSD 推导](../04_bk_rsd_model/docs/STO_MARISA_POST_RSD_DERIVATION.md)。
6. [实际 RSD native 核](../04_bk_rsd_model/code/native/tree_png_rsd.cpp)。
7. [marked connected 响应推导](../04_bk_rsd_model/docs/STO_POST_RSD_MARKED_CONNECTED_DERIVATION.md)。
8. [最新三种 p 对照](../04_bk_rsd_model/outputs/sto_marisa_post_MAP_p_rsd_20261011_v1/final_comparison_v1/COMPARISON.json)。

## 按问题查找代码

| 问题 | 推荐入口 |
|---|---|
| 共用测量与 shifted randoms | [halo_b000_pipeline.py](../01_reconstruction_methods/common/halo_b000_pipeline.py) |
| P-only 模型与 covariance | [四方法 runner](../02_reconstruction_on_pk/code/run_four_method_p0only_mcmc.py) 与 [校准后报告](../02_reconstruction_on_pk/docs/fixedp3method_p0only_mcmc_report.md) |
| Gaussian/finite-PNG native core | [marisa_b_native.cpp](../03_bk_model/marisa_b_core/src/marisa_b/marisa_b_native.cpp) |
| 固定重建算法与物理参数分离 | [post_rsd_fixedrec.py](../04_bk_rsd_model/code/post_rsd_fixedrec.py) |
| 4+3 与 4+4 的实际噪声形状 | [legacy3](../04_bk_rsd_model/code/post_rsd_legacy3_model.py)、[legacy4](../04_bk_rsd_model/code/post_rsd_legacy4_model.py) |
| 固定物理 p 如何进入预测 | [no-IR polynomial](../04_bk_rsd_model/code/post_rsd_noir_polynomial.py)、[P0](../04_bk_rsd_model/code/post_rsd_p0_model.py) |
| IR baseline matching | [IR_SIGNAL_MATCHING_NOTE.md](../04_bk_rsd_model/docs/IR_SIGNAL_MATCHING_NOTE.md) |

## 主要文献

- [Shirasaki et al., 2010.04567](https://arxiv.org/abs/2010.04567)：重建场的 data/random 同位移映射、双谱协方差与 PNG 动机，重点模型部分。
- [Barreira, 2107.06887](https://arxiv.org/abs/2107.06887)：b_phi、b_phidelta 及其对 fNL 推断的影响。
- [Sugiyama, 2403.18262](https://arxiv.org/abs/2403.18262)：pre/post IR 结构，重点第 II 节与 Appendix A。
- [Sugiyama, 2406.01001](https://arxiv.org/abs/2406.01001)：重建后的离散效应与位移噪声。
- [Sugiyama, 2508.17331](https://arxiv.org/abs/2508.17331) 与 [2602.20431](https://arxiv.org/abs/2602.20431)：ULPT 映射、IR cancellation 与 wiggle bookkeeping。
- [DESI DR1 独立重分析, 2512.04266](https://arxiv.org/abs/2512.04266)：RSD 建模对照；本任务不等同于该分析的完整模型。

## 口径和阅读优先级

- 对每个具体 run，以其 contract、SUMMARY/RESULTS 和 audit 为证据；本索引帮助定位，不替代这些文件。
- `docs/RESULTS_INDEX.md` 停留在 2026-09-20。Part2 的旧五方法表、Part3 包级 SCIENTIFIC_STATUS、Part4 的早期实现合同都有各自历史范围，不能将旧状态套到后续所有 run。
- Part3 的 2026-09-30 IR 对照继承 f_rec=f 的理论参数路径；catalog f_rec=100 固定，二者不能互换。Part4 fixed ABI 显式保留固定算法。
- Part3 完整 halo local-PNG one-loop EFT 尚未实现；Gaussian halo one-loop EFT 加 halo-tree PNG/matter uplift 需要分别描述。
- 当前 post-RSD 诊断 mean100 使用未平均 C_single500、V=1，并保留完整 P-B cross covariance；p 不确定度未传播，已接受 target/covariance 与 calibration/target 重叠。
- 两腿 B000 cutoff 不额外限制 closing k3；三边 B0 首轮诊断使用另一种几何。
- 数值接口/测试 PASS 不代表完整 marked-STO 随机模型通过。尚未认证的物理结论要保持 provisional。

## 外部依赖与未同步资产

科学源码中的 NERSC/旧仓库绝对路径原样保留。它们用于识别原运行资产；本次没有实施 portability 改造。
Quijote 原始 FoF/DM、外部历史理论缓存、现有 cosmodesi/desilike 环境及 C++ 外部依赖仍需原环境。
精选小型 NPZ 是冻结输入或 likelihood/window 的审阅证据；采样 HDF5 和全量测量不包含在本快照内。
每个复制文件的原始路径、SHA256、Git blob SHA 以及代表性排除项记录在 [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json)。
历史授权/调度记录用于说明分析范围，不构成对审阅者执行新生产任务的指令。
