# 现行结果与分析口径

整理时间 UTC：2026-10-12T12:03:47.075360+00:00。下列数值直接从原结果 JSON 提取；不是新实验。

## Part2：p 校准后的 P0-only

数据为 matched100；covariance 是每方法 Fid500 单盒 covariance /42.875，加对应维数 Hartlap。
这里的误差是中央 68% 区间半宽。p 来自相同 +100 target 的固定 fNL 校准，+100 恢复属于同数据自洽检查。

| target | method | fNL median | 68% half-width |
|---|---|---:|---:|
| fNL0 | pre | -1.6313 | 11.8885 |
| fNL0 | sto_only | -0.6356 | 11.5826 |
| fNL0 | pre_sto_only | -1.1746 | 11.8208 |
| fNL100 | pre | 100.5676 | 12.1398 |
| fNL100 | sto_only | 100.0842 | 11.6793 |
| fNL100 | pre_sto_only | 99.6101 | 11.7594 |

来源：[fixedp3method_p0only_mcmc_results.json](../02_reconstruction_on_pk/results/fixedp3method_p0only_mcmc_results.json)。
校准后的 STO-only/pre+STO-only 宽度仅小幅改变，旧固定 p 的明显收紧和低偏应作为响应失配诊断阅读。
环境 multitracer 的 G/R 模板和 target 共用模拟；中心恢复是构造性自洽，协方差与模板噪声尚限制精度比较。

## Part3：源核修正和 IR 分离对照

以下是 +100 target、V=42.875 的 2026-09-30 实空间对照。保留 inherited f_rec=f 路径；不能称为固定非零 f_rec 的完整推断。

| case | fNL median | central 68% | MAP residual metric GoF | truth in 68% |
|---|---:|---|---:|---|
| sto_marisa_noir_k0p12 | 102.0026 | [94.4082, 109.8545] | 0.743261 | True |
| sto_marisa_noir_k0p14 | 112.9519 | [105.7052, 120.2607] | 0.076049 | False |
| pre_ir_k0p12 | 116.9297 | [105.6844, 128.3232] | 0.998577 | False |
| pre_ir_k0p14 | 117.7429 | [106.8459, 128.6902] | 0.999784 | False |
| sto_marisa_ir_k0p12 | 102.1490 | [94.4362, 109.8883] | 0.718669 | True |
| sto_marisa_ir_k0p14 | 112.9894 | [105.8181, 120.4785] | 0.059366 | False |

来源：[SUMMARY.json](../03_bk_model/envclass_b/outputs/paware_pb_ir_brecfix_k012_k014_v2/SUMMARY.json)。
源核修正与 IR 效应分别比较；0.14 中心仍偏离输入，IR 不显著消除该偏移。
DM one-loop 的高 k 共同 offset 主要指向 Gaussian baseline mismatch，详见 [DM 报告](../03_bk_model/dm/docs/z0p5_r10_spt1loop_report.md)。

## Part3：2026-10-11 单盒精度重拟合

mean100、C_single500、V=1、no-IR、P0/.08 或 4+3 P+B/.08/.14。
这里的 sigma 是后验标准差；保留原实空间先验和模型，与 RSD 的 flat-prior 配置并非完全同条件。
STO+Marisa fNL=-100 existing matched realspace measurement unavailable；不得补写该缺失结果。

| case | truth | fNL median | posterior std |
|---|---:|---:|---:|
| pre_fiducial_P0 | +0 | -1.0106 | 83.4037 |
| pre_fiducial_PB_4plus3_k0.08 | +0 | -2.7897 | 85.0928 |
| pre_fiducial_PB_4plus3_k0.14 | +0 | -1.4500 | 73.0078 |
| pre_LC_p_P0 | +100 | 101.2770 | 77.3011 |
| pre_LC_p_PB_4plus3_k0.08 | +100 | 102.3864 | 78.1797 |
| pre_LC_p_PB_4plus3_k0.14 | +100 | 106.9544 | 70.4859 |
| pre_LC_m_P0 | -100 | -106.2074 | 84.7914 |
| pre_LC_m_PB_4plus3_k0.08 | -100 | -108.3756 | 84.1366 |
| pre_LC_m_PB_4plus3_k0.14 | -100 | -104.1939 | 75.0039 |
| sto_marisa_fiducial_P0 | +0 | 7.1551 | 75.9726 |
| sto_marisa_fiducial_PB_4plus3_k0.08 | +0 | 5.2801 | 71.1652 |
| sto_marisa_fiducial_PB_4plus3_k0.14 | +0 | 31.9529 | 52.8541 |
| sto_marisa_LC_p_P0 | +100 | 100.8125 | 71.0285 |
| sto_marisa_LC_p_PB_4plus3_k0.08 | +100 | 94.2179 | 67.2002 |
| sto_marisa_LC_p_PB_4plus3_k0.14 | +100 | 115.2174 | 49.8788 |

来源：[RESULTS.json](../03_bk_model/envclass_b/outputs/realspace_V1_existing_measurements_20261011_v1/RESULTS.json)。

## Part4：最新 post-RSD 三种固定 p 对照

固定算法仍为 b_rec=2.2280546353635335、p_rec=1.2822、fNL_rec=100、R10；物理 p 与算法 p_rec 分开。
post-P0 校准后验中位数为 1.3079259941635721，联合 MAP 为 1.2494076995227128；实空间回代值为 1.2822。
三种值是不同的条件对照；主拟合没有传播校准 p 的不确定度。
mean100、C_single500、V=1、P0/.08、P+B 4+4、no-IR、完整 P-B cross。已接受重叠按原文件记录。

| statistic | truth | MAP-p fNL median | MAP-p posterior std | median-p fNL median | realspace-p fNL median |
|---|---:|---:|---:|---:|---:|
| P0 only | +0 | 11.4565 | 82.3177 | 11.3339 | 14.3428 |
| P0 only | +100 | 103.0670 | 82.5458 | 110.1983 | 106.8940 |
| P0 only | -100 | -80.6887 | 83.9903 | -86.9407 | -84.2229 |
| P+B kB=0.08 | +0 | 6.9615 | 81.1951 | 9.4965 | 8.8816 |
| P+B kB=0.08 | +100 | 97.1429 | 80.4093 | 100.6509 | 101.3878 |
| P+B kB=0.08 | -100 | -76.8986 | 82.4347 | -83.0428 | -81.3704 |
| P+B kB=0.14 | +0 | 34.4529 | 70.0334 | 35.6397 | 35.3304 |
| P+B kB=0.14 | +100 | 110.2440 | 69.4362 | 115.2223 | 113.7768 |
| P+B kB=0.14 | -100 | -39.8714 | 71.1247 | -37.3908 | -38.9786 |

来源：[COMPARISON.json](../04_bk_rsd_model/outputs/sto_marisa_post_MAP_p_rsd_20261011_v1/final_comparison_v1/COMPARISON.json)。
[BEST_FIT_RESULTS.json](../04_bk_rsd_model/outputs/sto_marisa_post_MAP_p_rsd_20261011_v1/final_comparison_v1/BEST_FIT_RESULTS.json) 单列联合 MAP，不能与边缘后验中位数混用。
采样门已通过，但完整 STO noise closure 和无偏 kmax 尚未认证；mean100 配 C_single 的 chi2 不能解释为均值数据的校准 GoF。
0.14 的残余中心偏移在不同 p 对照下仍存在；有限样本的覆盖不能单独证明模型无偏。

## 推荐图与证据

- [P0 实测对比](../02_reconstruction_on_pk/figures/pre_vs_sto_only_measured_p0.pdf)：展示幅度与 ratio；图中的 mean error 与推断使用的精度口径分别记录。
- [B/P+B 历史全景](../03_bk_model/envclass_b/figures/fnl_vs_bkmax_panorama_3methods.pdf)：2026-09-20 历史结果，后续源核修正另见上述 SUMMARY。
- [post-P0 重校准对照](../04_bk_rsd_model/outputs/sto_marisa_postP_calibration_20261011_v1/final_comparison_v1/POST_P0_RECALIBRATED_RSD_COMPARISON.pdf)。
- [实空间 p 回代 RSD](../04_bk_rsd_model/outputs/sto_marisa_realspace_p_rsd_20261011_v1/final_comparison_v1/REALSPACE_P_RSD_COMPARISON.pdf)。
