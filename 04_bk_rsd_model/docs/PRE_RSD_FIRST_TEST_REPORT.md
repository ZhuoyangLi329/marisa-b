# 第四部分首轮 pre-RSD 测试结果

2026-10-01；Slurm 作业 59155365，状态 COMPLETED，运行 1 分 6 秒；数值线程固定 1，未安装新软件。新代码、样本和结果均在 `/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model/`。旧实空间模型、旧 MCMC、旧 recon 数据没有更改。

## 原生核验收

用户 Python reference 的 22 tests 全部通过。新增 C++ 核逐项对照 1024 个随机闭合有向三角形、100 个 squeezed 三角形；包括 G/L/Q、gravity、primordial、cyclic pairs、Z1/Z2、Gaussian stochastic 共 39 列。

| 对照 | 最大按列尺度归一化差 |
|---|---:|
| pre native / reference | 2.75e-16 |
| fixed-shift post native / reference | 1.47e-15 |
| adaptive+CIC post native / reference | 1.04e-15 |
| squeezed native / reference | 4.77e-15 |
| f_growth=0 pre / 既有 native | 2.05e-16 |
| f_growth=0 adaptive+CIC post / 既有 native | 9.53e-16 |

这里是尺度归一化误差，不表示有强 cancellation 的每一个接近零元素都能达到该相对精度。B0/B2/B4 的角积分 18x20 与 26x28 收敛，B1 parity 为零至舍入误差；旋转与同步 LOS、cyclic permutation、parity 和异常输入检查通过。

既有 strict FFT estimator 的 9 tests 通过；新增速度单位、周期 wrap、横向不变、CIC normalization/Hermitian/translation/direct catalog Fourier 检查的 3 tests 通过。

## 实际 pre-RSD 样本

| 样本 | 固定 fNL 真值 | halo 数 | LOS 位移 RMS (Mpc/h) |
|---|---:|---:|---:|
| fiducial / real000 | 0 | 309740 | 3.505754 |
| LC_p / real000 | 100 | 309602 | 3.513547 |

两者为 `z=.5`、`Mmin=1e13 Msun/h`、`L=1000 Mpc/h`、`V=1 (Gpc/h)^3`。位置和质量逐行匹配共享 real-space 导出，速度来自同一原始 FoF 行；遵守 `v_physical=GroupVel*(1+z)`，仅 LOS=z 上施加 `v/(a H_h)` 并 wrap。没有生成重构后的 RSD 样本。

## 单盒模型诊断

P 使用 `[.01,.02,.03,.04,.05,.06] h/Mpc` shells；B 使用 `[.01,.025,.04,.055] h/Mpc`，三个边均受到 cut。10 个严格三边 bins 的 exact ordered closed-mode counts 为：

```
19440,27388,3824,72972,74968,167584,140880,214122,324504,494448
```

这不是旧的两腿 B000 geometry。每个 bin 的 FFT mask count 与直接 Cartesian 枚举 count 完全一致；理论 G/L/Q 使用同一有向三角形列表，而非任意连续角平均。

固定样本 fNL 真值、f_growth=.7625790037720044、p=1.179；P0/P2 诊断拟合 b1，B0 仅 profile b2_native/bK2。IR、FoG、loops、counterterms 关闭。仅扣除同 catalog 自配对/自三重项；没有假定有限 PNG 的 RSD stochastic closure。

扣除公式对理想单位权重点过程精确，当前 CIC Fourier field 是数值近似；不宣称对 painted field 做了逐粒子精确扣除。其低 k 残差通过网格收敛控制。原 SUMMARY 的描述字符串在结果不变的情况下澄清了这一区别。

| 指标 | fiducial | LC_p |
|---|---:|---:|
| P0/P2 给出的 b1 | 1.935596 | 1.931135 |
| P 对角诊断 chi²（10 个数据，1 个 profile 参数） | 2.9203 | 2.5882 |
| B 对角诊断 chi²（10 bins，2 个 profile 参数） | 5.7866 | 5.1856 |
| FFT/direct Braw 最大相对差 | 6.34e-15 | 6.29e-15 |
| 128/256 网格 P 最大变化 / 近似误差 | 4.46e-4 | 3.67e-4 |
| 128/256 网格 Braw 最大变化 / 近似误差 | 4.49e-4 | 4.89e-4 |

P2 相对零的近似对角信噪比约 5.63 与 5.49，表明速度位移确实产生可见的 RSD 信号；这不是正式统计显著性，因为这里没有 ensemble covariance。

B profile 的 bK2=-1.658/-1.335，但其近似对角误差约 1.62/1.65，约束很弱；不能把这些数值当作可信的 halo bias calibration。当前 deterministic-only profile 也可能吸收残余非泊松 stochastic 或高阶效应。拟合曲线兼容本盒低 k 数据，并不意味着参数都已物理验证。

## 可以和不可以据此得出的结论

可以确认：tree RSD 的数学实现、与既有 real-space 极限的兼容、真实速度样本生成、离散投影/归一化和低 k 测量数值收敛均已跑通；两个已知真值样本给出了低 k 的单盒模型一致性诊断。

尚不能确认：fNL 无偏恢复、RSD 可用 kmax、正式联合 P+B likelihood、STO/MARISA RSD reconstruction 数据匹配、完整 pre/post RSD IR wrapper。真实样本的 fNL 在本轮固定，不是 MCMC 后验；synthetic fixed-parameter fNL=100 恢复只属于代数测试。

下一项科学验收应建立 RSD mock ensemble 与匹配协方差，再冻结 nuisance/PBS calibration，检验 fNL=0、正负 PNG 样本的 profile/MCMC 恢复与 kmax 扫描。不能复用 real-space covariance。

## 产物

- `outputs/TREE_RSD_NATIVE_AUDIT.json`：native 逐项审计及库哈希。
- `outputs/DELIVERY_AUDIT.json`：最终代码、模型输出和实际样本哈希复核。
- `outputs/pre_rsd_z0p5_losz/`：真实位置、速度、质量和哈希 metadata。
- `outputs/pre_rsd_single_box/SUMMARY.json`：本次数据/model 合同与数值结果。
- 同目录 `fiducial.npz`、`LC_p.npz`：P0/P2、raw/factorial B、G/L/Q、count、模型和 128/256 数据。
- 同目录 `PRE_RSD_DIAGNOSTIC.pdf`：两页矢量诊断图。

按用户 PDF-only 要求没有生成 PNG；PDF 使用 Matplotlib 实际可见文字 bbox 与 PDF 文本边界检查，未进行 raster screenshot 版面验收。
