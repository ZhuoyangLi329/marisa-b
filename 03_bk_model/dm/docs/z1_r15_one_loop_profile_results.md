# Quijote DM z=1、R=15 one-loop B000 fNL profile recovery

## 结论摘要

本测试使用 446 个 `fiducial/LC_m/LC_p` 公共 realization、15 个 diagonal
`B000(k,k)` bin，以及冻结的 MARISA-B pre/post one-loop 理论：

```text
B(fNL) = B_Gaussian^1loop + fNL * (dB/dfNL)^1loop_at_fNL=0
```

在参考选择 `kmax=0.15 h/Mpc`、单盒体积 `V=1 (Gpc/h)^3` 下：

| stage | input fNL | profile fNL | Delta-chi2=1 interval | truth Delta-chi2 |
|---|---:|---:|---:|---:|
| pre | -100 | -152.480 | [-223.018, -81.941] | 0.554 |
| pre | 0 | -58.161 | [-128.700, 12.377] | 0.680 |
| pre | +100 | 39.871 | [-30.667, 110.410] | 0.727 |
| post | -100 | -95.154 | [-159.818, -30.489] | 0.006 |
| post | 0 | 0.776 | [-63.888, 65.441] | 0.00014 |
| post | +100 | 99.675 | [35.010, 164.339] | 0.00003 |

pre-recon 的恢复斜率为 `0.9618`，但截距为 `-56.92`，说明 Gaussian
one-loop 基线的小幅残差会沿 PNG 模板方向投影为共同的 fNL 偏移。post-recon
的恢复斜率为 `0.9741`、截距为 `+1.77`，三个输入中心均接近真值。

## kmax 稳定性

| kmax [h/Mpc] | diagonal bins | pre: truth 全部在 68% 内 | post: truth 全部在 68% 内 |
|---:|---:|:---:|:---:|
| 0.05 | 2 | yes | yes |
| 0.08 | 4 | yes | yes |
| 0.10 | 5 | yes | yes |
| 0.12 | 6 | yes | yes |
| 0.15 | 7 | yes | yes |
| 0.18 | 9 | no | no |
| 0.20 | 10 | no | no |

两种 stage 都在 `kmax=0.18` 开始出现 `truth Delta-chi2 > 1`，到
`kmax=0.20` 时约为 `4.3--5.8`。因此这套 diagonal one-loop profile 的
当前经验安全上限是 `kmax=0.15 h/Mpc`，不能把更高 k 的精度提升解释为
可靠约束。

## 统计约定与限制

- target 是每个 cosmology 的 446-realization mean；covariance 是对应 stage
  的 fiducial 446-realization 单盒 sample covariance，不是 target mean 的
  covariance。
- likelihood 体积固定为 `1 (Gpc/h)^3`，每个 kmax 使用实际维数对应的
  Hartlap 因子。
- 当前测试仅使用 diagonal `B000(k,k)`，不是完整 120-bin upper triangle。
- 理论在 `fNL=0` 处线性化，不包含有限 fNL 的 even 或更高阶项。
- fiducial target 与 fiducial covariance 来自同一组公共 realization；本结果
  不是 disjoint-covariance validation。
- “truth 在单盒 68% 区间内”不代表把体积放大后仍然通过。

## 数值审计

- 主路径使用 Hartlap-corrected pseudo-inverse precision。
- 独立路径直接求解 covariance 线性方程；所有中心和宽度最大差异均小于
  `1e-7`。
- `kmax=0.15` 另做步长 `0.25` 的显式 chi2 网格，网格最小值与解析中心
  的差异不超过半个网格步长。
- JSON 中记录了所有输入文件的 SHA256、逐 cut covariance 条件数、profile
  结果和解释边界。

## 产物

- 复现代码：`codes/run_quijote_dm_z1_r15_1loop_fnl_profiles.py`
- 完整数值：`quijote_dm_z1_r15_1loop_b000_fnl_profile_kmax0p20_v1.npz`
- 审计记录：`quijote_dm_z1_r15_1loop_b000_fnl_profile_kmax0p20_v1.json`
- 会议图：`9.11meeting/dm/quijote_dm_z1_r15_1loop_b000_fnl_profile_kmax0p20_v1.pdf`
