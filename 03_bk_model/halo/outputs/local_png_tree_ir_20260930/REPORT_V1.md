# B-only IR kmax 收敛检验：V=1 对照

日期：2026-09-30。任务仓库：`/pscratch/sd/l/lzy/reconstuction-png-v2`。

## 结论

将 `ir_kmax_scan_convergence` 的拟合体积从 **42.875 改为 1 (Gpc/h)³**，保持高精度模型模板、数据均值、先验、bias 校准、kmax 档位、拟合边界和通过门槛不变，重新完成 245 组拟合。

**V=1 时，no-IR、strict LO IR 和 factorized IR 在全部扫描档位均通过，直到 0.20 h/Mpc。** 因而现在只能给出“通过上限至少为 0.20”的下界；未扫描更高 k，不能认定真正失效边界恰为 0.20。

| PNG bias 校准 | 模型 | V=42.875 最大连续通过档位 | V=1 已通过至 |
|---|---|---:|---:|
| 当前 p=1.1788659191136683 | no-IR | 0.10 | 0.20 |
| 当前 p=1.1788659191136683 | strict LO IR | 0.10 | 0.20 |
| 当前 p=1.1788659191136683 | factorized IR | 0.10 | 0.20 |
| 旧 p=1.12 | no-IR | 0.08 | 0.20 |
| 旧 p=1.12 | strict LO IR | 0.10 | 0.20 |
| 旧 p=1.12 | factorized IR | 0.10 | 0.20 |

单位均为 h/Mpc。原封存 historical 模型也通过至 0.20。

这次 **没有显示 IR 相对于 no-IR 独立提高可用 kmax**：二者在 V=1 下都尚未达到失效档位。与 V=42.875 相比提高的是在更大统计误差下满足既有验收门槛的尺度，不是模型本身的绝对精度发生了提升。

## V=1 的实际数值

以下为当前 p 校准、同一高精度模板 nrad=6、nmu=120，IR 使用 strict_lo、k_IR=0.15 h/Mpc。

| kmax | no-IR 最大真值 Δχ² | strict IR 最大真值 Δχ² | 真值通过数 no-IR / IR |
|---:|---:|---:|---:|
| 0.08 | 0.001284 | 0.002612 | 5/5，5/5 |
| 0.10 | 0.015881 | 0.012330 | 5/5，5/5 |
| 0.12 | 0.060731 | 0.031352 | 5/5，5/5 |
| 0.14 / 0.15 | 0.026333 | 0.042550 | 5/5，5/5 |
| 0.18 | 0.249837 | 0.149623 | 5/5，5/5 |
| 0.20 | 0.188270 | 0.163574 | 5/5，5/5 |

Gaussian fixed-fNL=0 的数据 χ² gate 在所有档位也均通过；当前校准最小 p 值约为 0.999989。这里使用很多 realization 的均值和 V=1 的单个调查协方差，数据均值噪声远小于拟合误差，因此接近 1 的 p 值不能解释为理论达到了异常高的绝对精度，也不是单个真实 survey 的拟合优度预测。

真值 fNL=0 的若干具体结果（区间为 profiled Δχ²=1，不是 MCMC credible interval）：

| kmax | 模型 | V=42.875：最佳 fNL [区间] | V=1：最佳 fNL [区间] |
|---:|---|---|---|
| 0.12 | no-IR | 18.29 [3.84, 32.90] | 16.13 [-76.37, 114.62] |
| 0.12 | strict IR | 12.15 [-2.12, 26.56] | 10.23 [-81.10, 107.52] |
| 0.20 | no-IR | 29.30 [18.16, 40.45] | 28.73 [-44.13, 101.82] |
| 0.20 | strict IR | 26.84 [15.73, 37.95] | 26.29 [-46.33, 99.10] |

例如 0.20 时 IR 的绝对偏差仍约为 +26，no-IR 仍约为 +29，二者并未变为无偏。区间半宽则从约 11 增至约 73，增长约 6.546 倍，接近 sqrt(42.875)=6.548。nuisance 先验没有随体积缩放，因此改变体积也改变了数据和先验的相对权重，最佳值会小幅移动；不能只把旧误差条乘以 sqrt(42.875) 而不重新拟合。

IR 改善仍不随 kmax 单调变化，例如 0.14/0.15 下 IR 的偏差反而略大；V=1 时这些差异均不足以触及既有统计门槛。

## 保持不变的检验设置

- pre-reconstruction local-PNG B000 only；未加入 P likelihood，未检验 adaptive/CIC post 分支。
- fNL 注入值 -100、-50、0、50、100；非零各 250 个 realization、零点 500 个 realization 的均值。
- 使用 Fid500 realization covariance，C_fit=C_realization/V，并采用原 Hartlap 修正；**不是 C/500，也没有换成某一个 V=1 realization**。
- 固定 b1=1.93；b2、gamma2、Ashot_residual、Bshot_residual 四个 nuisance 保留原高斯先验。
- 保留真值±200 搜索范围；所有 245 组最佳值有效、所有 Δχ²=1 区间未碰边界。
- 同时要求 Gaussian gate p≥0.05、五个注入真值全部满足 profiled Δχ²≤1，以及有效未截断区间；五点通过不是完整的 frequentist coverage 测量。
- kmax 仍指 B000 两条分 bin 测量腿的上限，第三条闭合腿可能更大；0.14 与 0.15 使用同一组 bins。

## 数值核对

- 原 V=42.875 JSON、PDF 和冻结数据的 SHA256 完全不变。
- 新旧所有模型设计矩阵逐元素完全相同，只有拟合体积变化。
- 白化矩阵按 sqrt(V) 缩放，最大绝对核对差 3.56e-14；先验完全未改。
- 不传 --volume 时保持原默认值 42.875；代表性回归的最佳 fNL 和 objective 与冻结结果完全一致。
- 拒绝 0、负数、NaN、无穷体积。
- 7 组代表性拟合用独立五维 Minuit 和 MINOS 交叉检查；全部有效，最佳 fNL 最大差 3.06e-4，objective 最大差 4.83e-11，区间端点最大差 0.0155。
- 原 native 模型实现和模板未改；本轮没有重新声称进行一轮独立的 native 回归或新的 shell 求积收敛扫描。

详细证据见 `IR_VOLUME_V1_AUDIT.json`；全部拟合见 `ir_kmax_scan_convergence_v1.json`。

## 文件与复现

- `ir_kmax_scan_convergence_v1.pdf`：7 页 V=1 拟合/通过门槛图。
- `ir_volume_comparison_v1.pdf`：1 页当前校准下 V=42.875 与 V=1 的直接对照。
- `IR_VOLUME_V1_AUDIT.json`：验证、源文件哈希、变更映射和对照数值。

NERSC 结果目录：

```text
/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/halo/outputs/local_png_tree_ir_20260930/
```

新增/修改的代码均在本任务仓库 `03_bk_model/halo/code/`：

1. `run_tree_png_ir_kmax_scan.py`：加入显式 --volume 和 Profile(volume=...)，默认保持 42.875，PDF 使用实际体积标签；未改理论模型。
2. `audit_tree_png_ir_volume.py`：新增本次体积对照的独立验证及汇总图。

原扫描脚本保存在 `03_bk_model/halo/work/tree_ir_v1_20260930_originals/`；原 V=42.875 结果及其 manifest 保持冻结，不覆盖。

```bash
cd /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/halo/code
PYTHONDONTWRITEBYTECODE=1 /global/homes/l/lzy/anaconda3/bin/python run_tree_png_ir_kmax_scan.py \
  --templates tree_ir_templates_r6_m120_s0.25_rbao110.0.npz \
  --tag convergence_v1 --volume 1 \
  --variants fixed_order strict_kir0.15 factorized_kir0.15
PYTHONDONTWRITEBYTECODE=1 /global/homes/l/lzy/anaconda3/bin/python audit_tree_png_ir_volume.py
```

只输出 PDF 图，未生成 PNG。
