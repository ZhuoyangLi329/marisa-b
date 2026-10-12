# MARISA-B tree-level local-PNG IR 与 B-only kmax 检验

任务日期：2026-09-30。任务仓库：`/pscratch/sd/l/lzy/reconstuction-png-v2`。

## 结论

已实现所给计划定义的 **覆盖完整现有 local-PNG halo tree 算符基底的 leading-IR、leading-wiggle resummation**，并在实际 Quijote pre-reconstruction B000 数据上完成 no-IR / IR 对照。

**当前 p-aware 校准下，IR 减小部分 fNL 恢复偏差，但没有把通过既有检验的 kmax 从 0.10 提升到 0.12 h/Mpc。** 旧 p=1.12 校准下则有一个较小的门槛改善：0.08 → 0.10 h/Mpc；旧版在 0.10 原本只是边缘失败，不能把这称为大幅延伸 tree 模型有效尺度。

| 固定的 PNG bias 校准 | no-IR 最大连续通过 kmax | strict LO IR 最大连续通过 kmax |
|---|---:|---:|
| 当前 p=1.1788659191136683 | 0.10 | 0.10 |
| 旧 p=1.12 | 0.08 | 0.10 |

这里的上限限于已扫描档位和以下明确的数据/先验/体积合同，不是任意 tracer 或 survey 的通用有效尺度。

## 模型实现

默认 `strict_lo` 计算

\[
X_{\rm IR}=X_{\rm nw}+\sum_i e^{-k_i^2\Sigma^2}
\left.\frac{\partial X}{\partial\epsilon_i}\right|_0,
\qquad
\mathcal M_{\rm nw}=\mathcal M\sqrt{P_{\rm nw}/P_L}.
\]

覆盖 Gaussian tree、primordial B111、bphi、bphidelta、advection、bphi²、独立 bphi2、fixed-brec post source kernel，以及现有 Gaussian/linear-PNG/quadratic-PNG stochastic raw bases，保留到 fNL²。

Native 使用一阶自动微分；同时注入各腿的阻尼 wiggle 方向，与逐腿求导后相加等价。没有差分步长，没有 hard loop，没有 tree/one-loop matching 因子。M 与 P 一致变化，保证 P/M²=Pphi 不变。

保留三个对照：完整 fixed-order tree、Sigma=0 的 strict leading-wiggle tree、`factorized_closure`。最后一种保留高次 wiggle products，仅作为系统学变体。

所有 Gaussian/PNG 运算共享一份 z=0.5 smooth split，使用仓库原有的 Gaussian Eisenstein–Hu ratio 方法，ln k 平滑宽度 0.25、8192 个网格点。r_BAO=110 Mpc/h 是显式配置。位移积分使用同红移 Pnw，并准确包含 k_IR 上限。

| k_IR [h/Mpc] | Sigma² [(Mpc/h)²] |
|---:|---:|
| 0.05 | 10.28427934 |
| 0.10 | 14.34808851 |
| 0.15，主设置 | 16.53135534 |
| 0.20 | 17.82275892 |

IR 在实际闭合 triangle 上完成，然后才投影。Gaussian 复用 exact-lattice measure；确定性 PNG 复用现有有限盒 continuum shell 规则；保留原 shell-pair 分母、零模剔除和冻结的 stochastic projection adapter。没有给 bin-center 总 B 乘经验阻尼。

## 统一的检验设置

- 数据：pre-reconstruction B000，fNL = -100、-50、0、50、100；0 点用 500 个 realization 的均值，非零各 250 个。
- 统计量：仅 B000；本轮没有加入 P0 likelihood，也没有重新调整 p 来改善 B 拟合。
- bias：固定 b1=1.93；bphi=2 delta_c(b1-p)，delta_c=1.686；bphidelta 保留原 finite-tree 的 universal anchor，bphi2 使用原关系。
- 自由 nuisance：b2、gamma2、Ashot_residual、Bshot_residual；原先验保持一致。
- 协方差：Fid500 的 realization covariance，按 V=42.875 (Gpc/h)³ 缩放，再用相同 Hartlap 修正；没有采用 covariance/500 作为拟合误差。
- fNL 范围：沿用注入检验的真值±200；所有报告区间均未撞边界。
- 检验：Gaussian fixed-fNL=0 gate 的 p≥0.05；五个真值全部满足 profiled Delta-chi²≤1；优化与区间有效且未截断。
- kmax：0.08、0.10、0.12、0.14、0.15、0.18、0.20。0.14 与 0.15 因冻结的 k≈0.155 shell exclusion 而使用同一组 bins。

重要的估计量定义：B000 只对两条测量腿分 bin，第三腿由角积分确定，可能超过 kmax；本报告的 kmax 不能解读为所有实际 triangle 的三条腿都小于该值。

采用对四个线性 nuisance 的解析 profile，再对 fNL 做一维多极值搜索并解 Delta-chi²=1 端点；另用五维 Minuit 独立交叉检查。这些是 profile 区间，不是 MCMC posterior credible intervals；“五点通过”也不是完整的 frequentist coverage 测量。

## 实际模拟恢复结果

以下采用加密后的 nrad=6、nmu=120，IR 主设置为 strict_lo、k_IR=0.15。

| 当前 p-aware 校准的 kmax | no-IR 最大真值 Delta-chi² | IR 最大真值 Delta-chi² | no-IR / IR 通过真值数 | no-IR / IR Gaussian gate p |
|---:|---:|---:|---:|---:|
| 0.08 | 0.0395 | 0.1269 | 5/5，5/5 | 0.9819，0.9504 |
| 0.10 | 0.9837 | 0.7961 | 5/5，5/5 | 0.9948，0.9970 |
| 0.12 | 3.0673 | 1.6428 | 0/5，1/5 | 0.9961，0.9998 |
| 0.14 / 0.15 | 1.4544 | 2.2622 | 1/5，0/5 | 0.9994，1.0000 |
| 0.18 | 11.1401 | 6.7242 | 0/5，0/5 | 0.2205，0.9186 |
| 0.20 | 8.3044 | 7.2199 | 0/5，0/5 | 0.2261，0.9147 |

0.12 时 IR 后恢复值对应的偏差依次为 +16.55、+17.05、+12.15、+16.97、+18.46；只有 fNL=0 真值落入其 Delta-chi²≤1 区间。Gaussian 拟合优度虽然改善，但不能据此宣布 fNL 恢复可靠。

IR 的效果不随 kmax 单调改善：0.14/0.15 的恢复偏差反而变大。这里观察到的是部分误差被减小，剩余偏差仍足以使 0.12 及之后的档位失败，尚无证据支持把当前可用尺度升至 0.12。

旧 p=1.12 的高精度对照在 0.10 的最大真值 Delta-chi²从 1.0312 降至 0.9056，4/5 → 5/5，通过上限由 0.08 升至 0.10。原封存代码的旧结果为 1.0507；重新生成基准带来的细小差异已单独记录，未与 IR 效应混在一起。

四个 k_IR 值下，strict IR 的上限均为 0.10，且没有任何一个让 0.12 通过。factorized_closure 也给出相同的上限。strict_zero 则在 0.10 边缘失败，说明确实要区分位移阻尼与 leading-wiggle 截断的影响。

## 数值与科学审计

| 项目 | 结果 |
|---|---|
| 新 C++ 与任务当前 native 回归、独立有限差分及极限检查 | 28,400 项通过 |
| 用户提供的 Python oracle | 14 项通过 |
| C++/Python 逐原子项、置换、旋转和配置拒绝测试 | 8 项通过 |
| 独立 Minuit 与解析 profile 的最大 fNL 差异 | 8.2e-5 |
| 历史封存拟合的最大 fNL / 真值 Delta-chi² 复现差异 | 0.00113 / 1.53e-8 |
| nrad4/nmu80 → nrad6/nmu120，最大逐 bin 模型变化 | 约 0.0144 个有效拟合 sigma |
| 加密求积导致的最大 fNL 变化 | 0.134，低于 profile 半宽的 0.65% |
| 加密后的 kmax 判定 | 不变 |
| quadratic PNG stochastic 的 IR 不变性 | 数值差异约 1e-18 |

对 IR 模型生成的无噪声合成数据，用同一 IR 模型恢复五个 fNL，最大绝对偏差低于 7e-7。用 no-IR 模型拟合这些合成数据，在 kmax=0.20 的该组 nuisance 先验中心设置下产生约 +2.34～+2.72 的 fNL 偏移。这个测试检验实现自洽性，不能替代上面的真实模拟恢复。

另完成了逐 bin 的 Delta G、Delta L、Delta Q 及 fNL=0、±100 的总模型变化；strict/factorized 的单 bin 差异在所选参考 nuisance 下最多约 0.097sigma。详细数值见 `IR_SCIENTIFIC_AUDIT.json` 和 `ir_binwise_diagnostics.pdf`。

BAO-sensitive 消融使用理论 IR 变化与 covariance 预选最敏感的一半 bins，比较这组与其余一半的恢复、nuisance 与 chi²。因为 B000 已积分第三腿，这不是“完全去掉所有 BAO”的操作；该消融仅作诊断，不改变上述预设通过门槛。

## 代码位置与复现

模型：

```text
/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/marisa_b_core/
  src/marisa_b/tree_png_ir.h
  src/marisa_b/tree_png_ir.cpp
  src/marisa_b/tree_ir_lattice_nodes.cpp
  python/marisa_b/tree_png_ir_resummation.py
  python/marisa_b/tree_png_ir_native.py
  scripts/build_tree_png_ir.sh
  docs/TREE_PNG_IR_NATIVE.md
```

分析与结果：

```text
/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/halo/code/
  inspect_tree_ir_inputs.py
  prepare_tree_png_ir_templates.py
  run_tree_png_ir_kmax_scan.py
  audit_tree_png_ir_scan.py

/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/halo/outputs/local_png_tree_ir_20260930/
```

在 NERSC 使用现有环境即可，不需要安装包：

```bash
cd /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/marisa_b_core
make tree-png-ir REACT_DIR=/pscratch/sd/l/lzy/reconstruction-png/codes/marisa_b/external/ACTio-ReACTio

cd ../halo/code
PYTHONDONTWRITEBYTECODE=1 /global/homes/l/lzy/anaconda3/envs/desilike/bin/python inspect_tree_ir_inputs.py
/global/homes/l/lzy/anaconda3/bin/python prepare_tree_png_ir_templates.py
/global/homes/l/lzy/anaconda3/bin/python run_tree_png_ir_kmax_scan.py
/global/homes/l/lzy/anaconda3/bin/python prepare_tree_png_ir_templates.py --nrad 6 --nmu 120
/global/homes/l/lzy/anaconda3/bin/python run_tree_png_ir_kmax_scan.py \
  --templates tree_ir_templates_r6_m120_s0.25_rbao110.0.npz \
  --tag convergence --variants fixed_order strict_kir0.15 factorized_kir0.15
/global/homes/l/lzy/anaconda3/bin/python audit_tree_png_ir_scan.py
```

原有历史代码/数据作为只读输入，旧冻结拟合结果未覆盖。新模板单独命名，JSON 保存 scheme、共享 Pnw、Sigma²、k_IR、r_BAO、投影参数和输入/源文件/二进制哈希。改动清单和输入/输出哈希另见 `TREE_PNG_IR_CHANGE_MANIFEST.json`。

## 适用边界

本次完成 native pre 和 fixed-brec、CIC-off post 的算符实现与代数验证；真实 N-body kmax 结论只针对 pre。所给计划明确把 Cartesian CIC 的 tensor displacement covariance，以及 adaptive brec 对 damping exponent 的一阶/二阶 fNL 响应列为独立后续工作；它们没有被固定标量 Sigma² 代替，native 入口会拒绝 CIC-on scalar IR。因此本报告不升级原 adaptive/CIC post 模型的 provisional 状态，也不给这些后重构分支宣称新的 kmax。

已直接核对现有 post cache header：R=10、b_rec=1.9326526728794324、cell_size=8、cic_power=4，确实不属于本计划允许接受的 CIC-off 情形；adaptive-brec source 也有单独的已存模板。不能拿 CIC-off 的新 IR 预测直接拟合这套后重构数据，再宣称完整验证。

依据：用户提供的 `TREE_LEVEL_PNG_IR_RESUMMATION_PLAN.md`；[TSPT BAO/IR](https://arxiv.org/abs/1605.02149)、[TSPT primordial non-Gaussianity](https://arxiv.org/abs/1906.08697)、[ULPT before/after reconstruction](https://arxiv.org/abs/2602.20431)。
