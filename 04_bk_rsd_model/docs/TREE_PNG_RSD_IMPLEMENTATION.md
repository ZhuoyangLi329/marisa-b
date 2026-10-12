# MARISA-B tree-level local-PNG RSD：实现合同

## 实现层次

第四部分新增 native 有向核，复用 `03_bk_model/marisa_b_core/src/marisa_b/marisa_b_native.h` 的 `HaloBiasV1Params`、`NativeConfig` 和 closed-triangle 类型。没有修改旧 real-space API、旧 IR 核或旧 MCMC 产物。编译输出为本部分 `work/build/libmarisa_tree_rsd.so`。

用户提供的参考包原样保存在 `work/reference_package/`；C++ 独立实现方程，Python reference 只用于类型/数值 oracle，不能替代原生核验收。

## 严格 tree source

全局固定 LOS、无 selection/velocity bias、无 AP/window/wide-angle，使用线性 P 和相同红移的 M。`f_growth` 与 `fNL` 是两个不同量：

\[
Z_1(\mathbf k)=b_1+f_g\mu^2+f_{\rm NL}b_\phi/M(k).
\]

\[
Z_2=K_2^{\rm real}+f_g\mu_{12}^2G_2+
\frac{f_g(\mathbf k_1+\mathbf k_2)\cdot\hat n}{2}
\left[\frac{\mu_1}{k_1}Z_1(\mathbf k_2)+\frac{\mu_2}{k_2}Z_1(\mathbf k_1)\right].
\]

最后一项使用完整 PNG `Z1`，不是仅用 Gaussian `b1+f_g mu²`。原有 bias 约定保持 `b2 delta²/2` 与 `bK2*(nu12²-1/3)`；`bphi2` 是独立参数，不是 `bphi*bphi`。外部 EFT 参数需先转换 `b2_native=b2_EFT-4*gamma2/3`、`bK2=gamma2`。

三个 cyclic pairs 的 `2 Pi Pj Z1_i Z1_j Z2_ij` 加上 primordial `fNL B0hat Z1_1 Z1_2 Z1_3`，通过截断多项式解析返回 G/L/Q。Q 是 fNL² 的系数，不是二阶导数。所有模块统一截断至二阶；不把有限 fNL 的更高阶外腿乘积混进结果。

## post source 的明确边界

实现 `same_shift_no_linear_rsd_removal`：Z1_rec=Z1，Z2_rec 加上当前 partition map 的二阶 shift 项。支持 Gaussian smoothing、Cartesian CIC 和 `b_rec+eta*fNL*bphi_rec/M` 的解析 reciprocal 展开。这个 source 在 `f_growth=0` 时与第三部分既有 native 逐项一致。

它不是 RecIso/RecSym，也不是 Kaiser-divided reconstruction。native enum 中这三种尚未实现的方法必须报错；Python `denominator_mode='kaiser'` 同样拒绝执行。STO 或 MARISA 的实际 RSD reconstructed catalog 需要单独确认算法/shift 合同后才能匹配，不能只按名称启用。

当前没有生成 RSD recon 样本。本轮只验证 pre 的实际数据。

## 投影与 stochastic

`NativeTreeRSD.project_multipole` 实现 k1 轴的 Scoccimarro/Legendre B_l，归一化 `(2l+1)/(4pi) integral dmu dphi L_l B`。不是 TriPoSH B_l1l2L。Cartesian CIC post 禁止使用两角平均，需实际三角形 modes 或完整 SO(3)。

`project_exact_lattice` 对输入的实际闭合 modes 加权；显式 normalization 可保留 estimator 中 excluded atoms 的原分母。首次数据入口使用 strict-three-leg B0：三条腿均在 `[.01,.025,.04,.055] h/Mpc` shell 范围，FFT estimator 的 ordered triangle counts 必须与直接 Cartesian 枚举完全相等。它与旧两腿 B000 的 120-bin geometry 不混用。

Gaussian stochastic raw bases 独立返回：

\[
S_B=\sum_i P_i b_1(b_1+f_g\mu_i^2),\quad
S_P=\sum_i P_i f_g\mu_i^2(b_1+f_g\mu_i^2),\quad S_A=1.
\]

这不是有限 PNG stochastic closure。首次实际测试默认不添加该 closure，采用单位权重点过程的自配对/自三重扣除：`B_factorial=B_raw-sum(P_raw_per_triangle)/nbar+2/nbar²`，所有 P 项按相同真实三角形 mode multiplicity 平均。这一关系对理想 point Fourier field 精确；实际 interlaced-CIC 是其数值近似，assignment 残差由 128/256 网格比较控制，不能称为对任意 painted field 的逐粒子精确扣除。残余非泊松 stochastic、loop/FoG 等物理未被这一步消除。

## pre-RSD 样本与单位

读取原始 Quijote FoF `GroupPos`、`GroupMass`、`GroupVel`，不下载新模拟。先以既有 float32 position 和 float64 mass 导出约定重建数据，必须与共享 mmin1e13 real-space 文件逐行完全相等，再使用同一 mask 的速度。

固定 `z=.5, Omega_m=.3175, L=1000 Mpc/h, LOS=z`，

\[
v_{\rm physical}=\mathrm{GroupVel}(1+z),\quad
H_h=100\sqrt{\Omega_m(1+z)^3+1-\Omega_m},\quad
s_z=(x_z+v_z(1+z)/H_h)\bmod L.
\]

这是 [Quijote 官方 FoF 速度约定](https://quijote-simulations.readthedocs.io/en/latest/Examples/Reading_FoF.html)。`GroupVel` 不可直接当物理 km/s，转换成 Mpc/h 后不可额外乘 h。横向坐标不变。metadata 保存原始文件/位置哈希、行顺序匹配、速度单位、wrap 数目和 LOS 位移 RMS。

真实样本：fiducial real000（fNL=0，309740 halo）和 LC_p real000（fNL=100，309602 halo），各 V=1 (Gpc/h)^3。没有重构。

## 首轮模型测试

使用 interlaced CIC + assignment-window deconvolution，128 网格测量并以 256 检查收敛。P0/P2 和 B0 均固定全局 LOS。P0/P2 在已知 fNL 真值与 f_g=.7625790037720044、p=1.179 下诊断拟合 b1；B0 在相同真值下仅 profile b2_native/bK2。高阶 PNG response recipe 复用第三部分现有 PBS 关系，并明确记录这一假设。

不拟合 fNL、不作 MCMC、不使用 real-space covariance、不报告正式 goodness-of-fit p-value。图中 Gaussian diagonal error 只是单盒尺度指示，并非冻结 RSD likelihood covariance。需要 ensemble 才能检验 fNL 无偏恢复与可用 kmax。

首轮严格关闭 IR、FoG、counterterms、loops。reference 的 `pre_recon_rsd_ir_covariance` helper 已暴露；它只提供 leading anisotropic covariance，不能称为完整 RSD IR wrapper。特别是 post 的位移张量需根据实际 reconstruction 推导，不能复制 real-space Sigma 或先投影再阻尼。

## 最小模型例子

```python
import numpy as np
from tree_png_rsd_native import NativeTreeRSD, HaloBiasV1, RsdConfig

v=np.array([[.052,.013,.027],[-.018,.047,-.011],[-.034,-.060,-.016]])
bias=HaloBiasV1(2.08,.41,-.27,3.25,1.36,2.42)
x=NativeTreeRSD().evaluate(v,[1180.,970.,830.],[820.,1070.,910.],
                         bias,RsdConfig(.7625790037720044))
fNL=100.
B=x.total[0]@np.array([1.,fNL,fNL**2])
print(x.total[0],B)  # G,L,Q 与有限 PNG prediction
```

执行 `bash code/build_rsd.sh` 编译，然后在现有 desilike Python 环境执行 audit 与 tests。数据/model 生产用 `sbatch code/pre_rsd_model_test.slurm`；Slurm stdout 建议显式放到 `outputs/logs/`。所有数值线程固定 1，不安装软件包；图只生成 PDF。
