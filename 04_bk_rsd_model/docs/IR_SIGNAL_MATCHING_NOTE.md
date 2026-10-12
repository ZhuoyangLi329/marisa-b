# Post-RSD IR信号：与原no-IR基线的matching边界

日期：2026-10-10。状态：数值诊断与实现说明，不是正式P+B模型放行。

## P与B必须沿用同一信号约定

`post_rsd_ir_p0_projection.py`直接复用B使用的`IRPointFields`、`D`与`TaggedField`。
P采用原生产窗口的9170个非零模式、12个labels/weights/nmodes，不改成连续mu平均。
physical p、b1和growth用于Z1及长位移；重构的b_rec/p_rec/fNL_rec仍完全冻结。
当前入口只提供Gaussian长位移、严格单-wiggle、fNL二次截断的确定性信号，
不处理未知STO raw噪声或loops，也不自动采用粒子self N2作为位移噪声。

[Sugiyama 2406.01001，第V节](https://arxiv.org/html/2406.01001v1#S5)
说明重构离散效应也进入IR阻尼，而不只影响最后的self subtraction；其文中模型包含
1-loop项。本任务的新入口没有实现这些1-loop项，亦不把Poisson幅度直接搬到STO中。
下列PNG spectral-t与matching公式是对本仓库既有信号约定的代数核验，非该论文的直接公式。

## 为什么零阻尼并不自动等于完整tree

定义t=sqrt(P/Pnw)，Mnw=M/t，保持P/M²不变，eta=(t²-1)/2。
对每个有向mode，R=b1+f*mu²，bphi=2*delta_c*(b1-p)，则平滑源为

\[
C_G=P_{nw}R^2,\quad C_L=2P_{nw}R b_\phi/M_{nw},\quad
C_Q=P_{nw}(b_\phi/M_{nw})^2.
\]

严格single-wiggle的源扰动是W_G=2*eta*C_G、W_L=eta*C_L、W_Q=0。
阻尼系数为D0,d1,d2，保留G/L/Q后有

\[
P_G=C_G+D_0W_G,\quad
P_L=C_L+D_0(W_L+d_1W_G),\quad
P_Q=C_Q+D_0(d_1W_L+d_2W_G).
\]

设D0=1、d1=d2=0，G与Q等于原完整线性tree源，但

\[
P_L^{strict}-P_L^{exact}=\frac{C_L}{2}(t-1)^2.
\]

这是截断产生的O(Pw²) matching余项，不是IR damping；physical p=b1时bphi=0，
它相应消失。B还有多腿wiggle乘积的余项，因此也不能仅用“Sigma=0”字样宣称
回到原完整no-IR源。旧no-IR与旧IR实现、数据和先验均未修改。

## 本次实际证据与下一步

`ir_p0_signal_actual_v1/AUDIT.json`使用原真实谱、原1025/48/96/lmax12位移规则，
在p=1.31数值探针和完整b1先验范围内的7个probe上回放全部12个P分箱；独立
手写Z1/t导数与逐mode求和最大scaled差7.9922e-16。三组uncached长位移投影也通过。
9/17/33节点表示最大scaled差分别3.1478e-10/1.5188e-14/3.5594e-14，原1e-9门不变。
共63个直接bias参数点，用时54.74秒、单CPU13、峰值93.60MiB。

真实模式上零阻尼matching余项的最大fNL-coefficient-scaled量为1.2679e-4；
这是按[-500,500]范围的系数尺度，不是sigma、chi²或fNL偏移。它尚未通过原
C_single500白化，也尚未与B的相应余项一起验收。

正式IR对照前必须明确matching处理：可以额外比较相同严格wiggle阶数的undamped/damped
源，也可以单独推导并验证保留完整hard-tree remainder的tree-matched表示。
不能把未经核验的余项加进模型，也不能静默把原no-IR基线改成single-wiggle基线。
届时应在同一实际p校准、同一原数据/先验/C_single500下量化此差异，避免把baseline
matching变化算作IR效果。当前没有选择production matching策略、放行MCMC或认证fNL恢复。
