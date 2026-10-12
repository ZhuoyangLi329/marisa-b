# 实际 marked factorial estimator 的连通项响应

2026-10-10。范围：固定data-only shift的一阶独立测试振幅eta，非零闭合外腿，真实data权重与有限独立uniform random。以下是本任务从粒子标签和坐标微分推导的恒等式，不是已放行的STO stochastic likelihood。

## 标签与均值

使用整数周期Fourier modes，data系数c_a=w_a/sum(w)，random系数c_r=-1/N_r：

\[
D_j(k)=\sum_a c_a^j e^{-2\pi i k\cdot x_a},\qquad
F_j(k)=D_j(k)+(-1/N_r)^j\sum_r e^{-2\pi i k\cdot x_r}.
\]

D1(0)=1、F1(0)=0、F2(0)=N2_data/V+1/N_r，F3(0)=N3_data/V²-1/N_r²。F2非零mode是真实权重平方场，不能替换成N2*F1。

三真值real000实测中，k=.02.. .08、两个|mu|区间的归一化w²场与RSD matter的cross相对w场约1.105..1.151。128/256最大auto-normalized mixed谱差约4.3e-5。此处每真值只有一盒，不给ensemble显著性、pure stochastic幅度或fNL偏差。

## 完整一阶响应

外腿k_l+k_a+k_b=0、q非零，固定算法响应R_*(q)，g_l(q)=(k_l.q/q²)R_*(q)。shift源是D1(q)，不是F1(q)。对原factorial三重乘积

\[
\widehat B_{fac}=\prod_lF_1(k_l)-\sum_lF_2(-k_l)F_1(k_l)+2F_3(0)
\]

微分得到

\[
\partial_\eta\langle\widehat B_{fac}\rangle|_0
=-\sum_{l,q}g_l(q)[E^{(4)}_{lq}+E^{(3,a)}_{lq}-E^{(3,b)}_{lq}],
\]

\[
E^{(4)}=\langle D_1(q)F_1(k_l-q)F_1(k_a)F_1(k_b)\rangle,
\]
\[
E^{(3,a)}=\langle D_1(q)F_2(-k_l-q)F_1(k_l)\rangle,\qquad
E^{(3,b)}=\langle F_2(-k_l)D_1(q)F_1(k_l-q)\rangle.
\]

后两项来自两个self-pair因子的微分；F3(0)坐标响应为零。全部权重标签必须保留。post_rsd_marked_moments.factorial_shift_response是唯一标签遍历，新模块不复制符号/循环。

平移齐次非零外腿的四点矩含三种pair contractions和connected fourth cumulant；三点矩含F2可能的零模均值乘二点矩，以及connected third cumulant。post_rsd_marked_cumulants.py显式Möbius反演cumulants，再用集合分割重建raw moments，分出pair+means、全部third-cumulant、fourth-cumulant响应。三项和必须回到完整导数；缺第三/第四cumulants时full接口拒绝，而非默认0。

connected4也可能含确定性非线性/PNG trispectrum，必须进一步作tree power counting和signal/noise分离，不能将全部connected4当作新tree噪声模板。

## 有限uniform random的精确条件积分

data固定时，对random Fourier因子按random粒子指标分割pi求和，每项为

\[
(N_r)_{|\pi|}(-1/N_r)^{\sum_jp_j}
\prod_{A\in\pi}\mathbf1[\sum_{j\in A}k_j=0].
\]

F因子逐项展开data/random，D始终只含data。这给出任意真实带权data的有限N_r条件一到四点矩，无需生成并直接DFT九百万random，也不将random加入shift source。这是对independent uniform random的精确期望，不是单个随机种子的有限shift样本。

独立测试以两个真实random点在19个周期坐标格逐一穷举，对照条件矩；本测试波矢和不alias。另检查独立中心三/四点矩、显式parent标签、真实坐标导数、逐粒子distinct三重指标、Gaussian pair/random相消、缺项拒绝、精确particle DFT和multirow组装。9项在NERSC实际通过，报告MARKED_CUMULANT_UNIT_AUDIT.json。它们证明接口/恒等式，不证明STO完整物理模型。

## 真实300样本诊断

measure_sto_marked_connected_probes.py读取原三真值各0..99的300份已准入GSR2权重/FoF，做精确particle DFT并保留w¹/w²/w³场。固定b_rec=2.2280546353635335、p_rec=1.2822、fNL_rec=100、R10、rec256和原alpha，不随目标真值改算法。整体平移相位平均只令非闭合矩为零，不增加simulation数。

先测3个有向闭合三角形，每个12个非零shift modes，总计169个需要的Fourier modes；这是有限q支撑缺项探针，不是原44-bin full-shell window或完整production shift。第一/第二外腿在.08以内，closing不追加kmax限制。原mean100 data vector、700测量、C_single/V1、先验和MCMC kernel均不改。

noise_actual_connected_probes_v1保存来源哈希、完整/分解响应与单盒散布。尚未分离pure epsilon和确定性高阶项，不能直接充当认证的B噪声系数，更不能将三目标真值外推为整个fNL先验域已验证。

参考包TREE_LEVEL_PNG_RSD_FORMULAE.md第7节和README明确没有唯一finite-PNG RSD stochastic closure；旧native real-space的PNG stochastic字段也不是简单Z1²替换。必须补推导/校准，不能将算子/moment接口通过写成formal noise PASS。内部/外部shot标签的区分可对照[Zhao et al. 2410.18524, Sec.2.4](https://arxiv.org/html/2410.18524v3#S2.SS4)；本文件marked公式以自身独立标签审计为依据。
