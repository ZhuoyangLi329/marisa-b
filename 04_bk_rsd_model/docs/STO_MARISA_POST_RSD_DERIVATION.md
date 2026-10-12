# STO Marisa 固定重构操作的 tree RSD 推导与验收

日期：2026-10-08。范围仅 STO+Marisa；pre 是冻结对照。当前已经有单 realization 的 post-RSD pilots，仍不是正式 ensemble 或 MCMC 完成报告。

## 从加权场的同位移映射出发

令 x 为已经加上 LOS 速度位移的红移空间坐标，w 为与 halo 身份对齐的冻结 STO 权重。以平均加权密度归一化得到 delta_w^s。data 与均匀参考场使用同一实际位移 s_rec；不另加去除线性 RSD 的 LOS 位移。连续极限中，两者的均匀项严格相消：

\[
\delta_{\rm rec}(\mathbf k)=\int d^3x\,e^{-i\mathbf k\cdot\mathbf x}
e^{-i\mathbf k\cdot\mathbf s_{\rm rec}(\mathbf x)}\delta_w^s(\mathbf x).
\]

这是 Shirasaki et al. [2010.04567](https://arxiv.org/html/2010.04567v2) 式 (5)-(8) 的同位移映射；将确定的 tracer 权重纳入 delta_w 的延伸只限于匹配的连续场。权重离散矩、有限 random、位移噪声和可能的 velocity/selection bias 必须另审，不能由此等式宣布 stochastic 闭合。

令 lambda=fNL 为拟合参数。物理 f_g、算法 fNL_rec、算法 b_rec/p_rec 与独立自由参数 b1 的职责必须分开。对固定 catalog，

\[
D_*(q)=b_{\rm rec}+f_{\rm NL}^{\rm rec}\,b_{\phi,\rm rec}\alpha_{\rm rec}(q),\quad
b_{\phi,\rm rec}=2\delta_c(b_{\rm rec}-p_{\rm rec}),\quad
\mathcal R_*(\mathbf q)=W_{\rm rec}(\mathbf q)/D_*(q),
\]
\[
\mathbf s^{(1)}_{\rm rec}(\mathbf q)=-i\frac{\mathbf q}{q^2}
\mathcal R_*(\mathbf q)\,Z_1^s(\mathbf q;\lambda)\delta_L(\mathbf q).
\]

W_rec 包含 Gaussian smoothing 和需要显式验证的 Cartesian assignment/readout response。本轮公式使用连续主像；不能把 sinc^4 近似等同于包含 alias 的实际有限网格算子。正式 catalog 放行需要单模响应、网格和随机数密度回归。D_* 不随 likelihood 的 lambda 改变，且触发 bias floor 的样本/参数不得继续用未裁剪的理论。

实际谱表审计发现重构 desilike-prim alpha_rec 与旧物理 source 的 1/M 相差约 0.929%，将 2048 节点加密至 8192 后差异仍存在；不能归因于普通插值误差，也未唯一确定其物理归一化来源。alpha_rec 是冻结算法输入，物理 Z1/Z2/primordial 外腿仍消费原 M：两者单独保存。新的 tabulated-response ABI 只改变 shift，独立测试断言 Z1 与 primordial 项不受 alpha_rec 的改变影响。此处理是匹配实际算子，不是修改物理 PNG convention 或重拟合 fNL。

## 二阶 kernel 与符号

展开上式到 delta_L 的二阶：delta_rec^(1)=delta_w^(1)，delta_rec^(2)=delta_w^(2)-i k.s_rec^(1) delta_w^(1)。因为 (-i)(-i)=-1，令 k=q1+q2，可得

\[
Z_{1,\rm rec}^s=Z_1^s,\qquad
Z_{2,\rm rec}^s=Z_2^s-\frac12\left[
\frac{\mathbf k\cdot\mathbf q_1}{q_1^2}\mathcal R_*(\mathbf q_1)+
\frac{\mathbf k\cdot\mathbf q_2}{q_2^2}\mathcal R_*(\mathbf q_2)
\right]Z_1^s(\mathbf q_1)Z_1^s(\mathbf q_2).
\]

没有额外 s_rec^(2) 的独立 tree 项：它乘均匀背景的贡献已在 data-random 相减中消失，乘 delta^(1) 则是三阶。上述结构也见 Sugiyama [2403.18262 Appendix A](https://arxiv.org/html/2403.18262v1#A1)。不能把上述同位移映射与 RecIso 或 RecSym 的 catalog 定义混用。

使用 Z1=b1+f_g mu^2+lambda bphi/M；Z2 包含原实空间 bias 源、f_g mu12^2 G2，以及使用完整 PNG Z1 的非线性 RSD mapping。f_g 始终是物理增长率，即使位移求解器为 bias-only 而设置其算法 growth_rate=0，也不把理论 f_g 置零。

## 有限 PNG 系数与固定算子

\[
B_{\rm rec}=\left[2\sum_{\rm cyc}P_iP_jZ_{1i}Z_{1j}Z_{2,\rm rec,ij}
+\lambda\widehat B_{\rm prim}Z_{11}Z_{12}Z_{13}\right]_{\lambda^{0,1,2}}.
\]

固定 R_* 下的新增项为

\[
\Delta B=-\left[\sum_{\rm cyc}P_iP_jZ_{1i}^2Z_{1j}^2
\left(\frac{\mathbf k_{ij}\cdot\mathbf k_i}{k_i^2}\mathcal R_i+
\frac{\mathbf k_{ij}\cdot\mathbf k_j}{k_j^2}\mathcal R_j\right)\right]_{\lambda^{0,1,2}}.
\]

闭合三角形中 k_ij=-k_ell，因此 FFT 可分离项可写成正的 k_ell,a k_i,a/k_i^2 形式。每条外腿 Zi 不能先取 Gaussian 部分。固定非零 fNL_rec 的 Gaussian 系数也会改变，这与旧 eta 路径在 lambda=0 时 fNL_rec=0 不同。

旧模式 D=b_rec+eta*lambda*bphi_rec/M 是另一条参数路径。旧 API 与旧库保持原义；新 fixed ABI 拒绝与 eta 混用。fixed 路径不对 D_* 按 lambda Taylor 展开。即使在某一点 fNL_rec=lambda，固定路径与联动路径的导数/截断系数也不能互换。

## Binning 与 bias 插值

沿用两腿 B000 registry 和原全 shell 分母；零模源置零，不补第三腿 cut。closing leg、CIC 的 Cartesian 方向依赖不得丢弃。新位移项最高含 b1^4，因此旧 pre 的四节点三次 b1 插值不再充分；post 至少使用五节点四次基底，并用独立非节点值校验。

物理 p_halo 与 p_rec 独立标记。首轮先保留当前 RSD PBS family，用显式 p_halo 替代硬编码 pre p；其 bphidelta recipe 与实空间约定的既有差异必须作为共同系统性记录，不仅修改 post 来改善真值恢复。

## 验收层次与当前数据门

### 加权 self-pair 与 self-triple 的精确点过程恒等式

将实际位移后的 data 和 random 合并为带符号系数 c_a 的点集；例如 data 为 w_a/sum(w)，uniform random 为 -1/N_r。定义 F_j(k)=sum_a c_a^j exp(-ik.y_a)。对固定的闭合三角形，去除相同粒子指标的三重乘积是

\[
F_1(k_1)F_1(k_2)F_1(k_3)-\sum_{\rm cyc}F_2(k_i+k_j)F_1(k_\ell)+2F_3(0).
\]

这是 inclusion-exclusion 恒等式，与点集是否由带噪位移生成无关；小型直接枚举可独立核验。乘合适的 box/estimator normalization 后才成为 B 的单位。对仅含等权 data 的情形，它退化到 raw B-sum(P_raw)/nbar+2/nbar²。

STO 的 F2 是权重平方场，不能把以上混合谱简单替换成普通 P/n_eff。实际 painted/compensated estimator 还必须核对窗口、bin、随机归一化和 mixed contractions。即使逐粒子 self-terms 全部正确扣除，位移由带噪加权密度估计仍会产生相关噪声响应；该恒等式不证明后者或有限 PNG stochastic 的理论闭合。

### 分层测试与生产状态

1. 旧 ABI 与旧共享库随机三角形逐列回归；新库不覆盖冻结的 pre 库。
2. 固定参数的独立完整标量多项式，与 native G/L/Q 比较。
3. 在周期网格直接计算上述指数坐标映射，以小振幅正负差分提取二阶响应；独立验证 shift 符号、1/2 和 RSD-PNG 外腿。
4. fNL_rec=0、关闭滤波响应、f_g=0、LOS/三角形对称性以及无效分母拒绝。
5. post FFT 对直接闭合 mode 枚举；五节点 bias 插值；再与实际重构单模响应比较。
6. 权重/covariance/随机项门通过后才能生产并拟合。代数 PASS 不代表科学闭合。

主分析只做 STO+Marisa：18 B-only、3 P0-only、18 P+B；数据 mean100、Fid500 C_single、V=1。最初计划要求独立Fid500，但当前配置的DM目录不含fiducial1000..1499；旧fid4000 source是halo-only lam3，不能替代目标matter-assisted env_lam1_gsr2_thr15。2026-10-09用户明确改选原实空间fiducial0..499，并接受与fiducial target0..99重叠；正式选择保存在`ENSEMBLE_SELECTION.json`，700个唯一catalog，不能称为严格独立mock recovery。仍须测量相同算法/estimator下的完整post covariance，禁止以旧pre covariance、较少盒数或其他STO方法替代，也不按100/500平均covariance。

IR 是通过以上基线后的独立阶段：从实际 R_* 推导各向异性残余位移张量，先 dressing 有向源再做 estimator 投影。本文不将固定参数树核测试冒称为完成 RSD IR。

## 实际 weighted estimator 的离散 self-contraction 审计

2026-10-08 的独立 signed-point 测试发现，当前共享 raw B 使用 TSC/interlacing=3，
而 shot helper 默认 CIC/interlacing=0；S113 还用 bin-center 的 j0(kmean r) 代替实际离散 shell window。
这些是已定位的数值差异，不是新的物理 stochastic 参数。原测量与失败的 WEIGHTED_ESTIMATOR_AUDIT 均保留。

新 `post_rsd_estimator.py` 只在显式 context 内将 shot paint 匹配到 raw，并替换 S113：

\[
W_i(x)=N_i^{-1}\sum_{q\in i}e^{iq\cdot x},\qquad
S_{113,ij}=\sum_x C(x)W_i(x)W_j(x),
\]

其中 C(x) 沿用原 helper 的 F2/F1 cross 和 S111 项；其他 contractions、raw 和 normalization 不改。
因此 \(\sum_x C(x)W_iW_j\) 等于对实际两腿 shell 的 closing-leg contraction 求平均，
而不是对 bin 中心求连续角平均。S111、S122、S121 的小型有权点枚举一起通过了独立检验。
完整三对小型 bins（包括 diagonal 和 zero-leg）的最大误差为 6.79e-7，
尺度是 raw/shot/corrected/all-same 中的最大绝对值；预设门为 5e-6。
这不意味着 corrected B 的相对误差也为 6.79e-7，更不意味着 shift-noise 理论通过。

`_selfmatched` pilot 独立保存；其 `B_post_bincenter_S113_with_TSC_shot` 是 TSC shot 下的 bin-center 对照，
不是原 CIC shot 的完整 legacy B。若将该修正作为正式 estimator，必须使用匹配的新 pre 控制与 post covariance，
不得把新 post 和原 pre 的未标记差异误称为重构增益。

## RSD 残余长位移：先验算 tensor，再实施 IR source dressing

令 \(R_{ab}=\delta_{ab}+f_g n_an_b\)。从实际同位移映射和线性 RSD 物质位移得到

\[
\Psi_{\mathrm{rec},a}^{(1)}(q)=i[d_{0,a}(q)+\lambda d_{1,a}(q)]\delta_L(q),
\]
\[
d_{0,a}=\frac{R_{ab}q_b-\mathcal R_*(q)(b_1+f_g\mu_q^2)q_a}{q^2},\qquad
d_{1,a}=-\frac{\mathcal R_*(q)b_\phi q_a}{q^2\mathcal M_{\rm nw}(q)}.
\]

这里 \(\lambda=f_{\rm NL}\)，算法的固定 fNL_rec 只进入 \(\mathcal R_*\)。
这是 [Sugiyama 2403.18262 Sec. II](https://arxiv.org/html/2403.18262v1#S2)
的 residual displacement 结构应用到本任务固定算法与有限 PNG 线性响应的推导；
论文的 Gaussian constant-bias 算法不能直接当作本任务的完整 PNG 公式。
实际重构 runner 从 weighted data 本身估计位移，不使用随机场估计位移，
故有限 random 的采样噪声不能自动加到 shift-noise 功率里。

对 Gaussian 长模，定义相对位移张量而非仅一个各向同性 Sigma：

\[
A_{ab}(r;\lambda)=2\int_{q_{\min}}^{k_{\rm IR}}\frac{d^3q}{(2\pi)^3}
P_{\rm nw}(q)[1-\cos(q\cdot r)]
[d_{0,a}+\lambda d_{1,a}][d_{0,b}+\lambda d_{1,b}].
\]

连续主像下它精确为 \(A_0+\lambda A_1+\lambda^2 A_2\)，任意真实 lambda 的总矩阵半正定。
同一算法的 alpha_rec 表始终是冻结外部输入；本子模块不随物理 fNL 或 wiggle bookkeeping 改它。
代码要求显式提供 P_nw/M_nw，后续 source 分解应保持 P_phi=P/M²。
可选加上假定独立的 data-shift noise 积分，但不能默认以 1/nbar 代表 STO 噪声。

逐腿方向 \(r=r_{\rm BAO}\hat k\) 的单-wiggle LO 近似将使用
\(E_i=\frac12 k_aA_{i,ab}k_b\)。指数系数是

\[
e^{-E(\lambda)}=e^{-E_0}\{1-\lambda E_1+
\lambda^2(E_1^2/2-E_2)\}+O(\lambda^3).
\]

它是 Gaussian 长位移下的有限-PNG指数展开，不是“把 fNL=100 的一个标量 damping 固定后用于全部 MCMC”。
`post_rsd_displacement.py` 只实现 tensor 和指数子模块；完整三角形 source wiggle 响应、
投影、噪声与模拟检验仍未完成，因此不能将 DISPLACEMENT_AUDIT 的 PASS 标成完整 RSD IR 完成。

独立验算包括无重构的 RSD 解析 tensor、实空间完全匹配 shift 极限、零分离、
有限 PNG 二次协方差/半正定性、CIC 网格对称与 LOS 反向，以及 Gaussian soft kernel：

\[
Z_{2,\rm rec}^{G}(q,k)\xrightarrow{q\ll k}
\tfrac12[k\cdot d_0(q)]Z_1^G(k)+O(q^0).
\]

该 soft 检验只覆盖 Gaussian 系数；PNG 长势调制及其高阶累积量并未由它自动证明。

### 同一 native 核的逐腿 wiggle source oracle

`post_rsd_ir_oracle.py` 不重写 Z2 或 PBS 方程，而是调用同一 fixed-response native source。
保持 \(P_\phi=P/\mathcal M^2\)，令 \(\mathcal M_{\rm nw}=\mathcal M\sqrt{P_{\rm nw}/P}\)。
对第 i 条腿用 \(P_i=P_{{\rm nw},i}t^2,\mathcal M_i=\mathcal M_{{\rm nw},i}t\)。
检查当前 tree source 的每个有限-PNG系数可知，它对单独的 t 至多二次，因此

\[
B_{{\rm w},i}=\frac{P_{{\rm w},i}}{2P_{{\rm nw},i}}
\frac{B(t=1+h)-B(t=1-h)}{2h}
\]

是这个 source 的精确逐腿线性 wiggle 响应，而不是需要 h 趋于零的数值微分。
每次调用额外核验第四点的二次多项式残差，并比较 h=.2/.4；未来 source 若新增更高阶依赖会直接拒绝。
Gaussian/Bias 与 primordial 源项分别保留。实际 frozen alpha_rec 表是必需输入，防止 source 的 M 扰动意外改变重构算法。

然后用逐腿的 D0、d1、d2 混合 G/L/Q，与实空间的有限 PNG 记账同构，但 damping 来自上述 RSD tensor。
关闭 damping 时返回 strict O(Pw) 的 source，不保证与包含 Pw² 的 full fixed-order tree 完全相等；
full - linearized 的余项应按 wiggle 幅度平方缩放。初次该测试使用随幅度变化的相对误差分母，
近零 G bin 导致比值 3.80；保留失败 JSON 和原测试 SHA，改为固定 no-wiggle source 范数后比值约
4.0000/4.0008/4.0009，原 .008 相对门不变，native/source 公式未改。

该入口限制为最多 64 个有向三角形的小批量 oracle，不是生产模板器。
完整 estimator-matched IR 投影、参数插值、weighted stochastic 和模拟恢复仍是放行条件。
尤其 E0/E1/E2 随物理 b1/bphi 改变后，IR 结果不再是 b1 的四次多项式；
no-IR 五节点的精确性证明不能用于 IR 模板，后者需要重新做参数插值与协方差加权误差验收。

## 2026-10-09：有向 IR 的球谐张量与共享 FFT 投影

对每个 q 定义三阶 PNG 系数的 Cartesian spectral tensor
\(S_{c,ab}(q,\hat q)=P_{\rm nw}(q)[d_0d_0,d_0d_1+d_1d_0,d_1d_1]_{c,ab}\)。
这里 c 是物理 fNL 次数，不是 tensor 分量；可选 noise 必须显式输入且另行验证。
用实正交球谐展开各向异性角核，固定 \(r=r_{\rm BAO}\hat k\) 后得到

\[
A_{c,ab}(r\hat k)=\sum_{\ell\,\mathrm{even},m}Y_{\ell m}(\hat k)
\frac1{\pi^2}\int dq\,q^2
\left[\delta_{\ell0}-(-1)^{\ell/2}j_\ell(qr)\right]
\int d\Omega_q\,Y_{\ell m}(\hat q)S_{c,ab}(q,\hat q).
\]

这由平面波展开得到，保留全部 m、Cartesian CIC 窗和 LOS，不将 damping 先角平均。
`post_rsd_ir_harmonics.py` 复用原 long-mode quadrature/source；对象带 b1/bphi/growth/LOS、
reconstruction 与 smooth-spectrum 哈希。参数不匹配会拒绝，禁止跨 b1 隐式复用旧 exponent。
当前数值探针取 qmin=2pi/1000、kIR=.15、rBAO=110，均为计算设置而非新增拟合参数。
qmin/IR split/rBAO 的物理稳健性仍须在正式同数据控制中检查，不能由 quadrature 收敛替代。

共享可分离 source 的单腿 Fourier factor F 在 \(P\to t^2P,M\to tM\) 下带幂 \(s_d\)：
`A/D/DR:2-d`，`T:1-d`，`P/Pmu2/Pmu4:2`，`one/mu2/kz:0`；显式 invM 再减1。
固定 shift response 不随 t 改变。逐腿 wiggle 因子是

\[
\widetilde F_d^{\rm w}=\frac{P_w}{2P_{\rm nw}}D_0
\sum_{j=0}^{d}g_{d-j}\,s_jF_j^{\rm nw},
\qquad(g_0,g_1,g_2)=(1,-E_1,E_1^2/2-E_2).
\]

`post_rsd_ir_projection.py` 不重写 source 方程或 convolution；将每个原 contraction
调用为 nw + 每次一条腿带 wiggle 的三个项。cyclic legs、full-shell denominator、
零源腿处理、closing leg 不切 kmax 均继承同一 pre FFT 引擎。它显式拒绝未经验证的 stochastic。
`post_rsd_ir_grid.py` 只在可达的 closing sphere 计算 damping，用精确整数格点反射及
横向 x/y 交换对称减少重复计算；要求 LOS=z、兼容 angular rule，绝非角平均近似。

实际冻结 P/M 与第三部分 Pnw 快照完全一致，但这里只复用谱分解，不复用实空间 damping。
与直接角核比较的 tensor 误差4.80e-12；513/1025径向节点的源差2.65e-8。
实际 bins1(0,1) 与99(9,9)在 theory FFT144/160 上通过，最大缩放差小于7.2e-14；
精确对称加速与未加速结果差小于5.86e-15，耗时约降为1/9–1/10。
测试的物理 p=1.31 只是数值探针，不是校准值；不据此冻结生产参数、44点全域插值或 MCMC。

## estimator 的零腿边界与 leading marked-Poisson 响应

按实际 signed-point factorial estimator，零腿仍有
\(B_{\rm fac}(0,k,-k)=-2\operatorname{Re}[F_2(k)^*F_1(k)]+2F_3(0)\)。
新诊断独立枚举该项，并保存 `B_post_nonzero_only_counterfactual=B_post+zero_leg_shot_contribution`；
这仅是去掉零腿 numerator、保留 full-shell denominator 的反事实，旧 B22个数组逐位不变。
若正式采用非零 source 投影，就须将数据和匹配的新 pre/covariance 一起按同一规则处理。
零腿不能当成普通常数 nuisance，亦不能静默更改 triangle 数据密度。

独立的 marked iid、finite-random、坐标导数与精确有限 N moments 支持一阶位移响应

\[
\Delta B^{P\epsilon}=-N_2\sum_{i\ne j}P_L(k_i)
\left[b_1+f\mu_i^2+f_{\rm NL}b_\phi/M(k_i)\right]^2
\frac{(k_i+k_j)\cdot k_j}{k_j^2}\mathcal R_*(k_j),
\quad N_2=V\frac{\sum_d w_d^2}{(\sum_d w_d)^2}.
\]

这里 N2 不含 random 白噪声，因为 random 不生成位移。
`post_rsd_poisson_projection.py` 以 `SR=R_*`、`D=P Z1²` 和 closing-vector factor
直接复用原 FFT，显式返回 fNL 的 G/L/Q 三系数；关闭位移/N2=0返回零，旧确定性源逐位不变。
三个测试验证任意三角形直接物理式、三组 shell 的完整闭合枚举（含零腿及原分母）与输入保护。
这只是经 specified Poisson 模型验证的候选，不包括 STO halo 排斥、环境相关噪声的全部响应，
不自动加入 likelihood，也不声称完成 stochastic IR dressing。

## 当前 physical p 校准不能直接沿用历史数字

实际比对旧 p 校准 source 的 diagnostics/权重：历史 GSR=1，当前 GSR=2，fid000/LC_p000
权重相对 L2差7.52%/7.98%，即使 halo 行完全相同也不是同一 tracer。
因此 p_halo 不能直接使用历史1.33946或历史 post1.28276，算法 p_rec=1.2822仍保持不变。
新增LC_p100..249的weighted-input RSD P0准备，其编号与主验证0..99分开。最初协方差计划为1000..1499；用户后改选fiducial0..499后，校准与fiducial covariance可能共享100..249的IC相位，必须明确标记，不再声称校准与covariance严格相位独立。
测量仍复用原 RSD velocity/mass selection、原 P0 bins/measure_pk；首盒 actual binner 的六个理论
谱函数核验误差9.99e-15，尚未拟合/采用 physical p。holdout 不是主协方差替代，也不改变 V=1。

2026-10-09 北京时间02:49:38，独立150份P0的全部来源/窗口/测量哈希验收通过；
仅汇总150×12矩阵与均值，不创建替代covariance，不产生p或fNL后验。

## finite-PNG IR 的阶数约定与宽先验高阶敏感性

用户参考方案明确采用 source 和 exponent 同时闭合到 fNL²，而非把部分高阶项静默加入主模型。
这里新增的未展开指数比较只使用同一 Gaussian tensor，定义

\[
B_{\rm exp}(\lambda)=B_{\rm nw}^{(\le2)}(\lambda)+
\sum_i e^{-E_{i,0}-\lambda E_{i,1}-\lambda^2E_{i,2}}
B_{{\rm w},i}^{(\le2)}(\lambda).
\]

这不是另一套完整高阶PNG理论；strict模型保留其Taylor系数到二阶，其余只作高阶敏感性诊断。
在原P/M、fixed rec256、p=1.31数值探针下，检查b1=.5/2.228/4、32个闭合整数三角形及
fNL=0/±100/±500等点。fNL=0逐位一致；中心b1的±100最大总B差/绝对source尺度5.75e-5。
但b1=4、fNL=500处二次damping factor在一条腿达到-0.00476，未展开指数保持正值。
同一角落总B差/绝对source尺度最大3.39%；这既不是B的直接相对误差，也不是covariance单位。
实际estimator投影和全先验白化验收仍必需，不能以裁剪负数、缩窄先验或静默改为B_exp来绕开。
这一诊断不修改当前G/L/Q、固定算法参数或旧结果；科学IR效果仍待完整同数据控制。

## 全44-bin数值验收与当前输入依赖门

2026-10-09北京时间03:29:49，全部44个原拟合bins在同一参数探针上的双theory-grid检查完成。
两格均满足原no-wrap条件；保留原shell分母和无限制closing leg，未改数据点密度。
另一个入口从88份grid checkpoint独立复算最大缩放误差1.44e-12，按|fNL|<=500
缩放G/L/Q后的最大误差8.64e-13；预设门1e-9不变。它不证明未知physical p、整个b1空间、
真实STO噪声或模拟恢复，不能作为39组MCMC完成的替代。

当时（北京时间03:32）的逐项审计为 `INPUT_DEPENDENCY_GATE_AUDIT.json`：300份主验证STO源、150份独立校准P0输入
和上述数值检查已完成，physical p拟合、post ensemble/covariance、主MCMC及同数据IR效果仍未完成。
复查指定fiducial1000..1499：500份raw FoF存在，但正确DM/权重输入0份；另一族500份STO均为
halo/halo-split、GSR1、lambda3，不能替代DM/DM、GSR2、lambda1。goal已在此输入门标记blocked，
等待所需目录或明确的数据方案变更，不缩小完整目标、不更换方法或协方差口径。

此段是历史检查点。后续goal已恢复，并完成fiducial0..19的20盒post pilot及实际B窗口验收，
详见README和GOAL_PROGRESS。1000..1499是助手选择的不重叠子集，不是RSD物理要求；
原实空间0..499正在作为候选复核，不能把旧子集输入缺口外推为所有实空间输入不可复用。

## physical p校准与模板生成可分离的代数结构

以下只针对当前PBS family、固定catalog重构算子和确定性源，不宣称真实STO随机项闭合。
令lambda=fNL。当前代码中b_phi=2 delta_c (b1-p)，b_phidelta对p为一次，b_phi2不依赖p。
每个物理p因子随至少一个lambda出现；重构响应R_star消费固定p_rec而不消费物理p。
因此截断到lambda平方后，每阶系数对p的最高次数不超过相应lambda次数：

\[
B=G(b_1,b_2,b_{K^2})+\lambda L(b_1,b_2,b_{K^2};p)
  +\lambda^2 Q(b_1,b_2,b_{K^2};p),\qquad
\deg_p G=0,\quad\deg_p L\le1,\quad\deg_p Q\le2.
\]

no-IR的固定shift项最多包含四个Z1，因此b1最高四次；b2与bK2仍是仿射。
于是可在同一个现有FFT源上采样5个b1节点、3个物理p节点，保存张量多项式系数，
最终p校准完成后再代入p并降为仅b1的五节点表示。p节点是数值坐标，不是p校准结果、
新增MCMC自由参数或新先验；不能将p_rec或旧GSR1校准直接代入当作物理p。
这也不意味着已经生成/验证这些正式模板；仍需对非节点参数作独立源/FFT回归和最终误差验收。

在本任务采用的Gaussian-long、strict O(Pw)、lambda平方IR方案中，E0不含物理p，
E1对p一次、E2对p二次，所以严格展开的G/L/Q保留上述p次数界。
但是exp(-E0)依赖b1，IR的b1轴不是精确四次多项式，必须独立认证其插值。
上述p平方表示也不能用于未展开的B_exp高阶参照，或用于把p_rec与物理p联动的另一种算法。

## 2026-10-09配对随机点与正式模板推进

同一fiducial0..19、rec256/meas256的nested nran30/60控制已20/20完成并验收，
post halo坐标与STO权重逐份哈希相同。B(kmax=.20)的最大配对平均差为原20盒逐点
单盒std的0.04389，P0(kmax=.08)为0.02505；这不是全协方差范数或fNL偏差。
配对差的std/sqrt(20)仅估计数值控制平均差的不确定性，绝不用于缩小主C_single。
两种零腿定义均保留，尚未据此把任一正式estimator选择或随机点密度宣布为已认证。

`build_post_rsd_noir_templates.py`已实际启动上述5x3张量模板构建；所有44个原分箱
均要求两个非节点、较细theory grid探针，按|fNL|<=500缩放G/L/Q的数值门为2e-10。
部分完成数以实时STATUS为准；全部模板及最终C500下的误差验收尚不能由部分节点推断。
另已启动独立matched-pre单盒对照，复用原pre测量入口并显式使用同一个self adapter，
要求核对raw B与旧pre的差异，避免将测量器差异与post重构/IR效果混为一谈。
