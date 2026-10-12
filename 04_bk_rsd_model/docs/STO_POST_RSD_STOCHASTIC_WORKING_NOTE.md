# STO+Marisa post RSD：随机项的工作推导，不用于正式拟合

更新时间：2026-10-10（北京时间）。状态：独立点过程审计进行中，非 Poisson STO 与有限位移尚未闭合。

## 最新数值证据与边界（2026-10-10 05:35）

06:28:30更新：`audit_post_rsd_cross_leg_all44.py`已经实际完成44-bin的原双网格88次独立input-pair卷积回放，最大fNL-scaled差2.5429e-15，原1e-9门未改，用时745.03秒。实际单核PID1877859已退出，报告`noise_cross_all44_v1/AUDIT.json`及SHA`5d370be503b4aa4e89736d3c227731ffb5f718c93625be65c24ba0bf3b2522b0`独立保留。结合PN_same/NN的88次scalar/integer证据，三个unit-amplitude算子都有全44几何/数值的独立验收；不是actual STO噪声幅度、connected/PNG闭合或physical-p校准，不改变production模型/先验，不放行主MCMC/IR。以下05:35叙述保留为历史检查点。

实际8核理论job59597420已COMPLETED/exit0，完整44-bin的396节点及88细网格探针经独立系数回放通过；原1e-9误差门不改。PN_same/NN另已完成44-bin、原双网格共88次scalar/integer闭合检查，最大缩放差2.4549e-14。上述旧审计记录和源文件不改，不能将它们称为PN_cross的独立全网格验收或实际STO随机项闭合。

新增`post_rsd_cross_leg_oracle.py`仅为PN_cross提供独立input-pair卷积检查：直接从实际谱构造D=P_L Z1²、从冻结catalog设置构造V=k R_*/k²。对于两条shell腿x/y及无限制closing腿z=-x-y，六项为

\[
V_x\cdot z\,D_y+D_xV_y\cdot z+V_z\cdot y\,D_x+D_zV_x\cdot y+V_y\cdot x\,D_z+D_yV_z\cdot x.
\]

实现只对两个input factors做加权卷积，随后显式查找负波矢的closing因子；不调用production `values/transform/integral`、不把weighted卷积四舍五入。分母仍为原full-shell Ni*Nj，三腿源零模从numerator删除，closing没有第三腿cut或输入shell限制。这是固定有向算子的代数重表达，不是新的噪声物理假设。

小网格逐对mode枚举、三种shell pair、三组b1/p、有限fNL[-500,0,73,500]、禁止生产投影访问、零腿/分母/无shift和错误几何保护共5项，在本地和NERSC各通过。证据`NOISE_CROSS_ORACLE_UNIT_AUDIT.json`。**当前真实44-bin交叉腿回放尚未执行**；入口`audit_post_rsd_cross_leg_all44.py`在实际原login parent为同一PID/starttime的T态、全部在途children终态、34项活跃控制源一致且有4GiB内存余量后才可启动。它只读取已验收的原双网格/系数，不改变正在运行的测量源码。即使完成，也不校准actual STO N(k)、connected/PNG噪声或physical p，不选择正式零腿定义，不放行MCMC/IR。

## 为什么不能把 self-contraction 通过等同于噪声模型通过

实际代码用 weighted data 构造位移，再对 data 和 uniform random 施加同一个位移。
测量的 factorial subtraction 去掉重复的外部粒子指标，但位移源中仍包含这些粒子。
因此不能将原 pre 的 Gaussian stochastic bases 直接标为完整 post stochastic。
这一一般问题亦见 [Zhao et al. 2410.18524 Sec. 2.4](https://arxiv.org/html/2410.18524v3#S2.SS4)
和 [Sugiyama 2406.01001](https://arxiv.org/html/2406.01001v1)。
下述具体算子与候选修正是本任务从实际 estimator 出发的推导，不声称是论文公式的直接抄用。

## 固定点集的一阶坐标响应

先取等权 data、连续 uniform reference，体积设为 1；使用整数周期 Fourier modes。
定义 Fd(k)=N^-1 sum exp(-2pi i k.x)，Fd(0)=1，F1(k)=Fd(k)-delta(k,0)。
在固定 data 生成的 shift 上乘独立测试振幅 eta，不改变 catalog 的 fNL_rec：

\[
s(x)=-\sum_{q\ne0}\frac{i q}{2\pi q^2}\mathcal R_*(q)F_d(q)e^{2\pi i q\cdot x}.
\]

同位移相减的导数为

\[
\left.\partial_\eta F_{\rm rec}(k)\right|_0
=-\sum_q\frac{k\cdot q}{q^2}\mathcal R_*(q)F_d(q)
[F_d(k-q)-\delta_{k,q}].
\]

而 self-pair 所需的 data-squared weight field 导数是

\[
\left.\partial_\eta F_2(k)\right|_0
=-\frac1N\sum_q\frac{k\cdot q}{q^2}\mathcal R_*(q)F_d(q)F_d(k-q).
\]

**后一个式子不能也删去 k-q=0 的均匀背景。** Uniform random 的 F2 在连续极限为零，
但其 F1 位移响应并非零。这是 estimator 标签需要独立跟踪的原因。
直接微分 `prod(F1)-sum(F2(-ki)*F1(ki))+2F3(0)`，可得到无需 tree/source/native 核的响应审计。

## 有限 N 的精确集合分割检验

采用强度 1+a cos(2pi ks.x+phi)、a=.02，phi 均匀随机，固定 N 的独立点过程。
通过集合分割精确求三/四点 Fourier moments：一个含 m 个 Fourier factors、s 个不同粒子指标的分割
权重为 (N)_s/N^m；每个 block 的波矢和决定单点强度矩，最后对 phi 精确平均。
无需 Monte Carlo，也没有通过噪声实现筛选结论。

`audit_post_rsd_poisson_response.py` 检查三角形 (1,0,0)、(0,2,0)、(-1,-2,0)，
三个 shift responses 为 .71/.42/.26，分别只在一条腿放强度信号，N 从1000到1000000。
均匀无信号测试中，raw 与 self 导数相消至约 8e-26；有信号的 leading 1/N 系数支持下列有向候选：

\[
\Delta B_{\rm shift,self-sub}^{P\epsilon}
=-N_2\sum_{i\ne j}P_{11}(k_i)
\frac{(k_i+k_j)\cdot k_j}{k_j^2}\mathcal R_*(k_j).
\]

这里等权测试的 N2=1/N，物理体积下 weighted 候选 N2=V sum(w²)/(sum w)²；
P11 为连续 tracer 的有向信号功率，而不是包含白噪声的 raw P。
上述 weighted 延伸尚未由这个等权测试证明，不能据此确认 STO halo 的非 Poisson 结构。

对应的 N*响应：候选分别为 -6.280e-5、-7.620e-5、0；精确 N=10^6 的结果为
-6.27996756e-5、-7.61995774e-5、2.26e-10，有限 N 余项随1/N下降。
以固定 signal/response 尺度衡量，最大差5.55e-6，小于预定1e-4门。

## 与常用对称 R2 形式的映射仍是审查项

令 Tij=[(ki+kj).ki/ki²] Ri。
将上述论文式(44)的对称 R2 直接作为当前 factorial estimator 的候选，会给
`-N2/2 sum_pairs (Tij+Tji)(Pi+Pj)`。
在这组独立点过程条件下，其三个 leading 系数是 -6.690e-5、-5.910e-5、-1.300e-5，
与直接位移/指标标签推导不相同。因此不能未经 estimator/random/指标约定核对就套用它。
这不是对论文整体的否定，也不将目前的小型反例升级为实际 STO 样本的最终模型。

进一步独立检查用 `audit_post_rsd_poisson_coordinate.py`：在固定真实点集上直接计算
exp[-2pi i k.(x+eta*s)]，对 data 和连续均匀参考场同样移动，再作 factorial subtraction。
其导数与前述 Fourier 标签恒等式比较；状态以 POISSON_COORDINATE_AUDIT.json 为准。
2026-10-09 的 8/17/32/79 点四组检验通过，最大复数导数相对误差为 1.08e-10，预定门1e-7；
这独立确认了上述实际坐标/参考场/factorial 的导数表达式，不认证 weighted/non-Poisson 闭合。

仍需：不同三角形/shift support、有限 random、真正 marked weights、有限 shift 幅度、非 Poisson
halo 排斥/噪声响应，以及对应精确 shell 投影与注入恢复。当前只有 audit 新增，
**没有将任何候选 stochastic 修正加入 production likelihood，也没有启动 post MCMC。**

## 2026-10-09 新增的 marked/finite-random 验收与零模待查项

`post_rsd_marked_moments.py` 保留 D1 作为纯 data 的位移源、F1=data-random 和
F2=权重平方的 data+random。两个标记族可有不同权重和不同正弦强度响应。
MARKED_FINITE_RANDOM_AUDIT 已通过：四组真实带权点集的坐标导数最大相对差2.36e-10；
两组三角形、三种信号腿、nran=2/30 的精确矩支持同一个 leading N2 噪声候选，
大 N 误差7.87e-6 < 1e-4。这里验证了 specified marked iid 模型，不是实际 STO 排斥模型。

另有一个需要单独核对的 estimator 边界：F1(0)=0 并不使 factorial-subtracted
B(0,k,-k) 自动为零，逐点恒等式实际给出

\[
B_{\mathrm{fac}}(0,k,-k)=-2\,\mathrm{Re}[F_2(k)^*F_1(k)]+2F_3(0).
\]

当前理论将任意零腿 source 置零，但测量 full-shell self-contractions 包含这些点。
因此必须显式量化零腿 self-terms，并决定如何使理论与数据在保持原 full-shell denominator
和 bin registry 的前提下一致；不能把这个差异当成一般物理噪声或先调 nuisance 抹掉。
此处先记录推导待审，不改变既有 data、shot subtraction、denominator 或 pre 结果。

## 零腿诊断和候选投影的实际验收

`audit_post_rsd_zero_atoms.py` 对零腿集合并集作独立精确枚举（避免三个零腿交集重复计数），
诊断最大缩放差6.77e-9，非零腿反事实差6.73e-7，分别通过5e-7/5e-6门。
`audit_post_rsd_zero_pilot.py` 确认新增诊断的真实 fid000 pilot 与旧 matched pilot 的原22个数组
逐位相同；去零腿 numerator 的44点 B 改变量相对L2为 .0803%，最大受影响bin约1.05%。
未据此做正式 estimator 选择或用 nuisance 吸收差异，旧 pre/likelihood 完全不变。

`post_rsd_poisson_projection.py` 为上述有向候选提供单独显式入口：SR=R_*、D=P Z1²，
负号通过 k_closing=-(ki+kj) 写为正的 closing-vector contraction；六个有向配对保留。
有限 PNG 下 D 已含0/1/2次系数，固定 SR 不含物理 fNL 次数，从而 G/L/Q 分别投影。
N2 必须显式传入且非负/有限，不混入随机点密度；旧 no-IR 确定性计算完全不变。
三个回归检验点级直接式、零腿/对角/off-diagonal完整闭合枚举、full-shell分母、关闭位移和保护。
仍不包含实际 STO 非 Poisson 响应或 finite-shift/high-order corrections，不是已放行的 likelihood。

## 相关child / signed marks 的独立点过程边界（2026-10-09）

新增`post_rsd_clustered_moments.py`按独立parent标签求一至四点精确矩，但同parent
具有多个固定相对位置/权重的child。因此child不是iid/Poisson；F2/F3扣除的仍是
实际child权重幂，不能将parent总权重当成一个粒子。允许signed weights，N3不加正值假定。
`test_post_rsd_clustered_moments.py`用不调用set partitions的显式parent标签枚举核验，
也检查single-child iid极限，以及真正坐标指数位移对Fourier导数的回归。

NERSC实测`CLUSTERED_NONPOISSON_RESPONSE_AUDIT.json`通过：5项独立测试，最大坐标导数
相对误差4.5095e-10；相关等位置child的闭式结果相对误差6.2767e-16。
这里设parent点数N、各child权重w_a，h=sum(w_a²)/(sum w_a)²。均匀无连续信号时，
保留有限random并按实际child扣self后，一阶同位移响应为

\[
\partial_\eta B_{\rm fac}|_0=-\frac{(1-h)(1-2/N)}{N^2}
\sum_{i<j}\left[T_i^{ij}+T_j^{ij}\right],\qquad
T_i^{ij}=\frac{(k_i+k_j)\cdot k_i}{k_i^2}\mathcal R_*(k_i).
\]

等权双child取h=1/2；固定示例三角形/response的N²响应趋于-0.695，而仅有
N2*连续P的iid候选在无信号时给零。有限分离、不同强度和signed marks例子的
signal response与iid候选最大缩放差0.3045。这些是指定cluster模型的边界反例，
**不是测得真实STO有30%偏差，更不是已完成STO非Poisson闭合。**

## 有向噪声transport与参数依赖的表示

`post_rsd_noise_transport.py`复用唯一FFT投影，分开输出以下unit-amplitude算子：

\[
K_\times=-\sum_{i<j}(T_iD_j+T_jD_i),\quad
K_{\rm same}=-\sum_{i<j}(T_iD_i+T_jD_j),\quad
K_{NN}=-\sum_{i<j}(T_i+T_j),\qquad D_i=P_L(k_i)Z_1(k_i)^2.
\]

Gaussian raw transport为N_noise*(K_cross+K_same)+N_noise²*K_NN；它不是
factorial-subtracted STO模型，mixed-weight self响应和噪声幅度必须另审。
三个算子也不等于三个已认证nuisance。固定R不随物理b1/p/fNL变，因此两个PN算子
在b1/p上各至二次、fNL严格至二次，NN的L/Q为零。这一证明不适用于IR指数。

似然接口现在显式支持原nuisance系数乘参数多项式形状；不增加自由参数、不改原bounds，
仍先在完整P+B空间用同一个joint Cholesky白化。独立手写Chebyshev、非节点PN物理式、
全shell闭合枚举、constant limit和Case适配均测试通过；不自动选择实际噪声形状。
原44-bin、396节点、88细网格probe模板构建入口为
`build_post_rsd_noise_transport_templates.py`；compute队列将按4GiB/单核worker预算并行。
所有产物标记为operator templates、NOT_STO_CLOSURE；不生成正式noise PASS或主MCMC。

## 独立标量/整数闭合验证（2026-10-10）

对于三个非零闭合腿，sum_{j!=i} T_i^{ij}=R_i，因此
K_same=-sum_i D_i R_i、K_NN=-sum_i R_i。新独立oracle不用生产vector contractions、
transform或integral，而是以两个非零input masks的整数卷积计算每个closing mode的
pair multiplicity，再直接按物理Z1²、冻结R逐格点求和。closing为0的配对从numerator
删除，分母仍用原full-shell Ni*Nj；整数FFT舍入残差有独立1e-5门，不能改科学谱/系数。

`NOISE_SCALAR_TWO_BIN_PROBE_AUDIT.json`实际运行bin1的32/48、bin99的144/160网格，
vector/scalar/integer三种方法最大fNL-scaled差8.9733e-15，原门1e-9不变。
最高bin原分母1,641,060,100，非零closing配对1,641,019,590；保留分母而删零腿的规则
被单独验证。六项数值保护测试也通过。它只是两个实际bin的理论探针，
不校准p_halo、不证明实际STO随机项幅度/映射，更不宣布全部44数值产品已完成。

`audit_post_rsd_noise_transport_coefficients.py`独立手写Chebyshev回放全部396节点/88探针，
并检查NN不能含物理fNL、b1/p非恒定项只能是roundoff，所有原source/contract/array SHA
必须一致。需要实际59594709 COMPLETED/exit0且完整44-bin/无worker；PENDING、观察缺失、
部分产物或单个PASS字符串不能放行，不能以本节的小范围检查替代全范围验收。

## 2026-10-10：raw噪声与self-moment分离的leading pair推导

此前相关child反例表明不能把实际raw随机功率N(k)设为粒子self N2。
新增`post_rsd_general_noise_transport.py`明确接受已经独立校准、data-only的N(k)，
不接受random或已扣self后的负残余作为raw noise。这里暂把N(k)视为不随拟合fNL变化
的外部输入；物理有限PNG响应若存在，须另校准，不能由此假定为零。

以下为本任务从实际标签估计器作出的推导，不是将论文的对称R2公式直接移植。
设D_i=P_L(k_i)Z1(k_i)^2、N_i=actual raw data stochastic power、N_r为独立均匀random
白噪声。data-only shift的Gaussian pair contraction使用
C_DF,i=D_i+N_i，C_FF,i=D_i+N_i+N_r；F2零模均值是N2_data+N_r，不能误删。
令T_i^ij=[(ki+kj).ki/ki²]R_i。raw pair响应为
-sum_pairs[T_i C_DF,i C_FF,j+T_j C_DF,j C_FF,i]，扣外部self后加
(N2_data+N_r)sum_i R_i C_DF,i。用闭合恒等式sum_{j!=i}T_i^ij=R_i，N_r在这一leading
阶精确相消；再减去deterministic D_iD_j项得到

\[
\Delta B_\epsilon=-\sum_{i<j}(T_i^{ij}+T_j^{ij})
\left(D_iN_j+D_jN_i+N_iN_j\right)
+N_{2,\rm data}\sum_i R_i(D_i+N_i).
\]

若实际N_i=Nraw是白噪声，使用已有三个有向unit-amplitude算子，化为

\[
\Delta B_\epsilon=N_{\rm raw}K_\times+
(N_{\rm raw}-N_2)K_{\rm same}+N_{\rm raw}(N_{\rm raw}-N_2)K_{NN}.
\]

Poisson极限Nraw=N2退回已有N2*Kcross候选；coincident correlated-child的
Nraw=1/Nparent、N2=h/Nparent则恢复之前纯noise响应的leading
-(1-h)sum(T_i+T_j)/Nparent²。signed weights允许h>1和负的扣self残余，
但raw N仍必须非负；代码不通过clipping将负残余冒充正的raw spectrum。

六项检验在本地和NERSC通过：独立手写三种Wick pairing/F2均值、不同N_r的leading相消、
white算子/Poisson极限、finite-fNL²有向point式、三种full-shell/zero-policy闭合枚举、
相关/colored/signed-child精确矩渐近极限。后者三条signal legs、nran2/30、Nparent=1000/10^6
都保存；大N的最大相对差7.8089e-6<原1e-4门，随1/N缩小。
详情`GENERAL_COLORED_NOISE_TRANSPORT_AUDIT.json`。这解决了指定cluster反例在leading pair阶
的缺项，不是测得STO拟合误差降到7.8e-6。

该入口保留原唯一FFT、物理Z1²的G/L/Q、two-shell full denominator、无第三腿cut及零腿规则。
仍未验证实际STO N(k)、connected3/4随机响应、PNG noise-response、finite-shift/IR或正式零腿
估计器选择；未修改production likelihood、自由参数或先验，不创建正式noise PASS或启动主链。
