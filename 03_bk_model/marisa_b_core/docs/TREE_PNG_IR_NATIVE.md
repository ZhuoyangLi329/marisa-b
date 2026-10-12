# Tree-level finite-local-PNG IR implementation

实现范围遵循 `TREE_LEVEL_PNG_IR_RESUMMATION_PLAN.md`：覆盖全部现有 local-PNG
halo tree 确定性算符与 stochastic raw bases，保留到 `fNL**2`，对长位移指数
非微扰，对 BAO wiggle 严格保留一阶。它不是 PNG hard-loop 完备模型。

## 入口和代数合同

- Python oracle：`python/marisa_b/tree_png_ir_resummation.py`。
- Native：`src/marisa_b/tree_png_ir.{h,cpp}`；公开 pre triangle 与 fixed-b_rec
  post vector 入口，返回当前 `ComponentResult`，并提供逐腿数组接口。
- 批量 binding：`python/marisa_b/tree_png_ir_native.py`，保留 21 个原子字段，
  另提供 Gaussian F2/b2/bK2 三个子项，便于 nuisance 模板组合。
- 投影节点：`src/marisa_b/tree_ir_lattice_nodes.cpp` 复用现有 exact-lattice
  radial/angular measure，保留原 shell-pair 分母和零模权重。

`strict_lo` 计算 `X_nw + sum_i exp(-k_i^2 Sigma^2) dX/deps_i`，并令
`M_nw=M sqrt(P_nw/P)`。Native 使用一次方向导数注入所有 `D_i P_w,i`；
一阶导数的线性性使它与三次单腿导数求和严格等价，也能处理等边/等长腿。
没有有限差分步长，也没有 `(1+k^2 Sigma^2)` 的 one-loop 匹配补偿。

`factorized_closure` 是高阶 wiggle 系统学分支；`fixed_order` 是对照。
`strict_lo` 在 Sigma=0 时恢复 single-wiggle tree，故另保存 `strict_zero`
分支区分 wiggle 截断效应和真正的 IR damping。

`NativeConfig.ir_sigma2` 在这些新入口中表示当前红移与重构状态的位移方差。
post 入口要求调用者显式使用 residual-displacement Sigma_rec；输入本身只能
表达 fixed-b_rec。CIC-on scalar IR 会抛异常，不能借此运行 adaptive 模型。
Python oracle 对 CIC 的 source algebra 可用于诊断，但 native production
IR 路径明确拒绝把它作为完整 tensor-covariance 模型。

## 编译与测试

NERSC 现有依赖位置：

```bash
cd /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/marisa_b_core
make tree-png-ir REACT_DIR=/pscratch/sd/l/lzy/reconstruction-png/codes/marisa_b/external/ACTio-ReACTio
PYTHONPATH=python /global/homes/l/lzy/anaconda3/bin/python -m pytest \
  tests/python/test_tree_png_ir_resummation.py tests/python/test_tree_png_ir_binding.py -q
```

构建只读旧仓库中的第三方头文件和源文件，重新编译本任务当前 native 用于回归，
不会修改旧仓库，也不需要安装包。构建及默认数值运行使用一个核。

验证覆盖当前 native 逐字段回归、随机 bias/triangle、pre/fixed-post、独立
finite difference、Python dual oracle、置换/旋转/等边/挤压/展平几何、
无 wiggle/零 Sigma 极限以及 `P/M^2` stochastic invariance。

## B-only scientific validation

实现与实际验证入口位于相邻 `halo/code/`：

1. `inspect_tree_ir_inputs.py` 冻结已有模型、五点真实模拟均值、Fid500 covariance
   和 nuisance priors；旧数据与模型只读，新输出在当前仓库。
2. `prepare_tree_png_ir_templates.py` 在同一个 z=0.5 P/M 表上冻结共享 Pnw；
   对实际闭合 triangle 做全部 IR 运算，然后分别执行原投影。
3. `run_tree_png_ir_kmax_scan.py` 比较 fixed、strict、factorized、strict-zero
   与四个 k_IR；使用相同 covariance/Hartlap/volume/priors。
4. `audit_tree_png_ir_scan.py` 复核历史拟合、独立 Minuit、合成 injection、
   更密求积、逐 bin 模型变化及理论预选的 BAO-sensitive bins 消融。

Gaussian 沿用 exact-lattice measure；确定性 PNG 沿用经过检验的有限盒
continuum shell 规则。PNG stochastic 保留历史冻结的 projection adapter，
该 adapter 只由 no-IR leading stochastic 决定，并不随 IR/sigma/fNL 变化。
常数 residual shot term 保持原 likelihood 的常数定义。所有 Gaussian/PNG
分支共享同一 Pnw，不在 bin center 额外乘 damping。

当前 B000 的 `kmax` 切在两个测量 shell 的 weighted k；第三腿通过角积分确定，
可能超过这个数值。因此报告的 kmax 不能直接解读为所有三角形三条腿的共同上限。
index 0 和 k≈0.155 shell 的历史 exclusion 保持不变；0.14/0.15 是同一组 bins。

所有新模板位于独立目录
`../halo/outputs/local_png_tree_ir_20260930/`，原子项、sigma2、k_IR、r_BAO、
split 参数、输入文件与 native source/library 的 SHA256 都保存在 NPZ/JSON 中。
原冻结结果没有被覆盖。

此次 N-body 接受范围是 **pre-reconstruction 的已测试数据与参数合同**。
fixed-post/CIC-off 的代数验证不能代替真实 catalogue inference；原有
adaptive/CIC post 数据仍需对应的 tensor covariance 和 fNL-dependent
exponent 推导及验证。此次结果不提升原 post 模型的 scientific status。
