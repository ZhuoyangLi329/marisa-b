# Post-reconstruction local-PNG tree IR：实现与验证报告

已将交付方案接入 NERSC MARISA-B C++ tree 内核，并提供 Python/批量 shell 模板显式入口。
适用于 std、Marisa 和 STO+Marisa 的方案参数路径；旧 likelihood 默认值与历史结果未改动。

## 本次补齐

- 固定/自适应位移的 Sigma(fNL) 到二阶；包括 std 分子的 scale-dependent bias 响应。
- 三个 adaptive 源基底、严格单 wiggle、G→L / L→Q / G→Q 指数混合。
- Cartesian CIC 逐外腿方向积分，以及 IR 后按原分母做 shell 投影。
- 可选 shift noise；stochastic 源项与 mixing 分离；不默认强加公共 alpha3 closure。

## 验证结果

- 当前 native 基底交叉检查：39,200 项通过；独立/回归 Python 测试：76 个通过。
- 真实 z=0.5 谱三方法探针，24x48/32x64/48x96 角积分 G/L/Q 最大 scaled error：4.167e-13。
- 1025→2049 径向网格：G/L/Q 最大 2.616e-12；位移方差最大 7.604e-10。
- 首个真实 FFT shell 306 节点、原分母 324；角积分变化最大 8.597e-15；分块模板与直接投影一致。
- qmin/kIR/rbao 敏感性、fNL=±100 方差截断检查已记录在 audit_fine/POST_TREE_IR_AUDIT.json。
- scaled error = max(abs(a-b)/max(1,abs(b)))，不是观测协方差加权误差。

## 不能混淆的边界

- fNL_rec=eta*fNL：std eta=0，adaptive 示例 eta=1；固定非零 fNL_rec 目录需要另一个展开。
- current-p 示例使用 parent pre/STO 校准；不是新 post 校准，演示 b2/gamma2 也不是拟合结果。
- 本次没有重跑可信 post PNG 目录的 B-only kmax 扫描，不能据此断言 kmax 提升。
- make release-test 因工作副本没有 Git 元数据未通过。全仓库 Python 测试为 107 通过、6 失败、14 错误；20 个未通过项均引用未迁入的 nonlocal/production 文件。
- 已有 post provisional 与 halo PNG catalogue 限制保持不变。

## 入口与产物

- 使用说明：marisa_b_core/docs/POST_RECON_TREE_PNG_IR_NATIVE.md。
- 完整模型：marisa_b_core/python/marisa_b/post_recon_tree_png_ir_native.py。
- 批量投影：marisa_b_core/scripts/production/generate_post_tree_png_ir.py。
- 实例：audit_fine/ 下三方法的 *_config.json、*_exact_shell_nodes.npz、*_shell_templates.npz。
- 图：audit_fine/post_tree_ir_current_p.pdf；没有生成 PNG。
- 原文件备份：original_source/；源码/输入/输出哈希：POST_TREE_IR_CHANGE_MANIFEST.json。
