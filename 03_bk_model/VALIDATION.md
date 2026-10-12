# 第三部分迁移核验

- MARISA-B source、python、production scripts、configs、schemas、provenance、tests 和 docs 已复制到 marisa_b_core。
- DM tree/SPT-one-loop 的 six 个外部脚本、报告、model audit JSON 和四份 PDF 已复制到 dm。
- halo finite-tree/EFT-v2 的 five 个外部脚本、报告、audit JSON 和 eight 份 PDF 已复制到 halo。
- 所有迁移 Python 文件通过 py_compile；C++ 源码未修改。
- 新目录没有 build 二进制、object 文件、PNG、Python cache、原始模拟数据或大批量日志。
- 活跃迁移文件不再包含旧 reconstruction-png 路径或 old_doc_codes 依赖。
- 原仓库只读检查保持不变；本轮只执行复制和新目录整理。

科学边界：DM 的 z=0.5 R=10 和 z=1 R=15 finite G/L/Q one-loop 结果可作为已审计模型结果；halo 的 production 表述必须保持为 Gaussian halo EFT-v2 one-loop 加 finite halo-tree PNG 与已实现 matter-PNG uplift，不能升级为完整 halo local-PNG EFT one-loop。post halo 的数值收敛和非零 PNG reconstruction 输入仍是 provisional。

没有在本轮重新构建 native binary，也没有提交生产作业；构建时应在 marisa_b_core 中按其 Makefile 和外部依赖说明执行。
