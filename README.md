# Reconstruction and local PNG: research review snapshot

本仓库当前文件树来自 NERSC 任务仓库 `/pscratch/sd/l/lzy/reconstuction-png-v2`。
研究问题是重建与 STO 加权能否改善 local PNG 的 fNL 测量，并保持可解释的响应与恢复。
目录名中的 `reconstuction` 是原任务仓库拼写；研究内容是 reconstruction。

**GPT Pro 请从 [审阅入口](review/START_HERE.md) 开始，再读 [现行结果](review/CURRENT_RESULTS.md) 和 [审阅任务](review/REVIEW_PROMPT.md)。**

| 部分 | 内容 |
|---|---|
| [01_reconstruction_methods](01_reconstruction_methods/README.md) | standard、Marisa、STO 及组合方法；共用 catalog/random/P/B 测量 |
| [02_reconstruction_on_pk](02_reconstruction_on_pk/README.md) | P0-only、响应 p 校准、pre+post 联合与环境 multitracer |
| [03_bk_model](03_bk_model/README.md) | MARISA-B native core；DM/halo 实空间 B、one-loop/EFT、IR 与 P+B |
| [04_bk_rsd_model](04_bk_rsd_model/README.md) | 有向 RSD 核、固定重建操作、噪声/窗口核验与最新 P+B 对照 |

保留核心源码、推导、配置、小型输入、结果/审计摘要和 PDF；大型原始模拟、采样链、运行日志、编译产物与缓存留在 NERSC。
原文件保持字节一致；此快照用于代码与科学审阅，不承诺脱离 NERSC 环境即可重跑所有生产任务。
源文档中的历史状态与绝对路径未被改写，阅读优先级见审阅入口。

来源、文件哈希和选择规则见 [REVIEW_MANIFEST.json](review/REVIEW_MANIFEST.json)。
旧 GitHub 文件树通过一个普通后继提交替换，原提交历史仍可追溯；移除清单位于 `old_doc_codes/deletion_manifests/`。
MARISA-B 核心许可证与第三方归属说明保存在 [LICENSE](LICENSE) 和 [模型归属说明](03_bk_model/marisa_b_core/THIRD_PARTY_NOTICES.md)。
