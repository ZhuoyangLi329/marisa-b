# 新仓库约定

1. 原仓库 `/pscratch/sd/l/lzy/reconstruction-png` 保持只读；迁移使用复制。
2. 主题分部分：`01_reconstruction_methods` 是第一部分。方法标识固定为
   `standard_gra`、`marisa_gra`、`sto_recon`、`sto_standard`、`sto_marisa`。
3. 共用数值流程只保留一份；示范入口简短，源文件映射与改动写入迁移清单。
4. 新代码使用中文注释与模块大纲；优先复用已经验证的实现。
5. 数据读取路径可指向共享 Quijote 目录；示例输出只写本部分 `outputs/`。
6. 不复制批量历史 manifest、结果、日志或废弃 wrapper；第一部分只留小型示例配置。
7. 登录节点严格限制 8 核；缺包先确认现有环境，不自行安装。
8. 图只保留 PDF；不批量删除用户文件。
