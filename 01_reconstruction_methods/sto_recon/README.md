# STO reconstruction

`run_halo_sto_variants.py` 是完整现役实现，不是指向 archive 的 wrapper。
阅读顺序：`StoConfig`/`ComponentConfig` → 环境与分量标签 → `solve*` 权重求解 → `run`。
基线为 matter-assisted N+Env，配置在 `../examples/sto_recon.jsonl`。
它只改 halo 权重，输出 `post_sto_halo.npz`；没有另外增加一个 STO 位移算法。
