# 组合方法

先运行 STO 得到 `post_sto_halo.npz`，再运行
`run_sto_standard.py` 或 `run_sto_marisa.py`。
组合入口固定 `input_mode=sto-weighted`，直接复用共用 halo pipeline。
本目录不再复制 standard 或 Marisa 的数值实现。
