# `post_recon_tree_png_ir.py` API 与运行说明

## 文件关系

```text
python/marisa_b/tree_png_ir_resummation.py
    └── native-compatible finite-PNG tree source oracle

python/marisa_b/post_recon_tree_png_ir.py
    ├── adaptive post source bases
    ├── reconstructed displacement variance
    ├── CIC-aware angular integration
    ├── damping exponent finite-PNG mixing
    └── strict O(Pw) final G/L/Q
```

## 安装与测试

```bash
PYTHONPATH=python \
python -m pytest tests/python/test_post_recon_tree_png_ir.py -q
```

完整基础与 post tests：

```bash
PYTHONPATH=python python -m pytest \
    tests/python/test_tree_png_ir_resummation.py \
    tests/python/test_post_recon_tree_png_ir.py \
    -q
```

## 主要配置

```python
AdaptiveReconstructionConfig(
    bphi_recon=0.0,
    fnl_recon_ratio=0.0,
)
```

- `fnl_recon_ratio=0`: fixed denominator。
- `fnl_recon_ratio=1`: `fNL_rec=fNL`。

```python
PostReconIrSettings(
    r_bao=105.0,
    k_ir=0.15,
    q_min=2*np.pi/Lbox,
    mesh_damping="auto",
    angular_mu_order=32,
    angular_phi_order=64,
)
```

`q_min` 必须严格为正。

## 结果字段

```python
result.deterministic       # 完整 G/L/Q，包括 exponent mixing
result.source_only         # 只使用 D0 阻尼 source，不含 exponent fNL response
result.no_wiggle_source    # no-wiggle post source
result.damping_mixing      # G->L, L->Q, G->Q
result.displacement        # Sigma_f0/f1/f2 per leg
result.damping             # D0/d1/d2 per leg
result.source_operator_terms
result.source_stochastic
result.stochastic_mixing
```

## Fixed denominator

```python
result = ir_resummed_post_recon_finite_png_tree(
    ...,
    adaptive=AdaptiveReconstructionConfig(
        bphi_recon=0.0,
        fnl_recon_ratio=0.0,
    ),
)
```

即使 `fnl_recon_ratio=0`，若 `bias.bphi != 0`，通常仍有

```python
result.displacement.sigma_f1 != 0
result.displacement.sigma_f2 != 0
```

因为 reconstruction shift numerator 来自 scale-dependent halo field。

## Matched adaptive

```python
adaptive = AdaptiveReconstructionConfig(
    bphi_recon=bias.bphi,
    fnl_recon_ratio=1.0,
)
reconstruction = ReconstructionConfig(
    smoothing_radius=R,
    bias_recon=bias.b1,
)
```

解析上应得到

```python
Sigma_f1 = 0
Sigma_f2 = 0
```

这是必做回归测试。

## CIC

打开 CIC：

```python
ReconstructionConfig(
    smoothing_radius=R,
    bias_recon=brec,
    cell_size=Delta,
    cic_window_power=4,
)
```

`mesh_damping="auto"` 会选择 `cartesian`。计算成本近似：

```text
Nq * Nmu * Nphi * 3 legs
```

生产前应做 24×48、32×64、48×96 convergence scan。

## Shift noise

```python
result = ir_resummed_post_recon_finite_png_tree(
    ...,
    shift_noise_power=P_epsilon_q,
)
```

默认 `None`。输入必须非负，可为与 q 同长度的数组或 callable。

## Direct finite-f diagnostic

```python
sigma_exact = post_recon_displacement_variance_at_fnl(
    ...,
    fnl=50.0,
)
```

它使用 exact adaptive denominator，不截断于 f²，适合检查二次 Taylor 模型在给定 fNL 范围内是否足够。

## 注意

- 该模块是 tree-only；不要加入 NLO matching factor。
- 不要先 shell average 再调用 CIC-aware damping。
- stochastic mixing 只有在显式 common-alpha3 closure 下才能通过
  `stochastic_if_common_alpha3()` 合并。
- native 移植前保持返回字段分离，便于科学审计。
