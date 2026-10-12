"""PNG-aware reconstruction tools for the local marisa-recon checkout.

代码大纲
1. `compute_png_alpha_table`：
   用 desilike 的 local PNG `method="prim"` 约定计算
   `alpha(k)=sqrt(P_phi(k) / P_delta(k,z))` 表格。
2. `MarisaPlaneParallelFFTReconstruction`：
   复用 `jaxrecon.zeldovich.BaseReconstruction` 的读位移、读重构位置等接口，
   只替换 PlaneParallelFFTReconstruction 中求解位移场的核心步骤。
   原始 jaxrecon 做法是 `delta(k) / b_fid`；
   这里改为 `delta(k) / b_eff(k)`，其中
   `b_eff(k)=b1 + 2 * delta_c * (b1 - p) * fNL * alpha(k)`。
3. `describe_png_bias_model`：
   把本次使用的 PNG bias 参数整理成 JSON metadata，方便后续审计。

输入和输出约定
- k 的单位是 h/Mpc，功率谱单位沿用 cosmoprimo/desilike 默认约定。
- `fnl_loc=0` 时 `b_eff(k)=b1`，应退化回标准标量 bias reconstruction。
- 本文件只改 `/pscratch/sd/l/lzy/marisa-recon` 这份本地拷贝；
  原始 `/pscratch/sd/l/lzy/desi-clustering` 不被修改。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np

from jaxrecon.zeldovich import BaseReconstruction


DELTA_C_DEFAULT = 1.686


@dataclass(frozen=True)
class PNGAlphaMetadata:
    """记录 `alpha(k)` 表格的来源和数值范围。

    参数
    ----------
    cosmology
        cosmology 标识。目前 production 使用 Quijote fiducial cosmology。
    redshift
        计算线性 matter power spectrum 的红移。
    method
        desilike PNG 模型里的 alpha 计算方法。这里固定为 `prim`。
    kmin, kmax, nk
        `alpha(k)` 插值表格的 k 范围和采样点数。
    alpha_min, alpha_max
        表格中 alpha 的最小值和最大值，用来快速检查量级是否异常。
    """

    cosmology: str
    redshift: float
    method: str
    kmin: float
    kmax: float
    nk: int
    alpha_min: float
    alpha_max: float


QUIJOTE_COSMOLOGY_PARAMS: dict[str, float] = {
    "h": 0.6711,
    "Omega_m": 0.3175,
    "Omega_b": 0.049,
    "sigma8": 0.834,
    "n_s": 0.9624,
}


def describe_png_bias_model(
    *,
    b1: float,
    fnl_loc: float,
    p: float,
    delta_c: float,
    alpha_metadata: PNGAlphaMetadata | dict[str, Any],
) -> dict[str, Any]:
    """整理本次 marisa reconstruction 的 PNG bias 配置。

    参数
    ----------
    b1
        Gaussian linear bias，也就是旧 reconstruction 中的标量 `b_fid`。
    fnl_loc
        local PNG 参数。fiducial 样本使用 0，LC_p50 样本使用 50。
    p
        desilike `mode="b-p"` 中的 p 参数，默认取 1。
    delta_c
        球坍缩阈值，desilike local PNG 默认使用 1.686。
    alpha_metadata
        `compute_png_alpha_table` 返回的 metadata。

    返回
    ----
    dict
        可直接写入 JSON 的 provenance 字段。
    """

    if isinstance(alpha_metadata, PNGAlphaMetadata):
        alpha_metadata = asdict(alpha_metadata)
    bphi = 2.0 * float(delta_c) * (float(b1) - float(p))
    return {
        "name": "desilike_local_png_prim_b_minus_p",
        "formula": "b_eff(k) = b1 + 2 * delta_c * (b1 - p) * fNL * alpha(k)",
        "desilike_reference": "PNGTracerPowerSpectrumMultipoles(method='prim', mode='b-p')",
        "b1": float(b1),
        "fnl_loc": float(fnl_loc),
        "p": float(p),
        "delta_c": float(delta_c),
        "bphi": float(bphi),
        "bfnl_loc": float(bphi * float(fnl_loc)),
        "alpha": dict(alpha_metadata),
    }


def compute_png_alpha_table(
    *,
    kmin: float = 1.0e-4,
    kmax: float = 10.0,
    nk: int = 2048,
    redshift: float = 0.5,
    cosmology: str = "quijote",
) -> tuple[np.ndarray, np.ndarray, PNGAlphaMetadata]:
    """计算 desilike local PNG `method="prim"` 使用的 `alpha(k)` 表格。

    参数
    ----------
    kmin, kmax
        插值表格的最小和最大波数，单位 h/Mpc。
    nk
        几何采样点数。production mesh 的 k 范围很窄，但表格故意取得更宽，
        避免未来调整 mesh 或 cellsize 时越界。
    redshift
        线性 matter power spectrum 的红移。当前 Quijote halo 样本为 z=0.5。
    cosmology
        cosmology 名称。目前只支持 `quijote`，对应 task.md 里的 Quijote fiducial 参数。

    返回
    ----
    tuple
        `(k, alpha, metadata)`：
        `k` 是 h/Mpc 的一维升序数组；
        `alpha` 是同长度数组；
        `metadata` 记录表格来源和范围。
    """

    if str(cosmology).lower() != "quijote":
        raise ValueError(f"目前只支持 cosmology='quijote'，收到 {cosmology!r}")
    if not (float(kmin) > 0.0 and float(kmax) > float(kmin)):
        raise ValueError(f"Bad k range for PNG alpha table: kmin={kmin}, kmax={kmax}")
    if int(nk) < 8:
        raise ValueError(f"PNG alpha table needs at least 8 samples, got nk={nk}")

    from cosmoprimo import Cosmology

    kin = np.geomspace(float(kmin), float(kmax), int(nk)).astype("f8")
    cosmo = Cosmology(engine="class", **QUIJOTE_COSMOLOGY_PARAMS)
    pk_dd = cosmo.get_fourier().pk_interpolator(of="delta_cb").to_1d(z=float(redshift))(kin)
    pk_prim = cosmo.get_primordial(mode="scalar").pk_interpolator()(kin)

    # 这里逐字采用 desilike local PNG `method="prim"` 的 convention：
    # pphi_prim = 9 / 25 * 2*pi^2/k^3 * P_R(k) / h^3
    # alpha = sqrt(P_phi / P_delta)
    pphi_prim = 9.0 / 25.0 * 2.0 * np.pi**2 / kin**3 * pk_prim / cosmo.h**3
    alpha = 1.0 / np.sqrt(pk_dd / pphi_prim)
    alpha = np.asarray(alpha, dtype="f8")

    if not np.all(np.isfinite(alpha)):
        bad = int(np.size(alpha) - np.count_nonzero(np.isfinite(alpha)))
        raise FloatingPointError(f"PNG alpha table contains {bad} non-finite values")
    if not np.all(alpha > 0.0):
        raise FloatingPointError("PNG alpha table must be strictly positive")

    meta = PNGAlphaMetadata(
        cosmology="quijote",
        redshift=float(redshift),
        method="desilike_prim",
        kmin=float(kin[0]),
        kmax=float(kin[-1]),
        nk=int(kin.size),
        alpha_min=float(np.min(alpha)),
        alpha_max=float(np.max(alpha)),
    )
    return kin, alpha, meta


@jax.tree_util.register_pytree_node_class
class MarisaPlaneParallelFFTReconstruction(BaseReconstruction):
    """Plane-parallel FFT reconstruction with k-dependent local PNG bias.

    这个类和 `jaxrecon.zeldovich.PlaneParallelFFTReconstruction` 的差别只在 `_run`：
    标准版本在傅里叶空间使用 `mesh_delta / bias`；
    本版本使用 `mesh_delta / b_eff(k)`。

    重要参数通过 `BaseReconstruction.__init__` 传入：
    `growth_rate`、`bias`、`los` 保持 jaxrecon 原接口；
    下面这些额外参数会被继续传给 `_run`：

    参数
    ----------
    png_alpha_k
        `alpha(k)` 插值表格的 k 坐标，单位 h/Mpc。
    png_alpha_values
        与 `png_alpha_k` 同长度的 alpha 数值。
    fnl_loc
        local PNG 参数。fiducial 为 0，LC_p50 为 50。
    png_p
        desilike `mode="b-p"` 中的 p 参数，默认 1。
    png_delta_c
        球坍缩阈值，默认 1.686。
    bias_floor
        防止极端参数使 `b_eff(k)` 接近 0 后造成数值发散的下限。

    输出
    ----
    self.mesh_psi
        三个方向的 displacement mesh，后续由 inherited `read_shifts` /
        `read_shifted_positions` 读取到 data/random 粒子位置上。
    """

    def _run(
        self,
        mesh_delta,
        *,
        png_alpha_k,
        png_alpha_values,
        fnl_loc: float = 0.0,
        png_p: float = 1.0,
        png_delta_c: float = DELTA_C_DEFAULT,
        bias_floor: float = 1.0e-3,
    ) -> None:
        """根据 PNG-aware bias 解 displacement mesh。

        参数
        ----------
        mesh_delta
            smoothed halo overdensity mesh。类型来自 jaxpower，必须支持 `r2c/c2r`。
        png_alpha_k, png_alpha_values
            host 或 device 上的一维数组，会被转成 JAX array 后在 k 空间插值。
        fnl_loc, png_p, png_delta_c, bias_floor
            见类 docstring。
        """

        if self.los is None:
            raise ValueError("Provide a global line-of-sight")

        real_dtype = mesh_delta.value.dtype
        b1 = jnp.asarray(self.bias(), dtype=real_dtype)
        growth_rate = jnp.asarray(self.growth_rate(), dtype=real_dtype)
        alpha_k = jnp.asarray(png_alpha_k, dtype=real_dtype)
        alpha_values = jnp.asarray(png_alpha_values, dtype=real_dtype)
        fnl_value = jnp.asarray(fnl_loc, dtype=b1.dtype)
        p_value = jnp.asarray(png_p, dtype=b1.dtype)
        delta_c_value = jnp.asarray(png_delta_c, dtype=b1.dtype)
        bias_floor_value = jnp.asarray(bias_floor, dtype=b1.dtype)

        kvec = self.mattrs.kcoords(sparse=True)
        ivec = self.mattrs.kcoords(kind="index", sparse=True)
        k2_raw = sum(kk**2 for kk in kvec)
        zero_mode = k2_raw == 0.0
        k2 = jnp.where(zero_mode, 1.0, k2_raw)
        k_abs = jnp.sqrt(k2)

        # 在 log10(k) 上插值，和 desilike 中 PNG alpha 插值的数值习惯保持一致。
        alpha = jnp.interp(jnp.log10(k_abs), jnp.log10(alpha_k), alpha_values)
        bphi = 2.0 * delta_c_value * (b1 - p_value)
        b_eff = b1 + bphi * fnl_value * alpha
        b_eff = jnp.where(zero_mode, b1, b_eff)

        # 正常 Quijote fNL=50 参数下不会触发 floor；保留这个保护是为了避免
        # 未来做极端参数扫描时一两个 k-bin 把 FFT displacement 变成 NaN。
        safe_sign = jnp.where(b_eff < 0.0, -1.0, 1.0)
        b_eff = jnp.where(jnp.abs(b_eff) < bias_floor_value, safe_sign * bias_floor_value, b_eff)

        mesh_delta = mesh_delta.r2c() / b_eff
        beta = growth_rate / b_eff

        mesh_psis = []
        for axis in range(self.mattrs.ndim):
            mu2 = sum(kk * ll for kk, ll in zip(kvec, self.los))**2 / k2
            mesh_psi = 1j * kvec[axis] / k2 / (1.0 + beta * mu2) * mesh_delta
            if self.mattrs.is_hermitian:
                mesh_psi *= ivec[axis] != self.mattrs.meshsize[axis] // 2
            mesh_psis.append(mesh_psi.c2r())
            del mesh_psi
        self.mesh_psi = mesh_psis
