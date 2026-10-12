"""完整 post tree IR 的 native 接口。

模块大纲：批量实际方向节点；显式 q 网格/噪声；逐项结果；保留原分母的 shell 投影。
只支持 f_rec=eta*f 的参数路径；固定非零 f_rec 目录不能冒充 eta=1。
"""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .tree_png_ir_native import NATIVE_TERMS
from .tree_png_ir_resummation import TreeComponents, StochasticBases
from .post_recon_tree_png_ir import (
    AdaptiveReconstructionConfig, PostReconIrSettings, PostReconTreeIrResult,
    DisplacementVarianceExpansion, LegDampingExpansion, DampingMixing,
    StochasticDampingMixing, ADAPTIVE_LINEAR_TERMS, ADAPTIVE_QUADRATIC_TERMS,
    _components_from_post_terms,
)

POST_NATIVE_TERMS = NATIVE_TERMS + ADAPTIVE_LINEAR_TERMS + ADAPTIVE_QUADRATIC_TERMS


@dataclass(frozen=True)
class NativePostTreeBatch:
    values: np.ndarray
    settings: PostReconIrSettings
    adaptive: AdaptiveReconstructionConfig
    mesh_damping: str
    includes_shift_noise: bool

    @property
    def source(self): return self.values[:, :27]
    @property
    def no_wiggle(self): return self.values[:, 27:54]
    @property
    def fixed_order(self): return self.values[:, 54:81]
    @property
    def variance(self): return self.values[:, 81:90].reshape(-1, 3, 3)
    @property
    def damping(self): return self.values[:, 90:99].reshape(-1, 3, 3)
    @property
    def mixing(self): return self.values[:, 99:102]
    @property
    def stochastic_mixing(self): return self.values[:, 102:105]
    @property
    def deterministic(self): return self.values[:, 105:108]

    def result(self, index=0):
        terms = dict(zip(POST_NATIVE_TERMS, self.source[index]))
        nw = dict(zip(POST_NATIVE_TERMS, self.no_wiggle[index]))
        sig = self.variance[index]
        damp = self.damping[index]
        return PostReconTreeIrResult(
            deterministic=TreeComponents(*self.deterministic[index]),
            source_only=_components_from_post_terms(terms),
            no_wiggle_source=_components_from_post_terms(nw),
            damping_mixing=DampingMixing(*self.mixing[index]),
            source_stochastic=StochasticBases(*self.source[index, 17:21]),
            stochastic_mixing=StochasticDampingMixing(*self.stochastic_mixing[index]),
            source_operator_terms=terms,
            displacement=DisplacementVarianceExpansion(
                *(tuple(x) for x in sig), self.mesh_damping == "cartesian",
                self.settings.q_min, self.settings.k_ir, self.includes_shift_noise),
            damping=LegDampingExpansion(*(tuple(x) for x in damp)),
            adaptive=self.adaptive, mesh_damping=self.mesh_damping,
        )


class NativePostTreeIR:
    def __init__(self, library: str | Path):
        self.path = Path(library).resolve(strict=True)
        self.lib = ctypes.CDLL(str(self.path))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.lib.marisa_post_tree_ir_batch.argtypes = [
            ctypes.c_size_t, ptr, ptr, ptr, ptr, ptr, ctypes.c_size_t,
            ptr, ptr, ptr, ptr, ptr, ptr, iptr, ptr,
        ]
        self.lib.marisa_post_tree_ir_batch.restype = ctypes.c_int
        self.lib.marisa_post_tree_ir_last_error.restype = ctypes.c_char_p

    def evaluate(self, vectors, power, no_wiggle, transfer, bias, reconstruction,
                 q, qpower, qnw, qm, settings: PostReconIrSettings,
                 adaptive: AdaptiveReconstructionConfig | None = None, *, shift_noise_power=None):
        v = np.ascontiguousarray(vectors, dtype=np.float64)
        if v.ndim != 3 or v.shape[1:] != (3, 3) or not len(v):
            raise ValueError("vectors must have shape (N,3,3), N>0")
        legs = [np.ascontiguousarray(x, dtype=np.float64) for x in (power, no_wiggle, transfer)]
        if any(x.shape != (len(v), 3) for x in legs):
            raise ValueError("leg spectra must have shape (N,3)")
        grids = [np.ascontiguousarray(x, dtype=np.float64) for x in (q, qpower, qnw, qm)]
        if any(x.ndim != 1 or x.shape != grids[0].shape for x in grids) or len(grids[0]) < 3:
            raise ValueError("IR spectra must be equal-length 1D arrays with >=3 nodes")
        noise = shift_noise_power(grids[0]) if callable(shift_noise_power) else shift_noise_power
        noise = np.zeros_like(grids[0]) if noise is None else np.ascontiguousarray(noise, dtype=np.float64)
        if noise.shape != grids[0].shape:
            raise ValueError("shift noise must match the IR grid")
        a = adaptive or AdaptiveReconstructionConfig()
        r, s = reconstruction, settings
        b = np.array([bias.b1, bias.b2, bias.bk2, bias.bphi, bias.bphidelta, bias.bphi2])
        cfg = np.array([r.smoothing_radius, r.bias_recon, r.cell_size,
                        s.r_bao, s.k_ir, s.q_min, a.bphi_recon, a.eta], dtype=np.float64)
        quad = np.array([r.cic_window_power, {"auto": 0, "isotropic": 1, "cartesian": 2}[s.mesh_damping],
                         s.angular_mu_order, s.angular_phi_order], dtype=np.int32)
        out = np.empty((len(v), 108), dtype=np.float64)
        status = self.lib.marisa_post_tree_ir_batch(len(v), v, *legs, b, len(grids[0]),
                                                  *grids, noise, cfg, quad, out)
        if status:
            raise ValueError(self.lib.marisa_post_tree_ir_last_error().decode())
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite post IR output")
        mesh = s.mesh_damping
        if mesh == "auto":
            mesh = "cartesian" if r.cell_size > 0 and r.cic_window_power > 0 else "isotropic"
        return NativePostTreeBatch(out, s, a, mesh, shift_noise_power is not None)


def project_post_tree_nodes(batch: NativePostTreeBatch, weights, bin_ids, *, number_of_bins,
                            weights_include_original_denominator: bool):
    """先逐实际方向节点 IR，再按原 shell 分母投影；绝不按保留下来的节点重新归一化。

    weights 必须已经除以完整 shell pair 数（包括被排除的零腿/闭合零模）。
    CIC 开启时，调用方必须提供完整的绝对方向积分节点，不能使用固定平面代表三角形。
    不返回平均 Sigma 或 D，因为平均阻尼不能用于替代平均后的 B。
    """
    if weights_include_original_denominator is not True:
        raise ValueError("explicit original shell denominator acknowledgement required")
    w = np.asarray(weights, dtype=float)
    ids = np.asarray(bin_ids)
    if w.shape != (len(batch.values),) or ids.shape != w.shape or not np.isfinite(w).all():
        raise ValueError("invalid projection weights or bin ids")
    if number_of_bins < 1 or int(number_of_bins) != number_of_bins:
        raise ValueError("number_of_bins must be a positive integer")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= number_of_bins):
        raise ValueError("bin ids must be integers inside the declared range")
    out = {}
    for name in ("source", "no_wiggle", "fixed_order", "mixing", "stochastic_mixing", "deterministic"):
        data = getattr(batch, name)
        value = np.zeros((int(number_of_bins), data.shape[1]))
        np.add.at(value, ids, w[:, None] * data)
        out[name] = value
    out["weight_sum"] = np.bincount(ids, weights=w, minlength=int(number_of_bins))
    return out
