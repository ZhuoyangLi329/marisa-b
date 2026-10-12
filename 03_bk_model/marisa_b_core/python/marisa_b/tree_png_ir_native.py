"""Native tree IR 批量入口。

模块大纲：显式加载独立 tree-only 库；验证数组合同；保留全部原子分量。
谱和转移函数由调用者在实际三角形上采样，投影必须在本函数之后进行。
"""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np
from .tree_png_ir_resummation import ALL_TERMS, HaloTreeBias, ReconstructionConfig

NATIVE_TERMS = ALL_TERMS + ("Btree_gaussian_f2", "Btree_gaussian_b2", "Btree_gaussian_bK2")
SCHEMES = {"fixed_order": 0, "strict_lo": 1, "factorized_closure": 2}

class NativeTreeIR:
    def __init__(self, library: str | Path):
        self.path = Path(library).resolve(strict=True)
        self.lib = ctypes.CDLL(str(self.path))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.marisa_tree_ir_batch.argtypes = [
            ctypes.c_size_t, ptr, ptr, ptr, ptr, ptr, ctypes.c_double,
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_int, ptr,
        ]
        self.lib.marisa_tree_ir_batch.restype = ctypes.c_int
        self.lib.marisa_tree_ir_last_error.restype = ctypes.c_char_p

    def evaluate(self, vectors, power, no_wiggle, transfer, bias: HaloTreeBias,
                 *, sigma2: float, scheme: str = "strict_lo",
                 reconstruction: ReconstructionConfig | None = None):
        v = np.ascontiguousarray(vectors, dtype=np.float64)
        if v.ndim != 3 or v.shape[1:] != (3, 3) or not len(v):
            raise ValueError("vectors must have shape (N,3,3), N>0")
        legs = [np.ascontiguousarray(x, dtype=np.float64) for x in (power, no_wiggle, transfer)]
        if any(x.shape != (len(v), 3) for x in legs):
            raise ValueError("spectra must have shape (N,3)")
        if scheme not in SCHEMES:
            raise ValueError(f"unknown tree IR scheme {scheme}")
        b = np.asarray([bias.b1,bias.b2,bias.bk2,bias.bphi,bias.bphidelta,bias.bphi2], dtype=np.float64)
        r = reconstruction
        result = np.empty((len(v),len(NATIVE_TERMS)), dtype=np.float64)
        code = self.lib.marisa_tree_ir_batch(
            len(v), v, *legs, b, sigma2, SCHEMES[scheme], int(r is not None),
            r.smoothing_radius if r else 0., r.bias_recon if r else 1.,
            r.cell_size if r else 0., r.cic_window_power if r else 0, result,
        )
        if code:
            raise ValueError(self.lib.marisa_tree_ir_last_error().decode())
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("native tree IR returned non-finite components")
        return result
