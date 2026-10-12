#!/usr/bin/env python3
"""模块大纲：只读旧资产；精确换算重构源核；保持数据、先验与 stochastic 闭合。

tree-level reconstruction 源项和三个 adaptive raw basis 都对 1/b_rec 线性。
因此可精确重标度已经投影的这些项，不必改变旧 shell 投影；外层 adaptive
系数另用新的 bphi_rec/b_rec。旧目录与缓存均不修改。
"""
from __future__ import annotations
import inspect
import numpy as np
import run_joint_pb_matched250 as original

PARTS = ("gaussian", "linear_coefficient", "quadratic_coefficient")
BNAMES = ("b2", "gamma2", "Ashot_residual", "Bshot_residual")
BREC = 2.2280546353635335
PREC = 1.2822


class CorrectedModel:
    def __init__(self, method):
        self.method = method
        self.low, self.scan, self.runtime = original.configure_models()
        cfg = original.CASES[method]
        self.p = cfg["p"]
        self.low.bphi_from_b1 = lambda b: 2*self.low.DELTA_C*(float(b)-self.p)
        if method == "sto_marisa":
            self.low.B_REC, self.low.P_REC = BREC, PREC
            self.low.BPHI_REC = 2*self.low.DELTA_C*(BREC-PREC)
        self.legacy = self.scan.DirectFiniteScanB000Model(self.runtime, cfg["method"])
        self.base_context = self.legacy.base_context
        self.names = self.legacy.names
        self.indices = self.scan.selected_indices(self.runtime.geometry, .14)
        self.assets = self.runtime.high_runtime.modules["assets"]
        self.old_brec = float(self.assets.png.b_rec_h)
        self.ratio = self.old_brec/BREC if method == "sto_marisa" else 1.
        if method == "sto_marisa":
            compiled = self.assets.post_templates.tied._compiled["diagram:tree"]
            assert set(compiled) == {"b1^2*b2^1", "b1^2*gamma2^1", "b1^3", "b1^4"}
            self.gaussian_shift_unit = np.asarray(compiled["b1^4"])
            assert float(self.runtime.marisa_finite["b_rec"]) == self.old_brec
            # 当前 likelihood 只保留 leading residual；这些 monomials 没有 shift。
            for name, keys in (("residual_gaussian", {"b1^2"}),
                               ("residual_linear", {"b1^1"}),
                               ("residual_quadratic", {"1"})):
                assert set(getattr(self.assets.png, name)["tied"]) == keys
            inner = inspect.getclosurevars(self.base_context.evaluate_model).nonlocals["original"]
            self.nbar = float(inspect.getclosurevars(inner).nonlocals["method"].nbar)
        else:
            self.nbar = float(self.runtime.low_runtime.pre_closure["nbar"])

    def parts(self, mapping, *, corrected=True):
        values = {k: np.array(v, copy=True) for k, v in self.legacy.parts(mapping).items()}
        if self.method == "pre" or not corrected:
            return values
        b = float(mapping["b1"])
        bp = self.low.bphi_from_b1(b)
        factor = self.ratio-1.
        tree = self.assets.png.tree
        values["gaussian"] += factor*b**4*self.gaussian_shift_unit
        values["linear_coefficient"] += factor*(
            b**3*bp*tree["dBdfNL_local_bphi_reconstruction"] + values["adaptive_linear"])
        values["quadratic_coefficient"] += factor*(
            b**2*bp**2*tree["Bhalo_tree_fNL2_bphi_sq_reconstruction"]
            + values["adaptive_quadratic"] + values["adaptive_cross"])
        for key in ("adaptive_linear", "adaptive_quadratic", "adaptive_cross"):
            values[key] *= self.ratio
        return values

    def design(self, b):
        mapping = {"b1": float(b), **dict.fromkeys(BNAMES, 0.)}
        def evaluate(mp):
            p = self.parts(mp)
            return np.asarray([p[k] for k in PARTS])
        base = evaluate(mapping)
        return np.asarray([base, *[evaluate({**mapping, key: 1.})-base for key in BNAMES]])

    def __call__(self, f, mapping, indices):
        p = self.parts(mapping)
        return (p[PARTS[0]]+f*p[PARTS[1]]+f*f*p[PARTS[2]])[indices]

    def provenance(self):
        return {"b_rec": BREC if self.method == "sto_marisa" else None,
                "p_rec": PREC if self.method == "sto_marisa" else None,
                "p_halo": self.p, "old_cache_b_rec": self.old_brec,
                "raw_shift_multiplier": self.ratio,
                "correction": "exact analytic 1/b_rec rescaling of G b1^4, PNG L/Q reconstruction, and all three adaptive raw bases",
                "reconstruction_parameter_path": "f_rec=f (inherited legacy theoretical path)",
                "catalogue_f_rec": 100.,
                "scope_warning": "controlled legacy-path comparison; not fixed-nonzero-f_rec inference",
                "stochastic_closure": "legacy_accepted_pre retained; no added shift-noise damping"}
