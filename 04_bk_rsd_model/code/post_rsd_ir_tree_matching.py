"""模块大纲：保留原完整hard tree的IR matching；不是新增物理参数或noise修复。

Hmatched=Hexact+(HIRstrict-Hstrict,Sigma0)，只重求和single-wiggle的阻尼差。
Sigma=0时返回原hard tree；高阶wiggle余项保持原tree值而未额外resum。
此模块只定义显式数值候选；不选择production策略或放行MCMC。
"""
import numpy as np
from post_rsd_scipy_harmonic_compat import install_legacy_sph_harm
install_legacy_sph_harm()
from post_rsd_ir_harmonics import HarmonicDisplacement, reconstruction_signature
from post_rsd_fixedrec import pbs_bias


def tree_matched_signal(exact_tree, strict_undamped, strict_damped):
    values = [np.asarray(x, dtype=float) for x in (exact_tree, strict_undamped, strict_damped)]
    if not values[0].size or any(x.shape != values[0].shape or not np.isfinite(x).all() for x in values):
        raise ValueError('same finite hard-tree/strict-undamped/strict-damped coefficient geometry required')
    return values[0]+(values[2]-values[1])


class IdentityDamping(HarmonicDisplacement):
    """明确的Sigma=0数值对照，重构仍存在；不把它称作实测长位移。"""
    def damping(self, vectors, **kwargs):
        v = np.asarray(vectors, dtype=float)
        if v.shape[-1] != 3 or not np.isfinite(v).all():
            raise ValueError('finite Fourier vectors required')
        result = np.zeros(v.shape[:-1]+(3,)); result[..., 0] = 1.
        return result


def identity_damping_context(*, b1, p_halo, f_growth, reconstruction):
    if not all(np.isscalar(x) and np.isfinite(x) for x in (b1, p_halo, f_growth)):
        raise ValueError('explicit finite physical parameters required')
    parameters = dict(b1=float(b1), bphi=pbs_bias(b1, p_halo=p_halo).bphi,
        f_growth=float(f_growth), los=[0., 0., 1.],
        reconstruction_signature=reconstruction_signature(reconstruction),
        numerical_Sigma_zero_control=True, measured_long_covariance=False)
    return IdentityDamping(np.zeros((1, 3, 3, 3)), 0, 110., parameters)
