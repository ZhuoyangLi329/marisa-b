"""模块大纲：严格二次的Gaussian长位移协方差缓存；不是IR-B多项式插值。

d0=U+b1*V，d1=bphi*W；六个Gram核足以重构任意物理b1/bphi。
仅原长位移积分调用五次提取系数，不冻结拟合bias，不增加自由参数。
IR-B仍含exp和有限PNG mixing，绝不据此假定B关于b1为四次多项式。
"""
from dataclasses import dataclass
import copy
import hashlib
import numpy as np
from post_rsd_scipy_harmonic_compat import install_legacy_sph_harm


@dataclass(frozen=True)
class HarmonicCovarianceBasis:
    gram_coefficients: object
    noise_offset: object
    lmax: int
    r_bao: float
    geometry: dict
    build_metadata: dict

    def __post_init__(self):
        values = np.asarray(self.gram_coefficients, dtype=float).copy()
        offset = np.asarray(self.noise_offset, dtype=float).copy()
        if values.ndim != 4 or values.shape[0] != 6 or values.shape[-2:] != (3, 3):
            raise ValueError('six finite harmonic Gram tensors required')
        if offset.shape != values.shape[1:] or not np.isfinite(values).all() or not np.isfinite(offset).all():
            raise ValueError('finite consistent covariance basis required')
        values.flags.writeable = False; offset.flags.writeable = False
        object.__setattr__(self, 'gram_coefficients', values)
        object.__setattr__(self, 'noise_offset', offset)
        object.__setattr__(self, 'geometry', copy.deepcopy(self.geometry))

    @classmethod
    def from_long_modes(cls, long_modes, *, f_growth, los, reconstruction=None,
                        r_bao=110., lmax=10, shift_noise=None):
        compatibility = install_legacy_sph_harm()
        from post_rsd_ir_harmonics import HarmonicDisplacement
        kwargs = dict(f_growth=f_growth, los=los, reconstruction=reconstruction, r_bao=r_bao, lmax=lmax)
        build = lambda b, phi: HarmonicDisplacement.from_long_modes(long_modes, b1=b, bphi=phi, **kwargs)
        zero = build(0., 0.); plus = build(1., 0.); minus = build(-1., 0.)
        phi_zero = build(0., 1.); phi_plus = build(1., 1.)
        c0 = np.asarray(zero.coefficients); cp = np.asarray(plus.coefficients); cm = np.asarray(minus.coefficients)
        cphi0 = np.asarray(phi_zero.coefficients); cphi1 = np.asarray(phi_plus.coefficients)
        gram = np.stack((c0[:, 0], (cp[:, 0]-cm[:, 0])/2,
            (cp[:, 0]+cm[:, 0])/2-c0[:, 0], cphi0[:, 1], cphi1[:, 1]-cphi0[:, 1], cphi0[:, 2]))
        offset = np.zeros_like(c0[:, 0]); geometry = dict(zero.physical_parameters)
        noise_hash = None
        if shift_noise is not None:
            noisy = HarmonicDisplacement.from_long_modes(long_modes, b1=0., bphi=0.,
                shift_noise=shift_noise, **kwargs)
            offset = np.asarray(noisy.coefficients)[:, 0]-c0[:, 0]
            geometry['shift_noise_explicit'] = True
            noise_hash = hashlib.sha256(np.asarray(shift_noise, dtype='f8').tobytes()).hexdigest()
        geometry.pop('b1'); geometry.pop('bphi')
        metadata = dict(compatibility=compatibility, harmonic_build_calls=5+(shift_noise is not None),
            coefficient_identity='A0=UU+b1*UV+b1^2*VV+noise; A1=bphi*(UW+b1*VW); A2=bphi^2*WW',
            algebraic_probes_are_not_prior_changes=True, signal_B_polynomial_interpolation=False,
            calibrated_physical_p_required_by_future_likelihood=True, stochastic_closure_validated=False,
            shift_noise_sha256=noise_hash, MCMC_released=False, IR_model_accepted=False)
        return cls(gram, offset, lmax, float(r_bao), geometry, metadata)

    def coefficients_at(self, *, b1, bphi):
        if not np.isfinite([b1, bphi]).all():
            raise ValueError('explicit finite physical bias parameters required')
        uu, uv, vv, uw, vw, ww = self.gram_coefficients
        return np.stack((uu+b1*uv+b1*b1*vv+self.noise_offset,
            bphi*(uw+b1*vw), bphi*bphi*ww), axis=1)

    def harmonics_at(self, *, b1, bphi):
        install_legacy_sph_harm()
        from post_rsd_ir_harmonics import HarmonicDisplacement
        geometry = dict(self.geometry, b1=float(b1), bphi=float(bphi))
        return HarmonicDisplacement(self.coefficients_at(b1=b1, bphi=bphi),
            self.lmax, self.r_bao, geometry)
