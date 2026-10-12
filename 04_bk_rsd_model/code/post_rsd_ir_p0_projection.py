"""模块大纲：P0的确定性LO IR源，复用B的tagged D因子与原生产P窗口。

严格O(Pw)、fNL²、Gaussian长位移；不加入loops或未知STO随机项。
源的P/M一同缩放，保持P_phi；零阻尼的L系数与精确tree仍可差O(Pw²)，
必须报告这项matching余项，不能把它伪称IR damping的净效应。
"""
import numpy as np
from post_rsd_scipy_harmonic_compat import install_legacy_sph_harm
install_legacy_sph_harm()
from post_rsd_ir_projection import IRPointFields, TaggedField
from post_rsd_ir_harmonics import HarmonicDisplacement, reconstruction_signature
from post_rsd_fixedrec import pbs_bias
from rsd_b000_fft_projection import D


def ir_p0_modes(vectors, power, power_nw, transfer, b1, f_growth, *, p_halo,
                reconstruction, damping):
    """返回[G,L,Q,Nmode]；不包含Pshot，不改变冻结的重构核。"""
    v = np.asarray(vectors, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('explicit finite original Fourier vectors required')
    if np.any(np.sum(v*v, axis=1) <= 0):
        raise ValueError('original P window contains only nonzero modes')
    spectra = [np.asarray(x, dtype=float) for x in (power, power_nw, transfer)]
    if any(x.shape != (len(v),) or not np.isfinite(x).all() or np.any(x <= 0) for x in spectra):
        raise ValueError('positive finite per-mode physical P, no-wiggle P, and physical M required')
    if not all(np.isscalar(x) and np.isfinite(x) for x in (b1, f_growth, p_halo)):
        raise ValueError('explicit finite physical parameters required')
    fields = IRPointFields(v, *spectra, b1, f_growth, p_halo=p_halo,
                          reconstruction=reconstruction, damping=damping)
    return np.array([fields.values(D, degree)+fields.values(TaggedField(D, True), degree)
                     for degree in range(3)])


def project_p0_ir_window(window, power, power_nw, transfer, b1, f_growth, *, p_halo,
                         reconstruction, harmonics):
    """直接使用原labels/weights/nmodes；不重新造壳层，不做角向平均替换。"""
    if not isinstance(harmonics, HarmonicDisplacement):
        raise TypeError('explicit parameter-tagged original displacement harmonics required')
    expected = dict(b1=float(b1), bphi=pbs_bias(b1, p_halo=p_halo).bphi, f_growth=float(f_growth))
    for name, value in expected.items():
        if not np.isclose(harmonics.physical_parameters[name], value, rtol=2e-14, atol=2e-14):
            raise ValueError('P/B damping parameter context differs: '+name)
    los = np.asarray(harmonics.physical_parameters['los'], dtype=float)
    los /= np.linalg.norm(los)
    if not np.allclose(np.abs(los), [0., 0., 1.], rtol=0., atol=1e-14):
        raise ValueError('original global LOS=z required')
    if harmonics.physical_parameters['reconstruction_signature'] != reconstruction_signature(reconstruction):
        raise ValueError('P/B fixed reconstruction contexts differ')
    integer = np.asarray(window['vectors_integer'])
    labels = np.asarray(window['labels']); weights = np.asarray(window['weights'], dtype=float)
    counts = np.asarray(window['nmodes'], dtype=float)
    if (integer.ndim != 2 or integer.shape[1] != 3 or not len(integer)
            or not np.isfinite(integer).all() or np.any(integer != np.rint(integer))
            or labels.shape != (len(integer),) or labels.dtype.kind not in 'iu'
            or np.any((labels < 0) | (labels >= 12)) or weights.shape != labels.shape
            or not np.isfinite(weights).all() or np.any(weights <= 0)
            or counts.shape != (12,) or not np.isfinite(counts).all() or np.any(counts <= 0)):
        raise ValueError('original twelve-bin weighted integer-mode P window required')
    np.testing.assert_array_equal(np.bincount(labels, weights=weights, minlength=12), counts)
    vectors = integer.astype(float)*(2*np.pi/1000.)
    values = ir_p0_modes(vectors, power, power_nw, transfer, b1, f_growth,
        p_halo=p_halo, reconstruction=reconstruction, damping=harmonics.damping(vectors))
    return np.array([np.bincount(labels, weights=weights*value, minlength=12)/counts for value in values])
