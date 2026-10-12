"""模块大纲：实际raw data噪声N(k)与粒子self N2分离的leading shift transport。

只保留Gaussian pair/tree计数和F2零模均值；不默认Poisson，不含connected4、
finite-shift/IR或未知PNG噪声响应，不认证真实STO的N(k)校准。
data-only shift、独立均匀random：random白噪声在此leading阶精确相消。
"""
import numpy as np
from rsd_b000_fft_projection import Field, D, ONE
from post_rsd_poisson_projection import PoissonShiftFields, PoissonShiftFFTProjection


def direct_colored_transport(vectors, signal_power, noise_data_power, shift_response, n2_data_self):
    """逐非零有向三角形：raw pairs减外部self，保留原full-shell规则。

Delta=-sum_pairs(Ti+Tj)*(Di*Nj+Dj*Ni+Ni*Nj)+N2*sum_i Ri*(Di+Ni)。
N为实际data raw stochastic power，不是已经扣self的残余，不能含random。
本入口的N(k)不随拟合fNL改变；若需PNG噪声响应必须另给阶数审计。
"""
    v = np.asarray(vectors, dtype=float); d = np.asarray(signal_power, dtype=float)
    n = np.asarray(noise_data_power, dtype=float); r = np.asarray(shift_response, dtype=float)
    if v.shape[-2:] != (3, 3) or d.shape != v.shape[:-1] or n.shape != d.shape or r.shape != d.shape:
        raise ValueError('three explicit directed-leg powers/responses required')
    if not all(np.isfinite(a).all() for a in (v, d, n, r)) or np.any(n < 0):
        raise ValueError('finite nonnegative raw data noise power required')
    if not np.isscalar(n2_data_self) or not np.isfinite(n2_data_self) or n2_data_self < 0:
        raise ValueError('explicit nonnegative actual particle self moment required')
    if not np.allclose(v.sum(axis=-2), 0., rtol=0., atol=1e-12):
        raise ValueError('closed triangles required')
    k2 = np.sum(v*v, axis=-1); active = k2 > 0
    inverse = np.divide(1., k2, out=np.zeros_like(k2), where=active)
    result = n2_data_self*np.sum(r*(d+n), axis=-1)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        total = v[..., i, :]+v[..., j, :]
        ti = np.sum(total*v[..., i, :], axis=-1)*inverse[..., i]*r[..., i]
        tj = np.sum(total*v[..., j, :], axis=-1)*inverse[..., j]*r[..., j]
        result -= (ti+tj)*(d[..., i]*n[..., j]+d[..., j]*n[..., i]+n[..., i]*n[..., j])
    return np.where(np.all(active, axis=-1), result, 0.)


def combine_white_operators(operators, *, noise_data_raw, n2_data_self):
    """Nraw*Kcross+(Nraw-N2)*Ksame+Nraw*(Nraw-N2)*KNN，不设Nraw=N2。"""
    if not all(np.isscalar(x) and np.isfinite(x) and x >= 0 for x in (noise_data_raw, n2_data_self)):
        raise ValueError('finite nonnegative raw noise and separate data self required')
    value = np.asarray(operators, dtype=float)
    if value.shape[0] != 3 or not np.isfinite(value).all():
        raise ValueError('explicit three independently validated transport operators required')
    residual = noise_data_raw-n2_data_self
    return noise_data_raw*value[0]+residual*value[1]+noise_data_raw*residual*value[2]


class _ColoredNoiseFactors:
    def configure_data_noise(self, noise, data_only):
        if data_only is not True:
            raise ValueError('raw noise must explicitly describe data only, not random/subtracted residual')
        value = noise(self.v) if callable(noise) else noise
        self.noise_data_power = np.asarray(np.broadcast_to(value, self.p.shape), dtype=float).copy()
        if not np.isfinite(self.noise_data_power).all() or np.any(self.noise_data_power < 0):
            raise ValueError('finite nonnegative explicit data noise spectrum required')
        self.noise_data_power.flags.writeable = False

    def degrees(self, field):
        return range(1) if field.root in ('N', 'NR') else super().degrees(field)

    def values(self, field, degree):
        if field.root in ('N', 'NR'):
            if degree != 0:
                raise ValueError('unvalidated physical-fNL-dependent noise spectrum')
            base = Field('one' if field.root == 'N' else 'SR', field.axes, field.inv2, field.invM)
            return super().values(base, 0)*self.noise_data_power
        return super().values(field, degree)


class GeneralNoisePointFields(_ColoredNoiseFactors, PoissonShiftFields):
    def __init__(self, *args, noise_data_power, noise_is_data_only, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_data_noise(noise_data_power, noise_is_data_only)


class GeneralNoiseFFTProjection(_ColoredNoiseFactors, PoissonShiftFFTProjection):
    def __init__(self, *args, noise_data_power, noise_is_data_only, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_data_noise(noise_data_power, noise_is_data_only)


def integrate_general_noise_transport(fields, *, n2_data_self):
    """复用唯一cyclic/FFT/全shell/零腿实现；输出原物理G/L/Q。

已知N(k)作为外部校准输入，绝不自动复用N2或旧pre stochastic bases。
当前不将此leading Gaussian-pair候选写成正式STO noise PASS。
"""
    if not isinstance(fields, _ColoredNoiseFactors):
        raise TypeError('explicit calibrated raw data noise fields required')
    if not np.isscalar(n2_data_self) or not np.isfinite(n2_data_self) or n2_data_self < 0:
        raise ValueError('explicit finite nonnegative particle data self required')
    shape = np.shape(fields.integral(ONE, 0, ONE, 0, ONE, 0)); result = np.zeros((3,)+shape)
    nr, noise = Field('NR'), Field('N')
    for axis in range(3):
        response_noise = nr.op((axis,), inv2=True)
        response_signal = Field('DR').op((axis,), inv2=True)
        closing = ONE.op((axis,))
        for degree in range(3):
            result[degree] += fields.integral(response_noise, 0, D, degree, closing, 0)
            result[degree] += fields.integral(D, degree, response_noise, 0, closing, 0)
            result[degree] += fields.integral(response_signal, degree, noise, 0, closing, 0)
            result[degree] += fields.integral(noise, 0, response_signal, degree, closing, 0)
        result[0] += fields.integral(response_noise, 0, noise, 0, closing, 0)
        result[0] += fields.integral(noise, 0, response_noise, 0, closing, 0)
    for degree in range(3):
        result[degree] += n2_data_self*fields.integral(Field('DR'), degree, ONE, 0, ONE, 0)
    result[0] += n2_data_self*fields.integral(nr, 0, ONE, 0, ONE, 0)
    return result
