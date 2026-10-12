"""模块大纲：PN_cross独立有向闭合卷积；不调用生产field/transform/integral。

直接输入整数周期网格、实际谱和两个shell masks，独立构造D和固定R。
六个有向配对按显式closing=-input_sum收缩；无第三腿cut，保留full-shell分母。
这只检查unit-amplitude算子，不校准实际STO噪声、不放行MCMC或IR。
"""
import numpy as np


def cross_leg_convolution(vectors, power, transfer, mask_i, mask_j, *, b1,
                          f_growth, p_halo, reconstruction):
    """返回PN_cross的G/L/Q与几何信息，不复用生产三factor IFFT积分。

令V(k)=k R_*(k)/k²，D=P_L[b1+f mu²+fNL bphi/M]²。
对于输入x,y与closing z=-x-y，逐三角形的六项为
    Vx.z Dy + Dx Vy.z + Vz.y Dx + Dz Vx.y + Vy.x Dz + Dy Vz.x。
这里FFT只求input-pair的加权卷积，随后显式查找负波矢closing并求和。
weighted卷积不作整数舍入；仅检查虚部舍入误差，原科学误差门由调用者执行。
"""
    v = np.asarray(vectors, dtype=float)
    p = np.asarray(power, dtype=float)
    m = np.asarray(transfer, dtype=float)
    if v.ndim != 4 or v.shape[-1] != 3 or len(set(v.shape[:3])) != 1:
        raise ValueError('one cubic full Fourier lattice required')
    n = v.shape[0]
    if n < 4 or n % 2 or p.shape != v.shape[:3] or m.shape != p.shape:
        raise ValueError('even lattice and matching spectrum arrays required')
    if not np.isfinite(v).all() or not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError('finite vectors and spectra required')
    if np.any(p < 0) or np.any(m <= 0) or not np.isfinite([b1, f_growth, p_halo]).all():
        raise ValueError('nonnegative power, positive transfer and finite physical parameters required')
    kf = v[1, 0, 0, 0]
    if kf <= 0:
        raise ValueError('positive fundamental frequency required')
    axis = np.rint(np.fft.fftfreq(n)*n)*kf
    for a in range(3):
        shape = [1, 1, 1]; shape[a] = n
        if not np.array_equal(v[..., a], np.broadcast_to(axis.reshape(shape), p.shape)):
            raise ValueError('standard full-lattice index ordering required')
    mi, mj = np.asarray(mask_i), np.asarray(mask_j)
    if mi.dtype != bool or mj.dtype != bool or mi.shape != p.shape or mj.shape != p.shape:
        raise ValueError('two explicit boolean full-shell masks required')
    denominator = int(mi.sum())*int(mj.sum())
    if denominator == 0:
        raise ValueError('nonempty full-shell denominator required')
    neg = (-np.arange(n)) % n
    opposite = np.ix_(neg, neg, neg)
    for mask in (mi, mj):
        if not np.array_equal(mask, mask[opposite]):
            raise ValueError('inversion-symmetric shells required')
    k2 = np.sum(v*v, axis=-1); active = k2 > 0
    ai, aj = mi & active, mj & active
    if not ai.any() or not aj.any():
        raise ValueError('nonzero source legs required')
    if np.sqrt(k2[ai].max())+np.sqrt(k2[aj].max()) >= n*kf/2:
        raise ValueError('closing sum would wrap at the Nyquist boundary')
    response = np.zeros_like(p)
    if reconstruction is None:
        return np.zeros(3), dict(full_denominator=denominator,
            nonzero_input_pairs=int(ai.sum())*int(aj.sum()), response_disabled=True)
    response[active] = reconstruction.response(v[active], m[active])
    inv = np.divide(1., k2, out=np.zeros_like(k2), where=active)
    z0 = b1+f_growth*v[..., 2]**2*inv
    zphi = 2*1.686*(b1-p_halo)/m
    design = np.stack((p*z0*z0, 2*p*z0*zphi, p*zphi*zphi))
    design[:, ~active] = 0.
    result = np.zeros(3); maximum_imaginary_roundoff = 0.

    def contraction(left_fft, right_fft, closing):
        nonlocal maximum_imaginary_roundoff
        convolution = np.fft.ifftn(left_fft*right_fft)
        scale = max(1., float(np.max(np.abs(convolution.real))))
        residual = float(np.max(np.abs(convolution.imag)))/scale
        maximum_imaginary_roundoff = max(maximum_imaginary_roundoff, residual)
        if residual > 5e-12:
            raise ValueError('weighted convolution complex roundoff check failed')
        return float(np.sum(convolution.real*closing))

    for a in range(3):
        # 不缓存所有axis/degrees，以限制独立检查的峰值内存。
        component = v[..., a]
        velocity = component*inv*response
        closing_k = np.where(active, component, 0.)[opposite]
        closing_v = velocity[opposite]
        vi, vj = np.fft.fftn(ai*velocity), np.fft.fftn(aj*velocity)
        ki, kj = np.fft.fftn(ai*component), np.fft.fftn(aj*component)
        for degree in range(3):
            di, dj = np.fft.fftn(ai*design[degree]), np.fft.fftn(aj*design[degree])
            closing_d = design[degree][opposite]
            result[degree] += contraction(vi, dj, closing_k)
            result[degree] += contraction(di, vj, closing_k)
            result[degree] += contraction(di, kj, closing_v)
            result[degree] += contraction(vi, kj, closing_d)
            result[degree] += contraction(ki, vj, closing_d)
            result[degree] += contraction(ki, dj, closing_v)
    result /= denominator
    if not np.isfinite(result).all():
        raise ValueError('nonfinite independent PN_cross projection')
    return result, dict(full_denominator=denominator,
        nonzero_input_pairs=int(ai.sum())*int(aj.sum()),
        maximum_weighted_convolution_imaginary_roundoff=maximum_imaginary_roundoff,
        closing_leg_unrestricted=True, numerator_source_zero_legs_removed=True,
        weighted_convolution_coefficients_rounded=False,
        production_values_transform_integral_called=False)
