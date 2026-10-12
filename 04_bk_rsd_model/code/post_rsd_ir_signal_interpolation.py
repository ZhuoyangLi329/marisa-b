"""模块大纲：IR信号的可加密b1节点表示；保留严格fNL²和native二阶bias。

IR指数不属于no-IR的精确四次多项式。这里仅对直接IR投影的b1依赖
作可验证的Chebyshev逼近；节点数不是物理参数，physical p必须显式固定。
本模块不认证精度、不选择STO随机项、不放行MCMC，也不替换旧模型。
"""
from dataclasses import dataclass
import numpy as np

B1_DOMAIN = (.5, 4.)
FNL_DOMAIN = (-500., 500.)


def bias_nodes(count):
    """原完整b1先验区间上的第一类Chebyshev根；不是缩小先验。"""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError('at least two explicit b1 nodes required')
    theta = (np.arange(count)+.5)*np.pi/count
    return 2.25+1.75*np.cos(theta)


@dataclass(frozen=True)
class IRSignalInterpolation:
    coefficients: object
    p_halo: float

    def __post_init__(self):
        c = np.asarray(self.coefficients, dtype=float).copy()
        if c.ndim != 4 or c.shape[0] < 2 or c.shape[1:3] != (3, 3) or c.shape[3] < 1:
            raise ValueError('explicit [b1 degrees,one/b2/bK2,G/L/Q,observable] tensor required')
        if not np.isfinite(c).all() or not np.isscalar(self.p_halo) or not np.isfinite(self.p_halo):
            raise ValueError('finite IR coefficients and explicit fixed physical p required')
        c.flags.writeable = False
        object.__setattr__(self, 'coefficients', c)
        object.__setattr__(self, 'p_halo', float(self.p_halo))

    @classmethod
    def from_nodes(cls, designs, *, p_halo):
        values = np.asarray(designs, dtype=float)
        if values.ndim != 4 or values.shape[1:3] != (3, 3):
            raise ValueError('direct IR samples must retain [node,one/b2/bK2,G/L/Q,bin]')
        n = values.shape[0]; bias_nodes(n)
        if not np.isfinite(values).all():
            raise ValueError('finite directly evaluated IR samples required')
        theta = (np.arange(n)+.5)*np.pi/n
        transform = (2./n)*np.cos(np.arange(n)[:, None]*theta[None, :])
        transform[0] *= .5
        coefficients = (transform@values.reshape(n, -1)).reshape(values.shape)
        return cls(coefficients, p_halo)

    @property
    def metadata(self):
        return dict(representation='approximate_b1_Chebyshev_direct_IR_designs',
            b1_nodes=len(self.coefficients), b1_domain=list(B1_DOMAIN),
            fNL_domain=list(FNL_DOMAIN), fNL_degree=2,
            native_bias_order=['one', 'b2_native', 'bK2'], p_fixed_physical=self.p_halo,
            exact_noIR_quartic_assumed=False, exponent_frozen_at_probe_bias=False,
            independent_projection_accuracy_certified=False,
            stochastic_closure_certified=False, MCMC_released=False)

    def designs_at(self, b1):
        b = np.atleast_1d(np.asarray(b1, dtype=float))
        if b.ndim != 1 or not np.isfinite(b).all() or np.any((b < .5) | (b > 4.)):
            raise ValueError('no b1 extrapolation outside the unchanged [.5,4] prior')
        x = ((b-2.25)/1.75)[:, None, None, None]
        # Clenshaw递推；不重新求解Vandermonde，也不把阻尼冻结在节点。
        next_term = np.zeros((len(b),)+self.coefficients.shape[1:])
        after_term = np.zeros_like(next_term)
        for coefficient in self.coefficients[:0:-1]:
            current = coefficient+2*x*next_term-after_term
            after_term, next_term = next_term, current
        return self.coefficients[0]+x*next_term-after_term

    def predictions(self, physical):
        rows = np.atleast_2d(np.asarray(physical, dtype=float))
        if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
            raise ValueError('finite [fNL,b1,b2_native,bK2] rows required')
        f, b, b2, bk = rows.T
        if np.any((f < -500.) | (f > 500.)):
            raise ValueError('no extrapolation outside the unchanged fNL prior')
        designs = self.designs_at(b)
        nuisance = np.column_stack((np.ones(len(rows)), b2, bk))
        powers = np.column_stack((np.ones(len(rows)), f, f*f))
        return np.einsum('ra,rf,rafk->rk', nuisance, powers, designs)

    def joint_whitened(self, cholesky, *, nP=0):
        """先把B信号放入完整P+B观测量空间，随后使用一个joint Cholesky。"""
        if nP not in (0, 12):
            raise ValueError('only original B-only or twelve-bin P+B geometry supported')
        c = np.asarray(cholesky, dtype=float); n = nP+self.coefficients.shape[-1]
        if (c.shape != (n, n) or not np.isfinite(c).all()
                or np.any(np.diag(c) <= 0) or np.any(np.triu(c, 1) != 0)):
            raise ValueError('positive finite lower joint Cholesky required; no regularization')
        joint = np.zeros(self.coefficients.shape[:-1]+(n,))
        joint[..., nP:] = self.coefficients
        white = np.linalg.solve(c, joint.reshape(-1, n).T).T.reshape(joint.shape)
        return IRSignalInterpolation(white, self.p_halo)
