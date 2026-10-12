"""模块大纲：signed weighted data/random 的离散 self-pair/triple 独立数学 oracle。

这是小型直接点枚举测试，不替代实际 estimator，也不宣称位移噪声闭合。
"""
import itertools
import unittest
import numpy as np


def contraction_identity(position,coefficient,vectors):
    # F_j(k)=sum c_a^j exp(-ik.x_a)，不把 w² 场替换成普通 P。
    phase=np.exp(-1j*np.asarray(vectors)@np.asarray(position).T)
    c=np.asarray(coefficient)
    f1=phase@c
    raw=np.prod(f1)
    pair=sum((phase[i]*phase[j])@(c*c)*f1[k] for i,j,k in ((0,1,2),(1,2,0),(2,0,1)))
    triple=(np.prod(phase,axis=0)@(c*c*c))
    return raw-pair+2*triple,raw,pair,triple


def brute_distinct(position,coefficient,vectors):
    phase=np.exp(-1j*np.asarray(vectors)@np.asarray(position).T)
    c=np.asarray(coefficient)
    return sum(c[i]*c[j]*c[k]*phase[0,i]*phase[1,j]*phase[2,k]
               for i,j,k in itertools.permutations(range(len(c)),3))


class WeightedFactorialTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(2026100804)
        self.x=rng.uniform(0,1000,(13,3));self.w=rng.uniform(.3,2.4,13)
        self.k=np.array([[1,2,1],[2,-1,0],[-3,-1,-1]])*2*np.pi/1000.

    def test_weighted_signed_catalog_exact_distinct_sum(self):
        # 8 个加权 data + 5 个负号 random，统一同一 field normalization。
        c=np.concatenate((self.w[:8]/self.w[:8].sum(),-np.ones(5)/5))
        for x in (self.x,self.x+np.sin(self.x/31.)*4.):
            actual,*_=contraction_identity(x,c,self.k)
            expected=brute_distinct(x,c,self.k)
            np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=1e-14)

    def test_equal_weight_reduces_to_standard_formula(self):
        n=len(self.x);c=np.ones(n)/n
        actual,raw,_,_=contraction_identity(self.x,c,self.k)
        f1=np.exp(-1j*self.k@self.x.T)@c
        expected=raw-np.sum(np.abs(f1)**2)/n+2/n**2
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=1e-14)

    def test_effective_density_does_not_replace_weight_squared_field(self):
        c=self.w/self.w.sum();actual,raw,_,_=contraction_identity(self.x,c,self.k)
        f1=np.exp(-1j*self.k@self.x.T)@c
        shortcut=raw-np.sum(np.abs(f1)**2)*np.sum(c*c)+2*np.sum(c*c*c)
        self.assertGreater(abs(actual-shortcut),1e-4)

    def test_weight_rescaling_and_common_translation(self):
        c=self.w/self.w.sum();expected=contraction_identity(self.x,c,self.k)[0]
        np.testing.assert_allclose(contraction_identity(self.x+np.array([17.,-31.,6.]),c,self.k)[0],expected,atol=2e-14)
        np.testing.assert_allclose(contraction_identity(self.x,(9*self.w)/(9*self.w).sum(),self.k)[0],expected,atol=2e-14)


if __name__=='__main__':unittest.main()
