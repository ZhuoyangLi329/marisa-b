"""模块大纲：独立中心矩、parent标签、Gaussian极限与真实坐标回放。"""
import unittest
from itertools import product
import numpy as np
from post_rsd_marked_cumulants import (MomentCumulants,CumulantMoments,
    input_factorial_bispectrum,decompose_factorial_shift,ConditionalUniformRandomMoments)
from post_rsd_marked_moments import factor,factorial_shift_response
from post_rsd_clustered_moments import ClusteredMarkedMoments
from test_post_rsd_clustered_moments import direct_parent_enumeration,clustered_coordinate_cases


class MarkedCumulantsTest(unittest.TestCase):
    def setUp(self):
        self.v=np.array([[1,0,1],[0,2,-1],[-1,-2,0]])
        self.q=np.concatenate((self.v,-self.v,[[1,-1,2],[-1,1,-2]]))
        self.r=np.r_[.71,.42,.26,.71,.42,.26,.18,.18]

    def test_independent_centered_third_fourth_moments(self):
        x=np.array([-2.,1.,3.]);prob=np.array([.2,.5,.3]);label=factor('D',1,(1,0,0))
        moment=lambda *items:prob@x**len(items)
        c=MomentCumulants(moment);mean=prob@x;dx=x-mean;var=prob@dx**2
        expected=[mean,var,prob@dx**3,prob@dx**4-3*var**2]
        for order in range(1,5):
            self.assertAlmostEqual(c(*[label]*order).real,expected[order-1],places=12)
            self.assertAlmostEqual(CumulantMoments(c)(*[label]*order).real,moment(*[label]*order),places=12)

    def test_independent_parent_labels_reconstruct_all_moments(self):
        mom=ClusteredMarkedMoments([2,1],[[.6,1.9],[1.2,-.3]],
            [[[0,0,0],[.011,.017,-.023]],[[.02,0,0],[-.007,.013,.019]]],
            [.016,.035],self.v[0],2)
        c=MomentCumulants(mom.expect)
        sets=[(factor('D',1,self.q[0]),factor('F',1,-self.q[0])),
            (factor('D',1,self.q[0]),factor('F',2,-self.q[0]),factor('F',1,(0,0,0))),
            (factor('D',1,self.q[0]),factor('F',1,-self.q[0]),factor('F',1,self.q[0]),factor('F',1,-self.q[0]))]
        for labels in sets:
            np.testing.assert_allclose(CumulantMoments(c)(*labels),direct_parent_enumeration(mom,labels),rtol=2e-12,atol=2e-14)
        d=decompose_factorial_shift(self.v,self.q,self.r,c)
        np.testing.assert_allclose(d['full'],factorial_shift_response(self.v,self.q,self.r,mom.expect),rtol=2e-12,atol=2e-14)
        np.testing.assert_allclose(d['full'],d['pair_and_means']+d['third_cumulant']+d['fourth_cumulant'],rtol=3e-15)
        self.assertGreater(abs(d['third_cumulant'][0])+abs(d['fourth_cumulant'][0]),1e-8)

    def test_Gaussian_pair_limit_random_cancellation(self):
        v=self.v;r=self.r[:3];D=np.array([.023,.041,.009]);N=np.array([.007,.011,.004]);n2=.009
        lookup={tuple(k):i for i,k in enumerate(v)};lookup.update({tuple(-k):i for i,k in enumerate(v)})
        expected=n2*np.sum(r*(D+N))
        for i,j in ((0,1),(1,2),(2,0)):
            k=v[i]+v[j];t=np.dot(k,v[i])/np.dot(v[i],v[i])*r[i]+np.dot(k,v[j])/np.dot(v[j],v[j])*r[j]
            expected-=t*(D[i]+N[i])*(D[j]+N[j])
        for nr in (0.,.0003,.01):
            def cumulant(*items):
                if len(items)==1:
                    kind,power,k=items[0]
                    return n2+nr if power==2 and k==(0,0,0) else 0.
                if len(items)>=3:return 0.
                a,b=items;ka=np.asarray(a[2]);kb=np.asarray(b[2])
                if a[1]!=1 or b[1]!=1 or np.any(ka+kb) or not np.any(ka):return 0.
                i=lookup[tuple(ka)];return D[i]+N[i]+(nr if a[0]==b[0]=='F' else 0.)
            d=decompose_factorial_shift(v,self.q,self.r,cumulant)
            np.testing.assert_allclose(d['full'][0],expected,rtol=3e-14,atol=2e-18)
            np.testing.assert_array_equal(d['third_cumulant'],np.zeros(3))
            np.testing.assert_array_equal(d['fourth_cumulant'],np.zeros(3))

    def test_intrinsic_factorial_labels_against_direct_particles(self):
        rng=np.random.default_rng(20261010);x=rng.uniform(0,1,(7,3));ran=rng.uniform(0,1,(14,3));w=rng.uniform(.2,2.,7)
        from post_rsd_marked_moments import ActualMarkedFields
        f=ActualMarkedFields(x,w,ran);pos=np.r_[x,ran];c=np.r_[f.cd,np.full(len(ran),f.cr)]
        phase=np.exp(-2j*np.pi*self.v@pos.T);expected=0j
        for i in range(len(pos)):
            for j in range(len(pos)):
                for k in range(len(pos)):
                    if len({i,j,k})==3:expected+=c[i]*c[j]*c[k]*phase[0,i]*phase[1,j]*phase[2,k]
        np.testing.assert_allclose(input_factorial_bispectrum(self.v,f.expect),expected,rtol=2e-13,atol=1e-14)

    def test_independent_coordinate_derivative_still_required(self):
        for row in clustered_coordinate_cases():self.assertLess(row['relative_error'],1e-7)

    def test_missing_connected_inputs_and_invalid_labels_rejected(self):
        def missing(*items):
            if len(items)>2:raise KeyError('connected3/4 not supplied')
            return 0.
        with self.assertRaises(KeyError):decompose_factorial_shift(self.v,self.q,self.r,missing)
        with self.assertRaises(ValueError):MomentCumulants(lambda *items:np.nan)(factor('D',1,self.v[0]))
        with self.assertRaises(ValueError):decompose_factorial_shift(self.v,self.q[:0],self.r,missing)
        with self.assertRaises(ValueError):MomentCumulants(lambda *items:0.)(factor('X',1,self.v[0]))

    def test_finite_uniform_random_by_independent_discrete_coordinates(self):
        from post_rsd_marked_moments import ActualMarkedFields
        x=np.array([[.125,0,0],[.347,0,0],[.621,0,0]]);w=np.array([.6,1.9,-.4]);c=w/w.sum()
        def data(power,mode):return np.exp(-2j*np.pi*x@np.array(mode))@(c**power)
        k=np.array([1,0,0]);zero=(0,0,0)
        sets=[(factor('F',2,zero),),(factor('F',3,zero),),
            (factor('D',1,k),factor('F',1,-k)),
            (factor('D',1,k),factor('F',2,-2*k),factor('F',1,k)),
            (factor('F',1,k),factor('F',1,-k),factor('F',1,k),factor('F',1,-k))]
        conditional=ConditionalUniformRandomMoments(data,2)
        # 19格足以使本测试所有非零wave sums都不alias；穷举两个真实random坐标。
        measured=np.zeros(len(sets),dtype=complex)
        for i,j in product(range(19),repeat=2):
            ran=np.array([[i/19,0,0],[j/19,0,0]])
            actual=ActualMarkedFields(x,w,ran)
            measured+=np.array([actual.expect(*labels) for labels in sets])/19**2
        np.testing.assert_allclose(np.array([conditional(*labels) for labels in sets]),measured,rtol=2e-13,atol=2e-14)
        np.testing.assert_array_equal(conditional(factor('F',1,zero)),0j)


if __name__=='__main__':unittest.main()
