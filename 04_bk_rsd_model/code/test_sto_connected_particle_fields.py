"""模块大纲：实际cohort入口的particle DFT和mean/connected组装独立预检。"""
import unittest
import numpy as np
from measure_sto_marked_connected_probes import direct_modes,summarize
from post_rsd_marked_moments import factorial_shift_response
from post_rsd_marked_cumulants import input_factorial_bispectrum


class ConnectedParticleFieldsTest(unittest.TestCase):
    def test_direct_particle_modes_weight_powers_and_conjugates(self):
        rng=np.random.default_rng(93110);x=rng.uniform(0,1000.,(11,3));w=rng.uniform(.2,2.,11)
        modes=np.array([[0,0,0],[1,2,-3],[-1,-2,3],[0,-3,1],[0,3,-1]])
        actual=direct_modes(x,w,modes);expected=np.empty_like(actual)
        for power in (1,2,3):
            for j,k in enumerate(modes):
                expected[power-1,j]=sum((w[i]/sum(w))**power*np.exp(-2j*np.pi*sum(x[i,a]*k[a] for a in range(3))/1000.) for i in range(len(w)))
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-15)
        np.testing.assert_allclose(direct_modes(x,7.3*w,modes),expected,rtol=2e-13,atol=2e-15)
        np.testing.assert_array_equal(actual[:,1],actual[:,2].conj())

    def test_actual_multirow_connected_assembly_and_finite_random(self):
        v=np.array([[1,0,1],[0,2,-1],[-1,-2,0]])
        q=np.r_[v,-v,[[2,-1,3],[-2,1,-3]]];r=np.r_[.71,.42,.26,.71,.42,.26,.18,.18]
        modes={(0,0,0)}
        def collect(*items):
            for _,_,k in items:modes.add(tuple(k))
            return 0j
        factorial_shift_response(v,q,r,collect);input_factorial_bispectrum(v,collect);modes=np.array(sorted(modes))
        rng=np.random.default_rng(93111);rows=[];nr=[]
        for i in range(7):
            x=rng.uniform(0,1000.,(11+i,3));w=rng.uniform(.2,2.,len(x))
            rows.append(direct_modes(x,w,modes));nr.append(30*len(x))
        result=summarize(np.array(rows),np.array(nr),modes,[dict(triangle=v,shift_modes=q,response=r)])[0]
        p=result['cumulant_parts_physical']
        np.testing.assert_allclose(np.array(p['full']),np.array(p['pair_and_means'])+np.array(p['third_cumulant'])+np.array(p['fourth_cumulant']),rtol=2e-13,atol=10.)
        self.assertTrue(result['connected_parts_are_not_identified_pure_noise'])


if __name__=='__main__':unittest.main()
