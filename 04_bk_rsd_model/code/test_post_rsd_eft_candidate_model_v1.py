"""模块大纲：逐盒直接公式对比mean/product，full P-B cross白化及raw positivity。"""
import unittest
import numpy as np
from post_rsd_eft_candidate_model_v1 import EFTCandidateModel
from post_rsd_stochastic_eft_v1 import MarkedNoiseEFT,intrinsic_from_primitives,zero_from_primitives
from post_rsd_general_noise_transport import combine_white_operators


class CandidateModelTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261010);self.nB=9
        self.n2=np.array([2700.,3500.,3200.,4100.]);self.n3=np.array([9e6,1.5e7,1.2e7,2.3e7])
        self.nr=np.array([93.,119.,111.,134.])
        moments=np.array([self.n2.mean(),self.n3.mean(),np.mean(self.n2**2),self.nr.mean(),np.mean(self.n2*self.nr)])
        self.model=EFTCandidateModel(np.zeros((5,3,3,9)),np.zeros((6,12)),
            rng.normal(0,1,(3,3,3,9)),rng.normal(0,1,(7,9)),rng.normal(0,1,(14,9)),moments,1.28874,.762579)
        self.rows=np.array([[100.,2.23,.1,-.4,.5,-.3,.11,.73,2.8,3.5,.89],
                            [-100.,1.92,1.,.4,-.2,.13,-.1,1.24,2.5,-.2,1.11]])

    def test_independent_average_of_each_actual_self_moment(self):
        for row in self.rows:
            fnl,b,_,_,ad,ap,s,u,bw,bpw,j=row;bp=2*1.686*(b-self.model.p_halo)
            cb=np.array([1.,(b-2.25)/1.75,2*((b-2.25)/1.75)**2-1]);power=np.array([1.,fnl,fnl*fnl])
            ops=np.einsum('i,iafb,f->ab',cb,self.model.operators,power)
            boxes=[]
            for n2,n3,nr in zip(self.n2,self.n3,self.nr):
                noise=MarkedNoiseEFT(u*n2,n2,n3,ad*b*n2,ap*2*1.686*n2,s*n2*n2,bw,bpw,j*n3,nr)
                bvalue=intrinsic_from_primitives(self.model.intrinsic_primitives,b1=b,bphi=bp,
                    f_growth=self.model.f_growth,fnl=fnl,noise=noise)
                bvalue+=combine_white_operators(ops,noise_data_raw=u*n2,n2_data_self=n2)
                bvalue+=zero_from_primitives(self.model.zero_primitives,b1=b,bphi=bp,
                    f_growth=self.model.f_growth,fnl=fnl,noise=noise)
                boxes.append(np.r_[np.full(12,(u-1)*n2),bvalue])
            np.testing.assert_allclose(self.model.predictions(row)[0],np.mean(boxes,axis=0),rtol=3e-14,atol=3e-7)

    def test_full_joint_cross_covariance_not_marginal_whitening(self):
        rng=np.random.default_rng(43);a=rng.normal(size=(21,21));c=a@a.T+np.eye(21)
        chol=np.linalg.cholesky(c);target=rng.normal(size=21)
        r=self.model.residual(self.rows,target,chol)
        for row,white in zip(self.rows,r):
            d=self.model.predictions(row)[0]-target
            np.testing.assert_allclose(white@white,d@np.linalg.solve(c,d),rtol=3e-14)
        with self.assertRaises(ValueError):self.model.residual(self.rows,target,chol.T)

    def test_negative_raw_noise_is_rejected_not_clipped(self):
        rows=self.rows.copy();rows[0,7]=-1
        with self.assertRaises(ValueError):self.model.predictions(rows)


if __name__=='__main__':unittest.main()
