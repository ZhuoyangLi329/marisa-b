"""模块大纲：有向物理式、非节点参数、全shell闭合枚举及NN标签独立检验。"""
import unittest
from unittest.mock import patch
import numpy as np
from post_rsd_fixedrec import FixedReconstruction
from post_rsd_poisson_projection import PoissonShiftFields,PoissonShiftFFTProjection
from post_rsd_noise_transport import integrate_noise_transport
from post_rsd_noir_polynomial import B_NODES,P_NODES,fit_noir_design,collapse_physical_p
from post_rsd_noir_likelihood import PostNoIRLikelihood
import rsd_b000_fft_projection as fft_source


def direct_operators(v,p,m,b,fg,ph,rec,fnl):
    k2=np.sum(v*v,axis=-1);active=k2>0
    inv=np.divide(1.,k2,out=np.zeros_like(k2),where=active)
    z=b+fg*v[...,2]**2*inv+fnl*2*1.686*(b-ph)/m
    d=p*z*z;r=np.zeros_like(p)
    if rec is not None:r[active]=rec.response(v[active],m[active])
    out=np.zeros((3,len(v)))
    for i,j in ((0,1),(1,2),(2,0)):
        total=v[:,i]+v[:,j]
        ti=np.sum(total*v[:,i],axis=-1)*inv[:,i]*r[:,i]
        tj=np.sum(total*v[:,j],axis=-1)*inv[:,j]*r[:,j]
        out[0]-=ti*d[:,j]+tj*d[:,i]
        out[1]-=ti*d[:,i]+tj*d[:,j]
        out[2]-=ti+tj
    return np.where(np.all(active,axis=-1)[None,:],out,0.)


def sample(k):return 10000./(1+(k/.09)**2),1000.*(1+(k/.02)**2)


class NoiseTransportTest(unittest.TestCase):
    def setUp(self):self.rec=FixedReconstruction(10.,2.2280546354,1.2822,100.)

    def test_oriented_point_formula_and_NN_no_physical_fnl(self):
        rng=np.random.default_rng(202610092237);v=rng.normal(size=(53,3,3))*.03;v[:,2]=-v[:,:2].sum(axis=1)
        p,m=sample(np.linalg.norm(v,axis=-1))
        for b,ph in ((.63,.9),(2.11,1.417),(3.83,1.72)):
            for rec in (None,self.rec):
                fields=PoissonShiftFields(v,p,m,b,.7625790037720044,p_halo=ph,reconstruction=rec)
                coefficients=integrate_noise_transport(fields)
                np.testing.assert_array_equal(coefficients[2,1:],0.)
                for f in (-499.,0.,100.,473.):
                    actual=np.einsum('afn,f->an',coefficients,[1,f,f*f])
                    expected=direct_operators(v,p,m,b,.7625790037720044,ph,rec,f)
                    scale=np.maximum(1.,np.max(np.abs(expected),axis=1))[:,None]
                    np.testing.assert_allclose((actual-expected)/scale,0.,atol=2e-14)

    def test_exact_full_shell_projection_including_zero_leg_rules(self):
        n=32;edges=np.array([0.,.02035714285714286,.04107142857142857]);kf=2*np.pi/1000
        axis=np.rint(np.fft.fftfreq(n)*n).astype(int);grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        norm=np.linalg.norm(grid,axis=-1)*kf
        shells=[grid[(norm>=edges[0])&(norm<edges[1])],grid[(norm>=edges[1])&(norm<=edges[2])]]
        with patch.object(fft_source,'load_spectrum',return_value=sample):
            for pair in ((0,0),(0,1),(1,1)):
                a,b=(shells[i] for i in pair);aa=np.repeat(a,len(b),axis=0);bb=np.tile(b,(len(a),1));cc=-aa-bb
                v=np.stack((aa,bb,cc),axis=1)*kf;active=np.linalg.norm(v,axis=-1)>0
                p=np.zeros(active.shape);m=np.ones(active.shape);p[active],m[active]=sample(np.linalg.norm(v[active],axis=-1))
                fields=PoissonShiftFFTProjection(n,edges,pair,2.11,.7625790037720044,p_halo=1.417,reconstruction=self.rec)
                coefficients=integrate_noise_transport(fields)
                self.assertEqual(fields.denominator,len(a)*len(b))
                for f in (-171.,0.,243.):
                    expected=direct_operators(v,p,m,2.11,.7625790037720044,1.417,self.rec,f).mean(axis=1)
                    actual=coefficients@np.array([1,f,f*f])
                    np.testing.assert_allclose(actual,expected,rtol=4e-13,atol=2e-10)

    def test_actual_PN_parameter_dependence_in_likelihood_at_nonnode_points(self):
        rng=np.random.default_rng(202610092238);v=rng.normal(size=(13,3,3))*.025;v[:,2]=-v[:,:2].sum(axis=1)
        p,m=sample(np.linalg.norm(v,axis=-1));nodes=np.zeros((5,3,3,3,len(v)))
        for ib,b in enumerate(B_NODES):
            for ip,ph in enumerate(P_NODES):
                fields=PoissonShiftFields(v,p,m,b,.7625790037720044,p_halo=ph,reconstruction=self.rec)
                nodes[ib,ip,0]=integrate_noise_transport(fields)[0]
        collapsed=collapse_physical_p(fit_noir_design(nodes),p_halo=1.417)
        nc=np.zeros((3,5,3,3,len(v)));nc[0]=collapsed
        matrix=rng.normal(size=(len(v),len(v)));chol=np.linalg.cholesky(matrix@matrix.T+np.eye(len(v)))
        model=PostNoIRLikelihood(np.zeros(len(v)),chol,np.zeros_like(collapsed),np.empty((6,0)),
            np.zeros((3,len(v))),1.417,noise_coefficients=nc)
        for f,b,amplitude in ((-499.,.63,.7),(100.,2.11,-.2),(473.,3.83,1.3)):
            row=np.array([[f,b,.3,-.2,amplitude,.4,-.1]])
            expected=amplitude*direct_operators(v,p,m,b,.7625790037720044,1.417,self.rec,f)[0]
            scale=max(1.,float(np.max(np.abs(expected))))
            np.testing.assert_allclose((model.predictions(row)[0]-expected)/scale,0.,atol=3e-13)
            np.testing.assert_allclose(model.residual(row)[0]/scale,np.linalg.solve(chol,expected)/scale,atol=3e-13)


if __name__=='__main__':unittest.main()
