"""模块大纲：独立物理P表达式/直接Chebyshev B；全cross-C白化；四节点拒绝。"""
import unittest
import numpy as np
from post_rsd_noir_likelihood import PostNoIRLikelihood,predict_p_rows


class PostLikelihoodTest(unittest.TestCase):
    def test_full_joint_whitening_and_native_parameter_order(self):
        rng=np.random.default_rng(2026100917);nP,nB=12,8;n=nP+nB
        coefficients=rng.normal(size=(5,3,3,nB))*1.e-3
        templates=rng.normal(size=(6,n));templates=templates[:,:nP]
        noise=rng.normal(size=(4,n));target=rng.normal(size=n)
        matrix=rng.normal(size=(n,n));cov=matrix@matrix.T+np.eye(n)
        chol=np.linalg.cholesky(cov);p=1.417
        model=PostNoIRLikelihood(target,chol,coefficients,templates,noise,p)
        values=np.column_stack((rng.uniform(-100,100,9),rng.uniform(.6,3.8,9),
            rng.uniform(-3,5,9),rng.uniform(-3,3,9),rng.normal(size=(9,4))))
        expected=[];growth=.7625790037720044
        for row in values:
            f,b,b2,bk=row[:4];x=(b-2.25)/1.75
            # 独立显式Chebyshev多项式；不用生产predict_noir或chebvander。
            cb=np.array([1,x,2*x*x-1,4*x**3-3*x,8*x**4-8*x*x+1])
            bk_value=np.einsum('i,j,k,ijkn->n',cb,[1,b2,bk],[1,f,f*f],coefficients)
            bp=2*1.686*(b-p)
            pk_value=(b*b*templates[0]+2*b*growth*templates[1]+growth**2*templates[2]
                +2*f*bp*(b*templates[3]+growth*templates[4])+f*f*bp*bp*templates[5])
            expected.append(np.concatenate((pk_value,bk_value))+row[4:]@noise)
        expected=np.array(expected)
        np.testing.assert_allclose(model.predictions(values),expected,rtol=2e-12,atol=1e-9)
        white=np.linalg.solve(chol,(expected-target).T).T
        np.testing.assert_allclose(model.residual(values),white,rtol=2e-12,atol=1e-9)

    def test_B_only_quartic_and_full_covariance(self):
        rng=np.random.default_rng(2026100918);n=9
        coefficients=rng.normal(size=(5,3,3,n));noise=rng.normal(size=(3,n))
        target=rng.normal(size=n);matrix=rng.normal(size=(n,n));chol=np.linalg.cholesky(matrix@matrix.T+np.eye(n))
        model=PostNoIRLikelihood(target,chol,coefficients,np.empty((6,0)),noise,1.4)
        values=np.array([[100.,2.7,.3,-.2,.5,-.1,.4]])
        np.testing.assert_allclose(model.residual(values),np.linalg.solve(chol,(model.predictions(values)-target).T).T,
            rtol=3e-13,atol=2e-9)
        with self.assertRaisesRegex(ValueError,'five-node'):
            PostNoIRLikelihood(target,chol,coefficients[:4],np.empty((6,0)),noise,1.4)

    def test_explicit_p_and_no_noise_or_covariance_fallback(self):
        with self.assertRaises(TypeError):predict_p_rows(np.array([[0.,2.]]),np.ones((6,12)))
        with self.assertRaises(ValueError):predict_p_rows(np.array([[0.,2.]]),np.ones((6,12)),p_halo=np.nan)
        c=np.zeros((5,3,3,3));p=np.empty((6,0));noise=np.empty((0,3))
        with self.assertRaises(ValueError):PostNoIRLikelihood(np.zeros(3),np.zeros((3,3)),c,p,noise,1.3)
        with self.assertRaises(ValueError):PostNoIRLikelihood(np.zeros(3),np.eye(3),c,p,np.zeros((2,2)),1.3)

    def test_parameter_dependent_noise_keeps_dimension_and_full_joint_whitening(self):
        rng=np.random.default_rng(202610092230);nP,nB=12,9;n=nP+nB
        matrix=rng.normal(size=(n,n));chol=np.linalg.cholesky(matrix@matrix.T+np.eye(n))
        c=rng.normal(size=(5,3,3,nB))*1e-4;p=rng.normal(size=(6,nP))
        noise=rng.normal(size=(4,n));nc=rng.normal(size=(4,5,3,3,n))*1e-5
        target=rng.normal(size=n);base=PostNoIRLikelihood(target,chol,c,p,noise,1.417)
        model=PostNoIRLikelihood(target,chol,c,p,noise,1.417,noise_coefficients=nc)
        rows=np.array([[100.,2.7,.3,-.2,.5,-.1,.4,.2],[-171.,1.4,-.7,.8,-.3,.2,.1,-.2]])
        self.assertEqual(model.ndim,base.ndim);expected=base.predictions(rows)
        for i,row in enumerate(rows):
            f,b,b2,bk=row[:4];x=(b-2.25)/1.75
            cb=[1,x,2*x*x-1,4*x**3-3*x,8*x**4-8*x*x+1]
            expected[i]+=np.einsum('a,i,j,k,aijkn->n',row[4:],cb,[1,b2,bk],[1,f,f*f],nc)
        np.testing.assert_allclose(model.predictions(rows),expected,rtol=3e-13,atol=2e-10)
        np.testing.assert_allclose(model.residual(rows),np.linalg.solve(chol,(expected-target).T).T,
            rtol=3e-13,atol=2e-10)

    def test_polynomial_constant_limit_and_invalid_tensor(self):
        rng=np.random.default_rng(202610092231);n=9
        c=rng.normal(size=(5,3,3,n))*1e-4;noise=rng.normal(size=(3,n))
        nc=np.zeros((3,5,3,3,n));nc[:,0,0,0]=noise
        args=(np.zeros(n),np.eye(n),c,np.empty((6,0)))
        constant=PostNoIRLikelihood(*args,noise,1.4)
        polynomial=PostNoIRLikelihood(*args,np.zeros_like(noise),1.4,noise_coefficients=nc)
        rows=np.array([[100.,2.7,.3,-.2,.5,-.1,.4]])
        np.testing.assert_allclose(polynomial.residual(rows),constant.residual(rows),rtol=3e-15)
        for bad in (nc[:,:4],np.full(nc.shape,np.nan),np.zeros((2,5,3,3,n))):
            with self.assertRaises(ValueError):PostNoIRLikelihood(*args,noise,1.4,noise_coefficients=bad)


if __name__=='__main__':unittest.main()
