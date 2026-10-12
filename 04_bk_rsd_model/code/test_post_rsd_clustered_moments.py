"""模块大纲：独立parent标签枚举、iid极限、signed marks、真实坐标位移回归。"""
from itertools import product
import unittest
import numpy as np
from post_rsd_clustered_moments import ClusteredMarkedMoments
from post_rsd_marked_moments import MarkedPointMoments,ActualMarkedFields,factor,factorial_shift_response


def direct_parent_enumeration(mom,factors):
    """逐个真实parent标签枚举；不使用set partitions或生产child_factor。"""
    parents=[(g,i) for g,n in enumerate(mom.counts) for i in range(n)]
    choices=[[(g,i) for g,i in parents if kind=='F' or g<mom.ngroup] for kind,_,_ in factors]
    total=0j
    for assignment in product(*choices):
        waves={};term=1.+0j
        for (_,power,k),(g,i) in zip(factors,assignment):
            waves.setdefault((g,i),np.zeros(3,dtype=int));waves[g,i]+=k
            if g==mom.ngroup:term*=(-1/mom.nrandom)**power
            else:
                term*=sum(c**power*np.exp(-2j*np.pi*np.dot(x,k)) for c,x in zip(mom.c[g],mom.offsets[g]))
        phase=0
        for (g,i),k in waves.items():
            if np.all(k==0):continue
            if tuple(k)==mom.signal_mode:phase+=1;term*=mom.amplitudes[g]/2
            elif tuple(-k)==mom.signal_mode:phase-=1;term*=mom.amplitudes[g]/2
            else:term=0j;break
        if phase==0:total+=term
    return total


def clustered_coordinate_cases():
    rng=np.random.default_rng(202610092221)
    v=np.array([[1,0,1],[0,2,-1],[-1,-2,0]])
    q=np.concatenate((v,-v,[[1,-1,2],[-1,1,-2]]));rq=np.array([.71,.42,.26]*2+[.18,.18])
    rows=[]
    for n,nran,signed in ((5,2,False),(11,5,True),(23,30,False),(37,30,True)):
        parents=rng.uniform(0,1,(n,3));offset=np.array([[0,0,0],[.011,-.017,.023],[-.013,.009,.019]])
        coords=((parents[:,None,:]+offset[None,:,:])%1).reshape(-1,3)
        weights=np.tile([.6,1.9,-.4] if signed else [.6,1.9,.4],n)
        ran=rng.uniform(0,1,(len(coords)*nran,3));fields=ActualMarkedFields(coords,weights,ran)
        src=np.array([fields.field(factor('D',1,k)) for k in q])
        multiplier=(-1j/(2*np.pi))*rq[:,None]*q/np.sum(q*q,axis=1)[:,None]*src[:,None]
        shifted=lambda x:(np.exp(2j*np.pi*(x@q.T))@multiplier).real
        locations=np.concatenate((coords,ran));s=np.concatenate((shifted(coords),shifted(ran)))
        c=np.concatenate((fields.cd,np.full(len(ran),fields.cr)))
        def direct(e):
            phase=np.exp(-2j*np.pi*(v@(locations+e*s).T));f1=phase@c;f2=phase@(c*c)
            return np.prod(f1)-np.sum(f2.conj()*f1)+2*np.sum(c**3)
        grad=lambda e:(direct(e)-direct(-e))/(2*e)
        numerical=(4*grad(5e-5)-grad(1e-4))/3
        exact,_,_=factorial_shift_response(v,q,rq,fields.expect)
        error=float(abs(numerical-exact)/max(abs(exact),1e-8))
        rows.append(dict(nparents=n,ndata=len(coords),nrandom=len(ran),signed_weights=signed,
            coordinate_derivative=[numerical.real,numerical.imag],Fourier_derivative=[exact.real,exact.imag],
            relative_error=error))
    return rows


class ClusteredMomentsTest(unittest.TestCase):
    def small(self):
        return ClusteredMarkedMoments([2,1],[[.6,1.9],[1.2,-.3]],
            [[[0,0,0],[.011,.017,-.023]],[[.02,0,0],[-.007,.013,.019]]],
            [.016,.035],[1,0,1],2)

    def test_independent_explicit_parent_enumeration(self):
        mom=self.small();k=(1,0,1);neg=tuple(-x for x in k);zero=(0,0,0)
        cases=[(factor('D',1,zero),),(factor('F',2,zero),),(factor('F',3,zero),),
            (factor('F',1,k),factor('F',1,neg)),
            (factor('D',1,k),factor('F',2,neg),factor('F',1,zero)),
            (factor('D',1,k),factor('F',1,neg),factor('F',1,k),factor('F',1,neg)),
            (factor('F',2,k),factor('D',1,neg),factor('F',1,zero))]
        for items in cases:
            np.testing.assert_allclose(mom.expect(*items),direct_parent_enumeration(mom,items),rtol=2e-12,atol=2e-14)

    def test_single_child_iid_limit_and_signed_normalization(self):
        counts=[17,23];weights=[.6,1.9];amps=[.016,.035];k=(1,0,1)
        old=MarkedPointMoments(counts,weights,amps,k,1200)
        new=ClusteredMarkedMoments(counts,[[w] for w in weights],[[[0,0,0]]]*2,amps,k,1200)
        v=np.array([[1,0,1],[0,2,-1],[-1,-2,0]]);q=np.concatenate((v,-v));r=np.tile([.71,.42,.26],2)
        np.testing.assert_allclose(factorial_shift_response(v,q,r,new.expect),
            factorial_shift_response(v,q,r,old.expect),rtol=5e-13,atol=1e-16)
        np.testing.assert_allclose([new.N2,new.N3],[old.N2,old.N3],rtol=3e-15)
        mom=self.small();np.testing.assert_allclose(mom.expect(factor('D',1,(0,0,0))),1.,atol=2e-15)
        np.testing.assert_array_equal(mom.expect(factor('F',1,(0,0,0))),0j)

    def test_coincident_cluster_residual_not_individual_shot(self):
        n=1000;mom=ClusteredMarkedMoments([n],[[1.,1.]],[[[0,0,0],[0,0,0]]],[0.],[1,0,0],30000)
        self.assertEqual(mom.N2,.5/n)
        np.testing.assert_allclose(mom.residual_power((0,2,0)),.5/n,rtol=2e-14)
        mom2=ClusteredMarkedMoments([n],[[-7.3,-7.3]],[[[0,0,0],[0,0,0]]],[0.],[1,0,0],30000)
        np.testing.assert_allclose(mom2.residual_power((0,2,0)),mom.residual_power((0,2,0)),rtol=3e-14)

    def test_correlated_signed_coordinates(self):
        for row in clustered_coordinate_cases():self.assertLess(row['relative_error'],1e-7)

    def test_invalid_normalization_and_labels(self):
        with self.assertRaises(ValueError):
            ClusteredMarkedMoments([2],[[1.,-1.]],[[[0,0,0],[0,0,0]]],[0.],[1,0,0],3)
        for items in ((factor('X',1,(1,0,0)),),(factor('D',0,(1,0,0)),),()):
            with self.assertRaises(ValueError):self.small().expect(*items)


if __name__=='__main__':unittest.main()
