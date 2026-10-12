"""C++/Python 独立实现逐算符交叉验证，包括等边腿、置换、旋转与拒绝路径。"""
import os
from pathlib import Path
import itertools
import numpy as np
import pytest
from marisa_b.tree_png_ir_resummation import (
    ALL_TERMS, HaloTreeBias, ReconstructionConfig, TreeIrConfig,
    triangle_vectors, finite_png_halo_tree_operator_terms, ir_resummed_finite_png_halo_tree,
)
from marisa_b.tree_png_ir_native import NativeTreeIR

@pytest.fixture(scope="module")
def native():
    root=Path(__file__).resolve().parents[2]
    return NativeTreeIR(os.environ.get('MARISA_TREE_IR_LIBRARY',str(root/'build/tree_png_ir/libmarisa_tree_png_ir.so')))

@pytest.mark.parametrize('post',[False,True])
@pytest.mark.parametrize('scheme',['fixed_order','strict_lo','factorized_closure'])
def test_random_atomic_native_oracle(native,post,scheme):
    rng=np.random.default_rng(20260930)
    shapes=[(.1,.1,-.5),(.1,.1,-.999),(.02,.15,.99),(.1,.08,.2)]
    shapes += [(rng.uniform(.008,.2),rng.uniform(.008,.2),rng.uniform(-.99,.99)) for _ in range(45)]
    for k1,k2,mu in shapes:
        v=triangle_vectors(k1,k2,mu)
        # 随机正交旋转验证 pre/post CIC-off 的标量不变性。
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        v=v@rotation
        k=np.linalg.norm(v,axis=1)
        p=1200*np.exp(-3*k*k)/(1+2*k);nw=p*(1+.05*np.sin(110*k));m=.35+4*k*k
        b=HaloTreeBias(*rng.uniform(.2,2.,6))
        r=ReconstructionConfig(10.,1.93) if post else None
        ref=(finite_png_halo_tree_operator_terms(v,p,m,b,r) if scheme=='fixed_order' else
             ir_resummed_finite_png_halo_tree(v,p,nw,m,b,TreeIrConfig(23.,scheme),r).operator_terms)
        got=native.evaluate(v[None],p[None],nw[None],m[None],b,sigma2=23.,scheme=scheme,reconstruction=r)[0,:21]
        np.testing.assert_allclose(got,[ref[x] for x in ALL_TERMS],rtol=1.e-12,atol=1.e-9)
        for perm in itertools.permutations(range(3)):
            ii=np.array(perm)
            rotated=native.evaluate(v[ii][None],p[ii][None],nw[ii][None],m[ii][None],b,sigma2=23.,scheme=scheme,reconstruction=r)[0,:21]
            np.testing.assert_allclose(rotated,got,rtol=1.e-12,atol=1.e-8)

def test_scalar_cic_ir_is_explicitly_rejected(native):
    v=triangle_vectors(.1,.1,-.5)[None];p=np.ones((1,3));b=HaloTreeBias(1,1,1,1,1,1)
    with pytest.raises(ValueError,match='CIC'):
        native.evaluate(v,p,p,p,b,sigma2=10.,reconstruction=ReconstructionConfig(10,1.93,1000/256,2))

def test_invalid_scheme_and_geometry_rejected(native):
    v=triangle_vectors(.1,.1,-.5)[None];p=np.ones((1,3));b=HaloTreeBias(1,1,1,1,1,1)
    with pytest.raises(ValueError,match='scheme'):
        native.evaluate(v,p,p,p,b,sigma2=10.,scheme='tree_nlo')
    v[0,0,0]+=.1
    with pytest.raises(ValueError,match='closed'):
        native.evaluate(v,p,p,p,b,sigma2=10.)
