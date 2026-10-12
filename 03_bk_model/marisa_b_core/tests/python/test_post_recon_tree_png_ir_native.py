"""模块大纲：新 C++/oracle 逐字段对照；旧 ABI 回归；shell 原分母；非法输入。"""
from dataclasses import asdict, replace
from pathlib import Path
import os
import numpy as np
import pytest
from marisa_b.post_recon_tree_png_ir_native import NativePostTreeIR, POST_NATIVE_TERMS, project_post_tree_nodes
from marisa_b.tree_png_ir_native import NativeTreeIR
from marisa_b.tree_png_ir_resummation import ReconstructionConfig
from marisa_b.post_recon_tree_png_ir import (
    AdaptiveReconstructionConfig, ir_resummed_post_recon_finite_png_tree,
    post_recon_source_operator_terms,
)
from test_post_recon_tree_png_ir import _external_setup, _ir_grid, _settings, _rotation

CORE = Path(__file__).resolve().parents[2]
LIBRARY = Path(os.environ.get("MARISA_POST_TREE_IR_LIBRARY", CORE / "build/tree_png_ir/libmarisa_post_tree_png_ir.so"))


@pytest.fixture(scope="module")
def native():
    if not LIBRARY.exists():
        pytest.skip("build scripts/build_tree_png_ir.sh before native validation")
    return NativePostTreeIR(LIBRARY)


@pytest.mark.parametrize("cell", [0., 8., 72.])
@pytest.mark.parametrize("eta", [0., 1., .64])
@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("nq", [90, 91])
def test_oracle_all_components(native, cell, eta, noise, nq):
    v, p, nw, m, b = _external_setup()
    v = v @ _rotation(np.array([1., 2., -.7]), .43).T
    grid = _ir_grid(nq)
    r = ReconstructionConfig(10., 1.83, cell, 4 if cell else 0)
    s = _settings(q_min=.0017, angular_mu_order=16, angular_phi_order=32)
    a = AdaptiveReconstructionConfig(.31, eta)
    eps = 120.*np.exp(-(grid[0]/.16)**2) if noise else None
    batch = native.evaluate(v[None], p[None], nw[None], m[None], b, r, *grid, s, a, shift_noise_power=eps)
    ref = ir_resummed_post_recon_finite_png_tree(v,p,nw,m,b,r,*grid,s,a,shift_noise_power=eps)
    got = batch.result()
    for name in ("deterministic", "source_only", "no_wiggle_source", "damping_mixing",
                 "source_stochastic", "stochastic_mixing", "damping"):
        np.testing.assert_allclose(list(asdict(getattr(got,name)).values()),
                                   list(asdict(getattr(ref,name)).values()), rtol=1e-10, atol=1e-9)
    for name in ("sigma_f0", "sigma_f1", "sigma_f2"):
        np.testing.assert_allclose(getattr(got.displacement,name), getattr(ref.displacement,name), rtol=1e-10, atol=1e-12)
    for name in ref.source_operator_terms:
        np.testing.assert_allclose(got.source_operator_terms[name], ref.source_operator_terms[name], rtol=1e-11, atol=1e-8)
    fixed = dict(zip(POST_NATIVE_TERMS,batch.fixed_order[0]))
    for name, value in post_recon_source_operator_terms(v,p,m,b,r,a).items():
        np.testing.assert_allclose(fixed[name], value, rtol=1e-11, atol=1e-8)


def test_matched_adaptive_and_nowiggle(native):
    v,p,nw,m,b = _external_setup()
    r = ReconstructionConfig(10., b.b1, 8., 4)
    a = AdaptiveReconstructionConfig(b.bphi, 1.)
    got = native.evaluate(v[None],p[None],p[None],m[None],b,r,*_ir_grid(),_settings(),a)
    np.testing.assert_allclose(got.variance[:,1:], 0., atol=1e-14)
    np.testing.assert_allclose(got.mixing, 0., atol=1e-14)
    np.testing.assert_allclose(got.source, got.fixed_order, rtol=1e-14)


def test_old_fixed_order_and_old_cic_guard(native):
    v,p,nw,m,b = _external_setup()
    old = NativeTreeIR(LIBRARY)
    r = ReconstructionConfig(10., 1.9, 8., 4)
    new = native.evaluate(v[None],p[None],nw[None],m[None],b,r,*_ir_grid(),_settings())
    fixed = old.evaluate(v[None],p[None],nw[None],m[None],b,sigma2=0.,scheme="fixed_order",reconstruction=r)
    np.testing.assert_allclose(new.fixed_order[:,:24], fixed, rtol=1e-14)
    with pytest.raises(ValueError, match="Cartesian CIC"):
        old.evaluate(v[None],p[None],nw[None],m[None],b,sigma2=1.,reconstruction=r)


def test_node_projection_keeps_original_denominator(native):
    v,p,nw,m,b = _external_setup()
    vv = np.array([v, v @ _rotation(np.array([1.,2.,3.]),.8).T])
    legs = [np.tile(x,(2,1)) for x in (p,nw,m)]
    batch = native.evaluate(vv,*legs,b,ReconstructionConfig(10.,1.9,72.,4),*_ir_grid(51),_settings())
    weights = np.array([.4,.3])
    result = project_post_tree_nodes(batch,weights,np.array([0,0]),number_of_bins=1,
                                    weights_include_original_denominator=True)
    np.testing.assert_allclose(result["deterministic"][0], weights @ batch.deterministic)
    assert result["weight_sum"][0] == pytest.approx(.7)
    assert "variance" not in result and "damping" not in result
    with pytest.raises(ValueError, match="denominator"):
        project_post_tree_nodes(batch,weights,np.array([0,0]),number_of_bins=1,
                                weights_include_original_denominator=False)


@pytest.mark.parametrize("case", ["grid_order", "negative_noise", "nan_leg", "unclosed", "isotropic_cic", "outside_grid"])
def test_native_rejects_invalid_inputs(native, case):
    v,p,nw,m,b = _external_setup()
    grid = list(_ir_grid(31))
    r = ReconstructionConfig(10.,1.9,8.,4)
    s = _settings()
    noise = None
    if case == "grid_order": grid[0][4] = grid[0][3]
    if case == "negative_noise": noise = -np.ones_like(grid[0])
    if case == "nan_leg": p[0] = np.nan
    if case == "unclosed": v[0,0] += .02
    if case == "isotropic_cic": s = replace(s,mesh_damping="isotropic")
    if case == "outside_grid": s = replace(s,q_min=.00001)
    with pytest.raises(ValueError):
        native.evaluate(v[None],p[None],nw[None],m[None],b,r,*grid,s,shift_noise_power=noise)
