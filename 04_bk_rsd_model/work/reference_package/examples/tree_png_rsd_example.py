"""Minimal pre/post tree-level finite-PNG RSD example."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from marisa_b.tree_png_rsd_reference import (  # noqa: E402
    HaloBiasV1,
    ReconstructionConfig,
    RsdConfig,
    finite_png_halo_bispectrum_rsd,
    project_scoccimarro_multipole,
)

vectors = np.array(
    [
        [0.052, 0.013, 0.027],
        [-0.018, 0.047, -0.011],
        [-0.034, -0.060, -0.016],
    ]
)
p_linear = np.array([1180.0, 970.0, 830.0])
transfer_m = np.array([0.82, 1.07, 0.91])

bias = HaloBiasV1(
    b1=2.08,
    b2=0.41,
    bK2=-0.27,
    bphi=3.25,
    bphidelta=1.36,
    bphi2=2.42,
)
rsd = RsdConfig(f_growth=0.79, los=(0.0, 0.0, 1.0))
reconstruction = ReconstructionConfig(
    smoothing_radius=15.0,
    bias_recon=2.4,
    # This matches the current same-shift/no-linear-RSD-removal convention.
    denominator_mode="bias_only",
)

pre = finite_png_halo_bispectrum_rsd(
    vectors,
    p_linear,
    transfer_m,
    bias,
    rsd,
)
post = finite_png_halo_bispectrum_rsd(
    vectors,
    p_linear,
    transfer_m,
    bias,
    rsd,
    reconstruction=reconstruction,
)

f_nl = 25.0
print("oriented pre  (G,L,Q):", pre.total)
print("oriented post (G,L,Q):", post.total)
print("pre prediction at f_NL=25:", pre.total.predict(f_nl))
print("post prediction at f_NL=25:", post.total.predict(f_nl))

k1, k2, k3 = np.linalg.norm(vectors, axis=1)
monopole = project_scoccimarro_multipole(
    k1,
    k2,
    k3,
    p_linear,
    transfer_m,
    bias,
    rsd,
    ell=0,
    n_mu=20,
    n_phi=22,
    reconstruction=reconstruction,
)
print("post monopole (G,L,Q):", monopole)
