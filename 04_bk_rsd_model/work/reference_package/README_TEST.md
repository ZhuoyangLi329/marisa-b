# MARISA-B tree-level PNG RSD reference package

This package contains a self-contained Python numerical oracle and two Chinese design documents for adding plane-parallel RSD to the current MARISA-B finite-local-PNG halo tree bispectrum.

## Scope

Implemented:

- pre-reconstruction oriented redshift-space `Z1` and `Z2`;
- finite local PNG through `G + f_NL L + f_NL^2 Q`;
- current same-shift/no-linear-RSD-removal post-reconstruction map;
- adaptive local-PNG reconstruction denominator through `f_NL^2`;
- continuum Scoccimarro/Legendre multipoles;
- Gaussian DESI-style tree stochastic raw bases;
- leading pre-reconstruction anisotropic RSD IR covariance helper.

Not implemented as fiducial physics:

- one-loop terms;
- GEO-FPT or nonlinear input spectra;
- FoG damping or `k^2` counterterms;
- wide-angle, AP, window, selection or velocity-bias effects;
- a unique finite-PNG RSD stochastic closure;
- the full post-reconstruction RSD IR covariance;
- TriPoSH estimator normalization.

## Files

```text
docs/TREE_LEVEL_PNG_RSD_FORMULAE.md
docs/TREE_LEVEL_PNG_RSD_IMPLEMENTATION_PLAN.md
python/marisa_b/tree_png_rsd_reference.py
tests/python/test_tree_png_rsd_reference.py
examples/tree_png_rsd_example.py
```

## Run tests

```bash
PYTHONPATH=python python -m pytest \
  tests/python/test_tree_png_rsd_reference.py -q
```

## Run example

```bash
PYTHONPATH=python python examples/tree_png_rsd_example.py
```

## Integration warning

The Python module is a scientific oracle, not a drop-in replacement for the native production pipeline. A native implementation should be compared against `G`, `L`, and `Q` separately over random oriented triangles before any likelihood integration.
