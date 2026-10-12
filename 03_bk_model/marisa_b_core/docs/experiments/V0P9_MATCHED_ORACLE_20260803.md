# v0.9 matched/oracle reconstruction experiment

## Status

This experiment is a frozen negative diagnostic.  It is intentionally kept
outside the production model contract.

The test uses the Quijote real-space halo `fNL=0`, `z=1`, `R=15` ensemble and
the single-realization covariance of 500 boxes.  The forward model follows
the conditional path `fNL_rec=fNL`, while the measured post-reconstruction
catalogue is the ordinary fiducial `fNL=0` catalogue.  It is therefore an
oracle performance bound, not an operational likelihood for an unknown
primordial amplitude.

The model combines the halo tree baseline, the pure-matter SPT one-loop
uplift, finite local-PNG halo tree terms through `fNL^2`, the matter PNG
one-loop/B112II uplift, and two leading stochastic amplitudes.  The fitted
parameters are `fNL`, free `b1`, `b2`, `gamma2`, `Ashot_residual`, and
`Bshot_residual`, with the hard `fNL` prior `[-150,150]`.

## Main result

At `kmax=0.14 h/Mpc`, the fixed-response covariance swap gives:

| Response | Covariance | sigma(fNL) |
|---|---|---:|
| pre | pre | 26.864 |
| pre | post | 13.524 |
| matched/oracle post | pre | 45.325 |
| matched/oracle post | post | 24.932 |

With the same post covariance, the matched post response retains
`(13.524/24.932)^2 = 0.294` of the pre-response Fisher information.  The
improved post covariance masks a loss of about 70.6% of the PNG response
information.  Changing only the scale dependence of `b_rec(k)` is therefore
not a sufficient repair.

The converged `kmax=0.14` marginal MCMC half-widths are 50.936 (pre) and
52.174 (post).  All eight pre/post and `kmax` contexts pass the registered
R-hat, bulk-ESS, tail-ESS, and boundary-mass gates.

The matter-loop reconstruction response has norm only
`5.231840844658714e-4 sigma_single`; adding more denominator-loop derivatives
cannot plausibly close the observed response gap.

## Finite-amplitude limitation

The literal linear denominator

```text
b_rec(k,fNL) = b1_rec + fNL * bphi_rec / M(k)
```

crosses zero at approximately `fNL=-146.004296` on the fundamental mode.
The local tangent at `fNL=0` remains valid, but an exact finite-amplitude path
cannot be positive over the entire nominal prior.  The machine-readable
release summary therefore reports `completed_with_failed_acceptance_gate`.

## Source and artifacts

The normalized source entry points are:

- `scripts/experiments/run_marisa_b_v0p9_matched_oracle.py`
- `src/experiments/marisa_b_v0p9_oracle_matter_driver.cpp`

The private GitHub Release associated with tag
`v0.9-matched-oracle-test` carries:

- `marisa-b-v0p9-matched-oracle-artifacts-20260804.tar.gz`, containing the
  compact registered inputs, successful raw records, final JSON/NPZ results,
  five PDFs, exact runtime source, and per-file checksums;
- `chains.h5`, the compact converged chain with its SHA256 recorded in the
  bundle README.

The exact runtime source in the artifact retains the original machine-local
layout and is the source identity recorded by the frozen result files.  The
branch versions normalize path discovery for a standalone clone; rerunning
them produces a new provenance identity even when the numerical calculation
is unchanged.

## Portable replay

Install the inference extras, initialize the pinned submodule, and extract
the artifact bundle outside the Git worktree:

```bash
python -m pip install -e '.[mcmc,plot,validate]'
git submodule update --init --recursive
```

Then compile the experiment driver with:

```bash
python scripts/experiments/run_marisa_b_v0p9_matched_oracle.py \
  --repo-root "$PWD" \
  --data-root /path/to/marisa-b-v0p9-matched-oracle-artifacts-20260804 \
  --output-root /path/to/new-output \
  --stage compile
```

The full 27-bin analytic response is intentionally not rerun as part of the
release smoke test because it is the long-running production calculation.
Its successful raw records and manifests are preserved in the artifact.
