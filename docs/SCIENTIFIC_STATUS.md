# Scientific status

Updated: 2026-09-29

## Production selection

The production Gaussian tier is `coevolution`.  It evaluates the same halo
one-loop diagrams, counterterms, and stochastic basis as `full`, while fixing
five higher-order bias coefficients with the registered coevolution
relations.  `full` remains a mandatory cross-check.  The old Gaussian
`uplift` approximation is not run.

The finite local-PNG prediction is assembled as

```text
B_h(fNL) =
    B_h,Gaussian^one-loop
  + fNL  L_h,tree
  + fNL² Q_h,tree
  + fNL  (L_m,one-loop - L_m,tree)
  + fNL² Q_m,B112II .
```

This is accurately described as a Gaussian halo one-loop baseline plus a
complete finite halo-tree local-PNG sector and the implemented matter-PNG
one-loop uplift.  It must not be described as a complete halo local-PNG
one-loop EFT calculation.

The universality closure is fixed at `p=1`.  `b1` is a free inference
parameter initialized at the fiducial power-spectrum value.  The default
finite-PNG inference prior is hard-bounded to `-150 <= fNL <= 150`.

## Pre reconstruction

The current finite-PNG pre-reconstruction implementation has been tested on
the fiducial and matched `fNL=+100/-100` halo samples using the full
two-dimensional `B000(k1,k2)` selection.  Existing acceptance plots and
profiles use a covariance of one equivalent realization, not the covariance
of the ensemble mean.

The current source migration preserves those numerical implementations and
their input hashes.  Re-running them with a different precision correction is
a new analysis version and must not silently replace the frozen result.

## Post reconstruction

Post-reconstruction Gaussian EFT-v2, the finite halo-tree PNG sector, the
matter-PNG uplift, and the reconstruction shift-field response are present in
the source tree.  The expensive convergence audit for every selected bin has
not completed, so the post model remains
`provisional_unvalidated_numerical_convergence`.

For halo catalogues, only the fiducial `fNL=0` post-reconstruction sample is
currently admitted for Gaussian model validation.  The `fNL=+100/-100` halo
catalogues are quarantined because their PNG reconstruction procedure is
suspect.  Dark-matter PNG reconstruction catalogues have a separate audit and
are not covered by that quarantine.

No README, tag, or release may upgrade this status merely because the code
builds or a short profiler/MCMC smoke test completes.

## Local-PNG adaptive reconstruction tree extension

The fixed-`b_rec` pre- and post-reconstruction local-PNG tree formulas retain
their existing `G + fNL L + fNL^2 Q` definitions.  The source additionally
exports three shell-projectable bases for an adaptive reconstruction
denominator,

```text
b_rec(k; fNL_rec) = b_rec + fNL_rec b_phi_rec / M(k):
    gaussian_linear,
    gaussian_quadratic,
    png_linear_cross.
```

The quadratic Gaussian basis and the local-PNG cross basis complete the
adaptive tree expansion through second order when `fNL_rec = fNL`.  Their
native regression tests compare the analytic coefficients with independent
finite differences.  This extension contains neither matter-loop terms nor a
stochastic closure and does not change the provisional status of the
post-reconstruction model.

The production adapters now read `b_rec_h` from the validated cache contract,
allow an explicit smoothing radius and reconstruction bias during cache
compilation, and support an explicit local-PNG `bphi` override together with
an explicit `preserve_universal_bphidelta` switch controlling the universality
anchor used for `bphidelta`.

## Statistical convention

- An ensemble mean is a low-noise estimate of the central data vector.
- `covariance_single = cov(realization rows, ddof=1)` defines the error scale
  for a survey volume equivalent to one simulation box.
- `covariance_mean = covariance_single / N` is available only as an explicitly
  named diagnostic and is forbidden for production adequacy or forecast
  claims.
- Precision treatment is a separate explicit choice:
  `none`, `hartlap`, or `student_t`.
- Pre/post comparisons must use the same registered precision treatment.
- Profiler curvature is not a substitute for posterior uncertainty; MCMC
  reports carry the accepted uncertainty when posterior inference is claimed.

Historical R0 documents under `docs/historical/` used a covariance-of-the-mean
closure test.  They are retained only to explain the code lineage.
