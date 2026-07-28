# Inference contract

Every new profiler, MCMC, or Fisher run must serialize and validate the
following choices before evaluating a likelihood.

## Model coordinates

- `gaussian_tier`: `coevolution` for production or `full` for cross-validation;
- `reconstruction`: `pre` or `post`;
- `png_model`: `finite_halo_tree_plus_matter_one_loop_uplift`;
- `png_order`: `quadratic`;
- the explicit PNG stochastic treatment;
- `p=1`;
- free `b1`, initialized at the registered fiducial value;
- free `fNL` with the registered hard prior.

The model prediction is decomposed into Gaussian, linear-PNG, and
quadratic-PNG vectors.  Tests must verify both the components and
`G + fNL L + fNL² Q` at `fNL=0,+100,-100`.

## Data and masks

Each asset is addressed relative to `MARISA_B_DATA_ROOT` and verified against
`configs/production_data_manifest.json`.  A run records the manifest hash,
asset hashes, reconstruction label, bin ordering, `kmin`, `kmax`, excluded
first-bin flag, and excluded shell centres.

The current full-2D selection excludes the first bin and the shell centred at
approximately `0.1553916 h/Mpc`.  A changed mask is a new run contract.

## Covariance and precision

`covariance_kind` must equal `single_realization`.  The covariance is computed
from realization rows with `ddof=1`; no implicit division by the realization
count is permitted.

`precision_correction` must be one of:

- `none`: invert the sample covariance directly;
- `hartlap`: multiply the inverse by
  `(Nmock - dimension - 2)/(Nmock - 1)`;
- `student_t`: use the registered finite-mock Student-t likelihood rather than
  a Gaussian precision rescaling.

The number of mocks and the final selected dimension are part of the run
manifest.  Hartlap must be recomputed after the final mask is known.

## Output status

Outputs distinguish:

- source/build verification;
- numerical-template convergence;
- likelihood smoke completion;
- scientific injection/recovery acceptance.

Passing an earlier gate never implies a later gate.  Post-reconstruction
outputs remain provisional until their registered numerical convergence and
fiducial-halo validation gates pass.
