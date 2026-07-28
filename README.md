# MARISA-B

MARISA-B is a real-space halo-bispectrum model and inference pipeline for
Quijote-like measurements.  The numerical core is C++17; Python supplies
data contracts, template assembly, profiling, MCMC, Fisher diagnostics, and
figures.

## Current model

The selected model is
`halo_eft_v2_coevolution_finite_png`:

- Gaussian halo one-loop EFT-v2 with
  `B222 + B321I + B321II + B411`, the full counterterm basis, and stochastic
  terms;
- the coevolution restriction as the production bias tier;
- the unrestricted `full` tier as a mandatory cross-check;
- pre- and post-reconstruction field kernels;
- local-PNG halo tree terms through
  `G + fNL L + fNL^2 Q`;
- the matter local-PNG linear one-loop uplift and the matter
  `B112II fNL^2` contribution.

The historical Gaussian `uplift` tier is retired.  It remains in a legacy
adapter only because frozen analyses import a few of its helper functions.
It is not a selectable production model.

There is exactly one native MARISA-B core:
`src/marisa_b/marisa_b_native.cpp`.  Gaussian, PNG, halo, reconstruction,
drivers, and tests compile against the same header and object, eliminating
the former cross-translation-unit ABI mismatch.

## Scientific status

Code integration and scientific acceptance are separate gates.

- The pre-reconstruction finite-PNG coevolution model has existing
  single-realization-covariance injection/recovery evidence with profiled
  `b1`; `full` is retained as a cross-check.
- The post-reconstruction Gaussian and finite-PNG implementation is
  integrated, but its expensive numerical convergence campaign is not
  complete.  It is therefore **provisional**, not a production-accepted
  cosmology result.
- Current halo post-reconstruction validation is restricted to the fiducial
  `fNL=0` catalogue because the halo PNG reconstruction inputs are not yet
  trusted.  This restriction does not apply to the separately audited dark
  matter PNG reconstruction catalogues.

Every current uncertainty, pull, goodness-of-fit decision, and plotted error
bar must use the covariance of one equivalent realization.  Ensemble means
are central vectors only.  A likelihood configuration must explicitly choose
`none`, `hartlap`, or `student_t` precision treatment; there is no implicit
covariance-of-the-mean fallback.

See [Scientific status](docs/SCIENTIFIC_STATUS.md) and
[Inference contract](docs/INFERENCE_CONTRACT.md) for the exact boundary.

## Fresh-clone build

System prerequisites are Git with submodule support, GNU Make, a C++17
compiler with OpenMP, GSL, `pkg-config`, `ripgrep`, `sha256sum`, and Python
3.10 or newer.  Initialize the pinned dependency and install the minimal test
environment before invoking Make:

```bash
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[validate]'
make -j28 release-test
```

`release-test` runs the C++ and synthetic Python suites, builds and smokes the
optional native triangle CLI, verifies the unique canonical native object,
and applies the strict clean-tree source/provenance boundary.  It does not
require Quijote data.

If GSL is not installed in a system location, pass its prefix explicitly:

```bash
make -j28 release-test GSL_PREFIX=/path/to/gsl-prefix
```

No target may request more than 28 CPU threads.

## History-clean public bundle

The private integration ancestry is not a public release boundary.  From a
clean verified integration branch, create a new two-commit repository and a
bundle containing only its exact release branch and tag:

```bash
python scripts/build_public_release.py \
  --destination /path/to/marisa-b-release \
  --bundle /path/to/marisa-b-v0.1.0-rc1.bundle
```

The first public commit contains the normalized source tree.  The second
adds a first-party import ledger whose commit and blob hashes are verifiable
inside that two-commit history.  The command never configures or pushes a
remote.

## Python workflow extras

Install only the extras needed for a production workflow:

```bash
python -m pip install -e '.[profile,plot,validate]'
python -m pip install -e '.[mcmc,plot,validate]'
```

The package-only checks can also be run directly:

```bash
python -m pytest -q tests/python
python -m compileall -q python scripts tests
```

## External production data

Large measurements, theory caches, MCMC chains, plots, and logs are not
stored in source Git.  Set `MARISA_B_DATA_ROOT` to the external bundle and
verify every registered asset before a production run:

```bash
export MARISA_B_DATA_ROOT=/path/to/marisa-b-data
python scripts/verify_production_data.py
```

`configs/production_data_manifest.json` records relative URIs, sizes, and
SHA256 identities.  The source repository never infers a portable parent
directory or a historical worktree.

`configs/inference_prepost_example.json` demonstrates the current mandatory
choices: coevolution, free `b1` initialized at its fiducial value, `p=1`,
the hard `fNL` prior `[-150, 150]`, single-realization covariance, and an
explicit precision correction.

After the native release test, the registered external profiler/MCMC/Fisher
regression can be run with:

```bash
make -j28 external-inference-regression
```

This verifies the nine production assets, four frozen result identities,
joint-MAP centres, current pre/post likelihood closure, and deterministic
Fisher matrices.  It does not rerun the full converged MCMC: the frozen MCMC
summary is hash- and field-checked, while the current collapsed likelihood is
recomputed.  Exact anchors and tolerances live in
`tests/data/inference_regression_v1.json`.

## Repository map

- `src/marisa_b/`: canonical matter, local-PNG, finite-PNG, and
  reconstruction core;
- `src/halo_v1/`: halo field kernels and shell averaging;
- `src/eft_v2/`: halo one-loop EFT-v2 diagrams, bias basis, counterterms,
  stochastic terms, reconstruction, and projectors;
- `python/marisa_b/`: path, parameter, and statistical APIs;
- `scripts/production/`: frozen analysis adapters and production entry
  points;
- `tests/`: C++ and synthetic Python regression tests;
- `configs/` and `schemas/`: model, inference, and external-data contracts;
- `provenance/`: first- and third-party source identities;
- `docs/historical/`: explicitly non-current R0 and bias-v1 records.

Generated files belong in `analysis/`, `figures/`, `manifests/`, `log/`, or
external release storage; these locations are intentionally excluded from
source Git.

## License and attribution

MARISA-B is distributed under GPL-3.0-only.  The pinned ReACT reactions
subtree and the OneLoopBispectrum-derived generated tables impose compatible
GPL terms.  See `LICENSE`, `THIRD_PARTY_NOTICES.md`, and
`provenance/third_party_sources.tsv`.
