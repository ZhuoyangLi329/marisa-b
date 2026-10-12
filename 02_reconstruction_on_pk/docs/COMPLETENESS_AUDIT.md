# Completeness audit

Present and sufficient for the documented questions:

- Sample definition and counts: README and results/four_method_p0_matrix_audit.json.
- P0 extraction logic: first-part common/halo_b000_pipeline.py; this part keeps the aggregate reader in code/aggregate_four_method_p0.py.
- P0 model and Fiducial bias calibration: code/fit_halo_pk_bias.py, inputs/fiducial_pk_bias_fit_input.npz, and results/fiducial_pk_bias_fit_summary.json.
- Four-method P0-only constraint: code/run_four_method_p0only_mcmc.py, its result JSON/CSV, report and two PDFs.
- Independent checks: code/audit_four_method_p0only_mcmc.py and matrix audit JSON.
- P+B cross-check and five-method Fisher comparison: the joint reports, Fisher JSON and corresponding PDFs.

Deliberately not copied:

The P0-only runner has transitive imports from old discrepancy/template analysis modules, and the aggregate runner expects full per-realization result trees. Those are implementation dependencies for a future clean rerun, not required to understand the method or audit the completed results. They are not silently copied because doing so would reintroduce a large, path-coupled legacy analysis tree.

The small P0 matrix, finite-PNG theory asset, preregistration, and bias-fit input are under inputs/. A fully standalone v2 MCMC execution still needs a later refactor of transitive analysis imports and a new output contract under work/; this is recorded explicitly.

Decision: for the four documented questions, the migrated set is complete. For an independent end-to-end rerun, it is intentionally incomplete at the transitive dependency layer and should be refactored separately.
