# Final ordinary-standard B000-only fNL model audit

This report contains only the halo B000 likelihood.  b1 is fitted by B000 with a flat hard-bounds prior; P0 supplies neither a likelihood term nor a parameter prior.

## Decision

The finite physical-K1 completion does not pass every frozen target gate at kmax_B=0.08.

Target profile: fNL=-8.83904, direct Delta-chi2=1 interval [-85.175, 70.6695], half-width=77.9223.

Target finite-response comparison: cosine=0.985246, LCp100/theory amplitude=0.479526, amplitude bootstrap 95%=[0.424706, 0.536269].

## Frozen model and implemented repair

The released Gaussian model is B_std = B_halo_tree(b1,b2,gamma2) + B_matter_1loop_rec + B_stochastic(Ashot,Bshot).  Its loop uplift is pure matter SPT rather than a full halo-bias one-loop expansion.

The physical response uses K1(k)=b1+fNL*bphi/M(k), bphi=2*1.686*(b1-1), while b_rec=1.9326526728794324 remains a fixed reconstruction input and is never differentiated.  For the already retained matter loops, the exact finite addition is sum_m=1^6 [fNL*bphi]^m C_m; no fitted response amplitude is introduced.

The Gaussian baseline passed its frozen gate, so the conditional mixed halo-bias loop and reconstructed stochastic repair branches were not activated.  Adding b2/tidal loop responses without matching Gaussian loops would change the v0.9 architecture and is outside this audit.

The b1 likelihood contract is flat hard-bounds prior on [0.1, 6.0], with Gaussian precision exactly 0.0.  Consequently every reported b1 value is inferred by B000 itself and no P0-derived centre or uncertainty enters.

## Gaussian baseline and residual projection

At kmax_B=0.08 the released baseline gives fNL=-7.03647, Fisher sigma=83.5917, pull=-0.0841766, and Gaussian GOF p=0.996599.

The nuisance-projected linear residual predicts Delta fNL=-7.91074, compared with the direct Minuit centre -7.03647; their difference is -0.874278.

The paired LCp100 response has only 2.56% derivative-noise information, while the released local response has cosine=0.961086 and LCp100/theory amplitude=0.429311.  This localizes the dominant mismatch to the PNG response rather than the Gaussian mean.

## Fid250/Fid250 TT-ET-TE-EE diagnostic at kmax_B=0.08

T denotes a theoretical Gaussian baseline or PNG response, and E denotes its held-out empirical diagnostic counterpart.  These combinations diagnose the failure only and are never used as a final correction.

| fold | combination | fNL centre | centre / Fisher sigma | profile GOF p |
|---|---|---:|---:|---:|
| fold_A_to_B | EE | -41.37507 | -0.41839 | 0.77453 |
| fold_A_to_B | ET | -42.54158 | -0.52770 | 0.80241 |
| fold_A_to_B | TE | -24.52940 | -0.24767 | 0.96011 |
| fold_A_to_B | TT | -25.01253 | -0.30314 | 0.96522 |
| fold_B_to_A | EE | 27.43366 | 0.26210 | 0.77195 |
| fold_B_to_A | ET | 26.43020 | 0.32142 | 0.77973 |
| fold_B_to_A | TE | 10.15116 | 0.09690 | 0.89275 |
| fold_B_to_A | TT | 7.64063 | 0.08717 | 0.89169 |

## B000-only profiles and response

| variant | kmax B | fNL centre | profile half-width | B000-fit b1 | Gaussian p | finite cosine | finite LCp100/theory amplitude | per-cut gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| released | 0.05 | 38.02378 | 150.00000 | 1.958852 | 0.98102 | 0.999996 | 0.410543 | False |
| released | 0.06 | 9.30316 | 135.02108 | 1.864919 | 0.88967 | 0.978063 | 0.415564 | False |
| released | 0.07 | 9.30316 | 135.02108 | 1.864919 | 0.88967 | 0.978063 | 0.415564 | False |
| released | 0.08 | -7.03647 | 82.10146 | 2.036801 | 0.9966 | 0.966067 | 0.388351 | False |
| released | 0.10 | 0.15679 | 68.71550 | 1.981613 | 0.99999 | 0.911504 | 0.328953 | False |
| released | 0.12 | 12.20239 | 50.41805 | 2.013690 | 0.99998 | 0.812270 | 0.334451 | False |
| physical_K1_loop | 0.05 | 37.51918 | 150.00000 | 1.960373 | 0.98102 | 0.999934 | 0.428239 | False |
| physical_K1_loop | 0.06 | 9.30226 | 131.89975 | 1.864702 | 0.88967 | 0.981375 | 0.442131 | False |
| physical_K1_loop | 0.07 | 9.30226 | 131.89975 | 1.864702 | 0.88967 | 0.981375 | 0.442131 | False |
| physical_K1_loop | 0.08 | -8.83904 | 77.92225 | 2.038824 | 0.9966 | 0.985246 | 0.479526 | False |
| physical_K1_loop | 0.10 | -3.20650 | 66.53023 | 1.986325 | 0.99999 | 0.926070 | 0.412715 | False |
| physical_K1_loop | 0.12 | 10.76773 | 45.10949 | 2.013524 | 0.99998 | 0.727042 | 0.338835 | False |

## Fid250/Fid250 physical-model validation

| held-out target / covariance half | fNL centre | 68% interval | profile GOF p |
|---|---:|---:|---:|
| physical_K1_loop__target000_249_cov250_499 | 4.29350 | [-80.75843, 90.55644] | 0.89002 |
| physical_K1_loop__target250_499_cov000_249 | -24.64023 | [-95.62552, 52.35493] | 0.96535 |
| released__target000_249_cov250_499 | 7.64063 | [-80.60499, 98.15446] | 0.89169 |
| released__target250_499_cov000_249 | -25.01253 | [-99.69057, 56.99837] | 0.96522 |

## Closure and scope

- The algebraic tree-kernel audit, Gaussian real-field closure, and physical-PNG real-field closure all pass.
- The Gaussian Fid500 baseline passed before response repair, so no mixed halo-bias loop, EFT term, or new stochastic parameter was added.
- The physical completion acts only on the pure-matter SPT loops already present in v0.9 and replaces the auxiliary marker by fNL*bphi through powers 1--6.
- The deterministic unit test and production canary verify marker powers 0--6, diagram sums, the serialized finite response against direct B(+100)-B(0) evaluation, and paired-bootstrap reproducibility; LCp100 uncertainty uses 4000 resamples of 100 paired realizations.
- No empirical response correction, DM backbone, adaptive/oracle denominator, STO field, MCMC, P-only likelihood, or P+B likelihood was used.

## Common-random smoothing diagnostic

| R [Mpc/h] | data-shift RMS [Mpc/h] | ||Bpost-Bpre|| / ||Bpre|| |
|---:|---:|---:|
| 10 | 3.508008 | 1.345598 |
| 15 | 3.083190 | 1.255631 |
| 20 | 2.757407 | 1.162761 |

This Fid10 scan establishes the expected monotonic smoothing control but is not an fNL likelihood or a substitute for the Fid500 result.

## Recommendation

Do not claim the ordinary-standard kmax_B=0.08 fNL constraint under the no-EFT/no-new-bias model. Use a separately validated lower B cut, increase the smoothing radius, or fall back to pre-reconstruction; do not tune an empirical response template to the Fid500 mean.

## Products

- Machine-readable audit: `/pscratch/sd/l/lzy/reconstruction-png/analysis/quijote_halo_z0p5_r10_standard_fnl_model_v1/FINAL_B000_AUDIT.json`
- Compact result table: `/pscratch/sd/l/lzy/reconstruction-png/analysis/quijote_halo_z0p5_r10_standard_fnl_model_v1/FINAL_B000_RESULTS.csv`
- PDF summary: `/pscratch/sd/l/lzy/reconstruction-png/figures/diagnostics/quijote_halo_z0p5_r10_standard_fnl_model_v1/quijote_halo_z0p5_r10_standard_fnl_model_v1_final_B000_model_audit.pdf`
- Physical-K1 detailed profiles: `/pscratch/sd/l/lzy/reconstruction-png/analysis/quijote_halo_z0p5_r10_standard_fnl_model_v1/physical_png_k1_loop/PHYSICAL_PNG_K1_LOOP_PROFILES.json`
