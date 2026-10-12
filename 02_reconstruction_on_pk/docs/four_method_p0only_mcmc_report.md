# 四方法 P0-only matched100 MCMC

Status: `complete`。

| target | method | fNL median [16,84] | half-width | b1 [16,84] | width/pre | truth in 68/95 |
|---|---|---:|---:|---:|---:|---:|
| fNL0 | pre | -1.569 [-12.695,9.485] | 11.090 | 1.95809 [1.94133,1.97502] | 1.0000 | True/True |
| fNL0 | standard post | 4.356 [-6.533,15.826] | 11.179 | 1.94907 [1.93250,1.96526] | 1.0081 | True/True |
| fNL0 | Marisa post | 4.607 [-6.252,15.769] | 11.010 | 1.94903 [1.93262,1.96540] | 0.9928 | True/True |
| fNL0 | STO+Marisa | 6.495 [-2.567,15.707] | 9.137 | 2.22805 [2.21023,2.24592] | 0.8239 | True/True |
| fNL100 | pre | 93.026 [81.980,104.400] | 11.210 | 1.94422 [1.92851,1.96044] | 1.0000 | True/True |
| fNL100 | standard post | 100.021 [88.934,111.260] | 11.163 | 1.93528 [1.92000,1.95040] | 0.9958 | True/True |
| fNL100 | Marisa post | 99.949 [88.899,110.960] | 11.031 | 1.93607 [1.92094,1.95130] | 0.9840 | True/True |
| fNL100 | STO+Marisa | 85.381 [76.488,94.323] | 8.917 | 2.21736 [2.20069,2.23410] | 0.7955 | False/True |

Covariance：每种方法自己的 fNL=0 Fid500 单盒 covariance，除以 42.875 和 Hartlap；target mean 不除以100。

结果 JSON：`/pscratch/sd/l/lzy/reconstruction-png/analysis/quijote_halo_z0p5_pre_std_marisa_sto_marisa_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_RESULTS.json`。
posterior NPZ：`/pscratch/sd/l/lzy/reconstruction-png/analysis/quijote_halo_z0p5_pre_std_marisa_sto_marisa_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_POSTERIORS.npz`。
PDF：`/pscratch/sd/l/lzy/reconstruction-png/figures/diagnostics/quijote_halo_z0p5_pre_std_marisa_sto_marisa_p0only_fid0_p100matched100_mcmc_v1/four_method_p0only_mcmc_contours.pdf`、`/pscratch/sd/l/lzy/reconstruction-png/figures/diagnostics/quijote_halo_z0p5_pre_std_marisa_sto_marisa_p0only_fid0_p100matched100_mcmc_v1/four_method_p0only_recovery_comparison.pdf`。
