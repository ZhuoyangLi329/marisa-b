# 五方法 P0-only matched100 MCMC（STO-only 扩展）

Status: `complete`。四方法行复制自冻结 v1（未重跑）；`sto_only` 为新运行。

| target | method | fNL median [16,84] | half-width | b1 [16,84] | width/pre | truth in 68/95 | source |
|---|---|---:|---:|---:|---:|---:|---|
| fNL0 | pre | -1.569 [-12.695,9.485] | 11.090 | 1.95809 [1.94133,1.97502] | 1.0000 | True/True | frozen_v1 |
| fNL0 | standard post | 4.356 [-6.533,15.826] | 11.179 | 1.94907 [1.93250,1.96526] | 1.0081 | True/True | frozen_v1 |
| fNL0 | Marisa post | 4.607 [-6.252,15.769] | 11.010 | 1.94903 [1.93262,1.96540] | 0.9928 | True/True | frozen_v1 |
| fNL0 | STO only | -0.546 [-9.617,8.960] | 9.288 | 2.22347 [2.20450,2.24221] | 0.8375 | True/True | five_method_new |
| fNL0 | STO+Marisa | 6.495 [-2.567,15.707] | 9.137 | 2.22805 [2.21023,2.24592] | 0.8239 | True/True | frozen_v1 |
| fNL100 | pre | 93.026 [81.980,104.400] | 11.210 | 1.94422 [1.92851,1.96044] | 1.0000 | True/True | frozen_v1 |
| fNL100 | standard post | 100.021 [88.934,111.260] | 11.163 | 1.93528 [1.92000,1.95040] | 0.9958 | True/True | frozen_v1 |
| fNL100 | Marisa post | 99.949 [88.899,110.960] | 11.031 | 1.93607 [1.92094,1.95130] | 0.9840 | True/True | frozen_v1 |
| fNL100 | STO only | 79.952 [71.003,89.078] | 9.038 | 2.21154 [2.19408,2.22931] | 0.8062 | False/False | five_method_new |
| fNL100 | STO+Marisa | 85.381 [76.488,94.323] | 8.917 | 2.21736 [2.20069,2.23410] | 0.7955 | False/True | frozen_v1 |

Covariance：每种方法自己的 fNL=0 Fid500 单盒 covariance，除以 42.875 和 Hartlap；target mean 不除以100。

四方法冻结来源：`results/four_method_p0only_mcmc_results.json` 与旧仓库 v1 posterior NPZ（sha256 已校验）。

结果 JSON：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_five_method_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_RESULTS.json`。
posterior NPZ：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_five_method_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_POSTERIORS.npz`。
PDF：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/figures/five_method_p0only_mcmc_contours.pdf`、`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/figures/five_method_p0only_recovery_comparison.pdf`。
