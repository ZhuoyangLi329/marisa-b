# pre+post 联合 P0-only MCMC（共享 fNL）

Status: `complete`。5 参数 `(fNL, b1_pre, b1_post, sn0_pre, sn0_post)`，24-bin 联合向量、全 cross-covariance。

| target | pair | fNL median [16,84] | half-width | width/pre | post-only width/pre | truth in 68/95 |
|---|---|---:|---:|---:|---:|---:|
| fNL0 | pre+STO only | 0.098 [-7.255,7.540] | 7.398 | 0.6671 | 0.8375 | True/True |
| fNL100 | pre+STO only | 49.843 [42.911,56.753] | 6.921 | 0.6174 | 0.8062 | False/False |

Covariance：联合 Fid500 单盒 24x24（含 pre--post cross 块），除以 42.875 和 Hartlap_24；target mean 不除以100。

post-only/pre 对照列取冻结五方法数字（未重跑）。

结果 JSON：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_pre_post_joint_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_RESULTS.json`。
posterior NPZ：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_pre_post_joint_p0only_fid0_p100matched100_mcmc_v1/P0ONLY_MCMC_POSTERIORS.npz`。
PDF：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/figures/pre_post_joint_p0only_mcmc_contours.pdf`、`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/figures/pre_post_joint_p0only_recovery_comparison.pdf`。
