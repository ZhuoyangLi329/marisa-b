# p 固定校准值的三方法 P0-only 主结果

Status: `complete`。p_pre=1.1789、p_sto=1.3395 固定（fNL=100 校准中位数）。
+100 行为同数据自洽检查（校准用了同一 +100 样本），非独立无偏检验。

| target | method | fNL median [16,84] | half-width | width/pre | truth in 68/95 |
|---|---|---:|---:|---:|---:|
| fNL0 | pre | -1.631 [-13.431,10.346] | 11.888 | 1.0000 | True/True |
| fNL0 | STO only | -0.636 [-12.043,11.123] | 11.583 | 0.9743 | True/True |
| fNL0 | pre+STO only (joint) | -1.175 [-12.821,10.821] | 11.821 | 0.9943 | True/True |
| fNL100 | pre | 100.568 [88.480,112.759] | 12.140 | 1.0000 | True/True |
| fNL100 | STO only | 100.084 [88.566,111.925] | 11.679 | 0.9621 | True/True |
| fNL100 | pre+STO only (joint) | 99.610 [87.884,111.403] | 11.759 | 0.9687 | True/True |

Covariance：单通道 12-bin 各自 Fid500（/42.875/Hartlap12）；joint 24-bin 全 cross（/42.875/Hartlap24）。

结果 JSON：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_fixedp3method_p0only_fid0_p100matched100_mcmc_v1/FIXEDP_RESULTS.json`。
posterior NPZ：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_fixedp3method_p0only_fid0_p100matched100_mcmc_v1/FIXEDP_POSTERIORS.npz`。
