# fNL 固定 100、p 自由的 P0 校准（pre / STO-only）

Status: `complete`。`(p, b1, Pshot_residual)` 三参数，fNL 固定为输入真值 +100。
这是校准测量，无 recovery gate。

| method | p median [16,84] | b1 [16,84] | GoF p |
|---|---:|---:|---:|
| pre | 1.1789 [1.0862,1.2695] | 1.94458 [1.92882,1.96034] | 0.6942 |
| sto_only | 1.3395 [1.2387,1.4432] | 2.21193 [2.19461,2.22942] | 0.6087 |

结果 JSON：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_pre_sto_only_fnlfixed100_freep_calib_v1/FREEP_RESULTS.json`。
posterior NPZ：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_pre_sto_only_fnlfixed100_freep_calib_v1/FREEP_POSTERIORS.npz`。
