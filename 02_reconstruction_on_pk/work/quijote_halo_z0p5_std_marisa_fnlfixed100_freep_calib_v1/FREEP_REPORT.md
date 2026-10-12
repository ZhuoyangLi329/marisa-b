# fNL 固定 100、p 自由的 P0 校准（pre / STO-only）

Status: `complete`。`(p, b1, Pshot_residual)` 三参数，fNL 固定为输入真值 +100。
这是校准测量，无 recovery gate。

| method | p median [16,84] | b1 [16,84] | GoF p |
|---|---:|---:|---:|
| standard | 1.1218 [1.0312,1.2125] | 1.93544 [1.92028,1.95043] | 0.3761 |
| marisa | 1.1235 [1.0338,1.2123] | 1.93651 [1.92134,1.95185] | 0.3952 |

结果 JSON：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_std_marisa_fnlfixed100_freep_calib_v1/FREEP_RESULTS.json`。
posterior NPZ：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_std_marisa_fnlfixed100_freep_calib_v1/FREEP_POSTERIORS.npz`。
