# fNL 固定 100、p 自由的 P0 校准（pre / STO-only）

Status: `complete`。`(p, b1, Pshot_residual)` 三参数，fNL 固定为输入真值 +100。
这是校准测量，无 recovery gate。

| method | p median [16,84] | b1 [16,84] | GoF p |
|---|---:|---:|---:|
| sto_marisa | 1.2828 [1.1853,1.3829] | 2.21789 [2.20139,2.23459] | 0.3484 |

结果 JSON：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_sto_marisa_fnlfixed100_freep_calib_v1/FREEP_RESULTS.json`。
posterior NPZ：`/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/work/quijote_halo_z0p5_sto_marisa_fnlfixed100_freep_calib_v1/FREEP_POSTERIORS.npz`。
