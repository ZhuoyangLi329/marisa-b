#!/usr/bin/env python3
"""口径因子剖析：裸涂场自谱缓存 vs FKP 自谱产品，逐 bin 差值判定。

判定逻辑
========
- 若 diff = P_bare - P_FKP 逐 bin 近似常数，且 ≈ V/n_c（逐类随 1/n_c
  缩放）→ 口径差 = 纯 shot noise 常数项，sn0_c 可吸收，无需投影。
- 若 diff 有光滑 k 形状且逐类形状相同 → 几何补偿因子，需把模板投影
  到裸口径（乘法因子 r(k)）。
输出：每类 n_mean、V/n、diff/(V/n) 12 bin、bare/FKP 比值。
"""
import json
from pathlib import Path

import numpy as np

PART = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/envclass_p0only")
CACHE = PART / "fullspec/bare_auto_cache"
FULL = PART / "full"
STO_SOURCE = Path(
    "/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_source"
)
CLS = [("void", "v", 0), ("sheet", "s", 1), ("filament", "f", 2), ("knot", "k", 3)]


def main():
    header = f"{'class':<10}{'n_mean':>9}{'V/n':>11}{'flat(std/mean)':>16}"
    print(header)
    payload = {}
    for name, short, cid in CLS:
        bare_list, fkp_list = [], []
        n_sum, cnt = 0, 0
        for real in range(100):
            b = CACHE / f"fiducial/real{real:03d}/{short}.npy"
            f = FULL / f"fiducial/real{real:03d}/{name}_pk.npz"
            if not (b.is_file() and f.is_file()):
                continue
            bare_list.append(np.load(b))
            with np.load(f) as z:
                fkp_list.append(np.asarray(z["pk"], dtype=np.float64))
            # metadata.json 被 knot 补跑覆盖过，类计数直接从 env_labels 数。
            labels = np.load(STO_SOURCE / f"fiducial/real{real:03d}/lam1_dm_nenv/env_labels.npy")
            n_sum += int(np.sum(labels == cid))
            cnt += 1
        bare = np.mean(bare_list, axis=0)
        fkp = np.mean(fkp_list, axis=0)
        diff = bare - fkp
        ratio = bare / fkp
        n_mean = n_sum / cnt
        vn = 1.0e9 / n_mean
        flatness = float(np.std(diff) / np.mean(diff))
        print(f"{name:<10}{n_mean:>9.0f}{vn:>11.3g}{flatness:>16.3f}   (n_real={cnt})")
        print(f"   diff/(V/n)   : {np.array2string(diff / vn, precision=3, floatmode='fixed')}")
        print(f"   bare/FKP     : {np.array2string(ratio, precision=2, floatmode='fixed')}")
        print(f"   FKP pk[:4]   : {np.array2string(fkp[:4], precision=1)}")
        print(f"   diff pk[:4]  : {np.array2string(diff[:4], precision=1)}")
        payload[name] = {
            "n_mean": n_mean, "V_over_n": vn, "n_real": cnt,
            "diff": diff.tolist(), "ratio": ratio.tolist(),
            "flatness_std_over_mean": flatness,
        }
    out = PART / "fullspec" / "probe_caliber_result.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
