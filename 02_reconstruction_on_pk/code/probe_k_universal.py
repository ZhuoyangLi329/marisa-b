#!/usr/bin/env python3
"""K 普适性验证：K = C_c*n_c^2 跨 realization / 跨 cosmology 是否恒定。

背景
====
单套（fid real000）分解实验确立转换公式：
  P_FKP,c(k) = K/(n_c^2 * nmodes(k)) * P_bare,c(k) - sn_c,  K ~= 7.94e10
其中 n_c 逐 realization 变（FKP 产品 npz 的 norm = n^2/V 可反推 n_c）。
本脚本对 fiducial/LC_p 各 100 套、v/s/f 三类，逐套逐 bin 计算
  K_hat = (P_FKP + ns/norm) * nmodes * n^2 / P_bare
输出 K 的全局均值、scatter、逐类/逐 bin 结构，判定 K 是否可作
universal 转换常数。
"""
import json
from pathlib import Path

import numpy as np

V2 = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
CACHE = V2 / "envclass_p0only/fullspec/bare_auto_cache"
FKP_ROOT = V2 / "envclass_p0only/full"
CLS = [("void", "v"), ("sheet", "s"), ("filament", "f")]
NM = np.asarray([26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018])


def main():
    payload = {}
    for cosmo in ("fiducial", "LC_p"):
        for name, short in CLS:
            k_hats = []
            n_list = []
            for real in range(100):
                b = CACHE / cosmo / f"real{real:03d}/{short}.npy"
                f = FKP_ROOT / cosmo / f"real{real:03d}/{name}_pk.npz"
                if not (b.is_file() and f.is_file()):
                    continue
                bare = np.load(b)
                with np.load(f) as z:
                    pk = np.asarray(z["pk"], dtype=np.float64)
                    norm = float(np.asarray(z["norm"])[0])
                    ns = np.asarray(z["num_shotnoise"], dtype=np.float64)
                n_c = np.sqrt(norm * 1.0e9)
                sn = ns / (norm * np.ones(12))
                k_hats.append((pk + sn) * NM * n_c**2 / bare)
                n_list.append(n_c)
            k_hats = np.asarray(k_hats)
            n_arr = np.asarray(n_list)
            # 每套先在 bins2-12 内平均（压 bin 噪声），再统计跨套分布
            per_real = np.mean(k_hats[:, 1:], axis=1)
            payload[f"{cosmo}/{name}"] = {
                "n_real": int(k_hats.shape[0]),
                "n_mean": float(np.mean(n_arr)),
                "n_std_over_mean": float(np.std(n_arr) / np.mean(n_arr)),
                "K_mean": float(np.mean(per_real)),
                "K_std_over_mean": float(np.std(per_real) / np.mean(per_real)),
                "K_binmean": np.mean(k_hats, axis=0).tolist(),
                "K_bin_std": (np.std(k_hats, axis=0) / np.mean(k_hats, axis=0)).tolist(),
            }
            print(f"{cosmo}/{name}: nreal={k_hats.shape[0]}  n={np.mean(n_arr):.0f}±{np.std(n_arr)/np.mean(n_arr)*100:.1f}%  "
                  f"K={np.mean(per_real):.4e}  scatter={np.std(per_real)/np.mean(per_real)*100:.2f}%")
            print(f"   K per-bin (mean over reals): {np.array2string(np.mean(k_hats, axis=0), precision=3)}")

    allk = [v["K_mean"] for v in payload.values()]
    print(f"\n全局 K 范围: {min(allk):.4e} ~ {max(allk):.4e}  (spread={(max(allk)-min(allk))/np.mean(allk)*100:.2f}%)")
    out = V2 / "envclass_p0only/fullspec/probe_k_universal_result.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
