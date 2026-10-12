#!/usr/bin/env python3
"""K̂_knot 标定：裸缓存 vs FKP 产品，bins2-12 平均（与 probe_k_universal 同法）。"""
import json
from pathlib import Path

import numpy as np

V2 = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
CACHE = V2 / "envclass_p0only/fullspec/bare_auto_cache"
FKP = V2 / "envclass_p0only/full"
NM = np.asarray([26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018])

out = {}
for cosmo in ("fiducial", "LC_p"):
    k_hats, n_list = [], []
    for real in range(100):
        b = CACHE / cosmo / f"real{real:03d}/k.npy"
        f = FKP / cosmo / f"real{real:03d}/knot_pk.npz"
        if not (b.is_file() and f.is_file()):
            continue
        bare = np.load(b)
        with np.load(f) as z:
            pk = np.asarray(z["pk"], dtype=np.float64)
            norm = float(np.asarray(z["norm"])[0])
            ns = np.asarray(z["num_shotnoise"], dtype=np.float64)
        n_c = np.sqrt(norm * 1.0e9)
        sn = ns / norm
        k_hats.append((pk + sn) * NM * n_c**2 / bare)
        n_list.append(n_c)
    if not k_hats:
        print(f"{cosmo}: no knot cache yet")
        continue
    k_hats = np.asarray(k_hats)
    per_real = np.mean(k_hats[:, 1:], axis=1)
    print(f"{cosmo}: nreal={k_hats.shape[0]}  n_knot={np.mean(n_list):.0f}  "
          f"K={np.mean(per_real):.4e}  scatter={np.std(per_real)/np.mean(per_real)*100:.2f}%")
    print(f"  per-bin: {np.array2string(np.mean(k_hats, axis=0), precision=3)}")
    out[cosmo] = {"n_real": int(k_hats.shape[0]), "K_mean": float(np.mean(per_real)),
                  "scatter": float(np.std(per_real) / np.mean(per_real)),
                  "per_bin": np.mean(k_hats, axis=0).tolist()}

save = V2 / "envclass_p0only/fullspec/probe_k_knot_result.json"
save.write_text(json.dumps(out, indent=2))
print(f"saved: {save}")
