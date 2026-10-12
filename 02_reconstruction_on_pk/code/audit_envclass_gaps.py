#!/usr/bin/env python3
"""盘点 fid/LC_p 0-99 内自谱产品、裸缓存、互谱的逐套缺口。"""
import json
from pathlib import Path

V2 = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
FULL = V2 / "envclass_p0only/full"
CACHE = V2 / "envclass_p0only/fullspec/bare_auto_cache"
CLS = [("void", "v"), ("sheet", "s"), ("filament", "f")]
PAIRS = [("void", "sheet"), ("void", "filament"), ("sheet", "filament")]

gaps = {"fkp": [], "bare": [], "cross": []}
for cosmo in ("fiducial", "LC_p"):
    for real in range(100):
        tag = f"{cosmo}/real{real:03d}"
        for name, short in CLS:
            if not (FULL / cosmo / f"real{real:03d}" / f"{name}_pk.npz").is_file():
                gaps["fkp"].append(f"{tag}/{name}")
            if not (CACHE / cosmo / f"real{real:03d}" / f"{short}.npy").is_file():
                gaps["bare"].append(f"{tag}/{short}")
        for a, b in PAIRS:
            if not (FULL / cosmo / f"real{real:03d}" / f"cross_{a}{b}_pk.npz").is_file():
                gaps["cross"].append(f"{tag}/{a}{b}")

for kind, items in gaps.items():
    print(f"{kind}: {len(items)} missing")
    for it in items:
        print(f"   {it}")
out = V2 / "envclass_p0only/fullspec/gap_audit.json"
out.write_text(json.dumps(gaps, indent=2))
print(f"saved: {out}")
