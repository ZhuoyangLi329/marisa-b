#!/usr/bin/env python3
"""测量环境类互功率谱 P_cd(k)（c≠d，无 shot noise、无随机样本）。

代码大纲
========
1. 读取 halo 位置与环境标签，按请求的类对 (c,d) 拆成两个 catalog。
2. 同一几何（mesh 256、12 bins、kmax 0.08、自建锚点 v1）；互谱估计
   直接用两类涂场 FFT 交叉（compute_box_mesh2_spectrum 的多 catalog
   路径），无随机样本（互 shot noise 理论为 0）。
3. 输出 cross_{c}{d}_pk.npz + metadata + DONE 增量更新。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from mpi4py import MPI


OLD_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
CODES = OLD_ROOT / "codes"
V2_ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
HALO_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions")
STO_SOURCE = OLD_ROOT / "result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_source"
OUTPUT_ROOT = V2_ROOT / "02_reconstruction_on_pk/envclass_p0only/full"

SCHEMA = "quijote-halo-z0p5-envclass-crosspk-product-v1"
CLASS_IDS = {"void": 0, "sheet": 1, "filament": 2, "knot": 3}
PK_KMIN = 0.006283185307179587
PK_KMAX = 0.08
PK_KSTEP = 0.006283185307179587
PK_MESH = 256
SELF_GEOMETRY = {
    "k": [0.0089, 0.016257, 0.022057, 0.027837, 0.034579, 0.040668,
          0.046767, 0.053235, 0.059556, 0.065895, 0.072131, 0.078453],
    "nmodes": [26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018],
    "note": "self-consistent geometry anchor v1",
}


def import_path(name: str, path: Path) -> Any:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmology", choices=("fiducial", "LC_p"), required=True)
    parser.add_argument("--realization", type=int, required=True)
    parser.add_argument("--pairs", required=True,
                        help="逗号分隔类对，如 vs,vf,sf（v=void s=sheet f=filament k=knot）")
    args = parser.parse_args()

    SHORT = {"v": "void", "s": "sheet", "f": "filament", "k": "knot"}
    pairs = []
    for token in args.pairs.split(","):
        token = token.strip()
        if len(token) != 2 or token[0] not in SHORT or token[1] not in SHORT:
            raise ValueError(f"bad pair token: {token}")
        pairs.append((SHORT[token[0]], SHORT[token[1]]))

    tools = import_path("envcross_tools", CODES / "run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py")

    boxsize = float(tools.QUIJOTE_BOXSIZE_MPC_H)
    mpicomm = MPI.COMM_SELF
    real = int(args.realization)
    position_path = HALO_ROOT / args.cosmology / f"real{real:03d}" / "positions_mpc_h.npy"
    label_path = STO_SOURCE / args.cosmology / f"real{real:03d}" / "lam1_dm_nenv" / "env_labels.npy"
    for path in (position_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    positions = tools.load_halo_positions(position_path, dtype=np.float32, max_rows=None)
    labels = np.load(label_path)
    if labels.shape[0] != positions.shape[0]:
        raise RuntimeError("env label / halo count mismatch")
    # 个别 realization 有粒子贴盒边界（x<0.01 或 >999.99），TSC+interlacing
    # 位移会推出界。整体 +0.5 平移（周期盒内全局平移不影响任何谱），
    # 使所有粒子远离两侧边界。
    positions = np.mod(positions + np.float32(0.5), np.float32(boxsize))

    edges = tools.make_pk_edges(PK_KMIN, PK_KMAX, PK_KSTEP)
    if edges.size != 13:
        raise RuntimeError("expected 12 bins")

    output_dir = OUTPUT_ROOT / args.cosmology / f"real{real:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    # 一次性涂好所有涉及的类的场（同一次 paint，保证口径一致）。
    involved = sorted({c for pair in pairs for c in pair})
    fields = {}
    with tools.jaxpower_runtime_compat(mpicomm):
        import jaxpower as _jp
        from clustering_statistics.spectrum2_tools import prepare_jaxpower_particles
        with _jp.create_sharding_mesh(meshsize=PK_MESH):
            mattrs = {"boxsize": boxsize * np.ones(3, dtype="f4"),
                      "boxcenter": (boxsize / 2.0) * np.ones(3, dtype="f4"),
                      "meshsize": PK_MESH}
            for cls in involved:
                pos_c = np.ascontiguousarray(positions[labels == CLASS_IDS[cls]])
                cat_c = tools.make_halo_catalog(pos_c, los="z", mpicomm=mpicomm, redshift=0.5)
                parts = prepare_jaxpower_particles(lambda c=cat_c: {"data": c}, mattrs=mattrs)
                f = parts[0]["data"].paint(resampler="tsc", interlacing=3, compensate=True, out="real")
                fields[cls] = f - f.mean()
                records[f"n_{cls}"] = int(pos_c.shape[0])
            binner = _jp.BinMesh2SpectrumPoles(fields[involved[0]].attrs, edges=edges, ells=(0,))

            def spectrum_of(fa, fb):
                s = _jp.compute_mesh2_spectrum(fa, fb, bin=binner, los="z")
                pole = next(iter(s))
                value = np.asarray(pole.value())
                nmodes = np.asarray(pole.values("nmodes"))
                norm = np.asarray(pole.norm)
                return value * nmodes * boxsize**3 / norm, np.asarray(pole.coords("k"), dtype=float), nmodes

    for c, d in pairs:
        pk_cross, k_cross, nmodes = spectrum_of(fields[c], fields[d])
        print(
            f"[{datetime.now(timezone.utc).isoformat():.19}] {args.cosmology}/real{real:03d} "
            f"cross {c}-{d} P[:3]={np.round(pk_cross[:3], 1)}",
            flush=True,
        )
        output = output_dir / f"cross_{c}{d}_pk.npz"
        temporary = output.with_name(output.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            k=k_cross,
            pk=np.asarray(pk_cross, dtype="f8"),
            nmodes=nmodes,
            edges=np.asarray(edges, dtype="f8"),
            boxsize=np.array(boxsize, dtype="f8"),
            meshsize=np.array(PK_MESH, dtype="i8"),
        )
        os.replace(temporary, output)
        gate = {
            "k_close": bool(np.allclose(k_cross, np.asarray(SELF_GEOMETRY["k"]), rtol=2e-4, atol=2e-7)),
            "nmodes_exact": bool(np.array_equal(nmodes, np.asarray(SELF_GEOMETRY["nmodes"]))),
            "pk_all_finite": bool(np.all(np.isfinite(pk_cross))),
        }
        gate["pass"] = bool(all(gate.values()))
        if not gate["pass"]:
            raise RuntimeError(f"geometry gate failed for {c}{d}: {gate}")
        records[f"{c}{d}"] = {
            "n_c": records.get(f"n_{c}", int(np.sum(labels == CLASS_IDS[c]))),
            "n_d": records.get(f"n_{d}", int(np.sum(labels == CLASS_IDS[d]))),
            "geometry_gate": gate,
            "pk_sha256": sha256_file(output),
        }
        # 几何门：与自建锚点一致。
        with np.load(output, allow_pickle=False) as z:
            k = np.asarray(z["k"], dtype=float)
            nmodes = np.asarray(z["nmodes"])
            pk = np.asarray(z["pk"], dtype=float)
        k_expected = np.asarray(SELF_GEOMETRY["k"])
        n_expected = np.asarray(SELF_GEOMETRY["nmodes"])
        gate = {
            "k_close": bool(np.allclose(k, k_expected, rtol=2e-4, atol=2e-7)),
            "nmodes_exact": bool(np.array_equal(nmodes, n_expected)),
            "pk_all_finite": bool(np.all(np.isfinite(pk))),
        }
        gate["pass"] = bool(all(gate.values()))
        if not gate["pass"]:
            raise RuntimeError(f"geometry gate failed for {c}{d}: {gate}")
        records[f"{c}{d}"] = {
            "n_c": records.get(f"n_{c}"), "n_d": records.get(f"n_{d}"),
            "geometry_gate": gate,
            "pk_sha256": sha256_file(output),
        }

    metadata_path = output_dir / "cross_metadata.json"
    metadata = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cosmology": args.cosmology,
        "realization": real,
        "pairs": records,
        "geometry": {"meshsize": PK_MESH, "n_bins": 12, "kmax": PK_KMAX,
                     "anchor": SELF_GEOMETRY["note"],
                     "cross_shot_noise": "theory zero; no randoms used"},
        "inputs": {
            "positions": {"path": str(position_path), "sha256": sha256_file(position_path)},
            "env_labels": {"path": str(label_path), "sha256": sha256_file(label_path)},
        },
    }
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing["pairs"].update(records)
        existing["created_utc"] = metadata["created_utc"]
        metadata = existing
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output_dir),
                      "pairs": list(records)}))


if __name__ == "__main__":
    main()
