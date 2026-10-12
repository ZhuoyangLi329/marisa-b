#!/usr/bin/env python3
"""按 STO 环境类别测量 halo B000（tree-level 主线同款几何）。

代码大纲
========
1. 读取 halo 位置与环境标签，按类拆 catalog。
2. 复刻 baseline B000 生产口径（execution_log/quijote_halo_z0p5_r10_lam1_dm_nenv）：
   paperbin 15 bins 0.01-0.30（120 个 sugiyama 三角形）、mesh 256、nran=30
   shifted randoms、mask_edges_policy=empty、los=z。
3. 每类输出 {class}_b000.npz：B(120,)、k_tri(120,2)、nmodes、edges、n_halo；
   几何门与 baseline 矩阵的 b000_k 逐 bin 对齐（certified 参照）。
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

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from mpi4py import MPI

OLD_ROOT = Path("/pscratch/sd/l/lzy/reconstruction-png")
CODES = OLD_ROOT / "codes"
V2_ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
HALO_ROOT = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions")
STO_SOURCE = OLD_ROOT / "result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_source"
GEOM_MATRIX = (
    OLD_ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
    / "matrices/baseline_pre_standard_matrix.npz"
)
OUTPUT_ROOT = V2_ROOT / "03_bk_model/envclass_b/work"

CLASS_IDS = {"void": 0, "sheet": 1, "filament": 2, "knot": 3}
B_KMIN, B_KMAX = 0.01, 0.30
B_NBIN = 15
MESH = 256
NRAN = 30
SEED_BASE = 20260918


def import_path(name: str, path: Path):
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
    parser.add_argument("--classes", default="filament,sheet")
    args = parser.parse_args()

    tools = import_path("envb_tools", CODES / "run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py")
    tools2 = import_path("envb_paperbin", CODES / "quijote_dm_2d_paperbin_runner_tools.py")
    from clustering_statistics.tools import Catalog

    with np.load(GEOM_MATRIX, allow_pickle=False) as z:
        ref_k = np.asarray(z["b000_k"], dtype=np.float64)  # (120, 2)

    boxsize = float(tools.QUIJOTE_BOXSIZE_MPC_H)
    mpicomm = MPI.COMM_SELF
    real = int(args.realization)
    position_path = HALO_ROOT / args.cosmology / f"real{real:03d}" / "positions_mpc_h.npy"
    label_path = STO_SOURCE / args.cosmology / f"real{real:03d}" / "lam1_dm_nenv/env_labels.npy"
    positions = tools.load_halo_positions(position_path, dtype=np.float32, max_rows=None)
    labels = np.load(label_path)
    if labels.shape[0] != positions.shape[0]:
        raise RuntimeError("env label / halo count mismatch")

    # paperbin B edges：centers 0.01..0.30 共 15 bins（certified 口径）
    kstep = (B_KMAX - B_KMIN) / (B_NBIN - 1)
    edges = tools2.make_centered_edges(B_KMIN, B_KMAX, kstep)

    output_dir = OUTPUT_ROOT / args.cosmology / f"real{real:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in [c.strip() for c in args.classes.split(",") if c.strip()]:
        cid = CLASS_IDS[name]
        pv = np.ascontiguousarray(positions[labels == cid])
        n = int(pv.shape[0])
        data = tools.make_halo_catalog(pv, los="z", mpicomm=mpicomm, redshift=0.5)
        seed = SEED_BASE + 1_000_000 * real + 1_000 * cid + (0 if args.cosmology == "fiducial" else 500_000)
        rpos, _meta = tools2.generate_rank_random_positions(
            local_data_count=n, boxsize=boxsize, seed=seed, dtype=np.float32,
            nran=NRAN, mpicomm=mpicomm,
        )
        rpos = tools2.wrap_positions_to_periodic_box(
            rpos, boxsize=boxsize, dtype=np.float32, label=f"envclassB_{name}", mpicomm=mpicomm)
        shifted = Catalog({"POSITION": rpos,
                           "INDWEIGHT": np.ones(rpos.shape[0], dtype="f4")},
                          attrs=data.attrs, mpicomm=mpicomm)
        print(f"[{datetime.now(timezone.utc).isoformat():.19}] {args.cosmology}/real{real:03d} "
              f"class={name} n={n} measuring B000", flush=True)
        with tools.jaxpower_runtime_compat(mpicomm):
            spectrum = tools.measure_b000(
                lambda d=data, s=shifted: {"data": d, "shifted": s},
                boxsize=boxsize, meshsize=MESH, edges=edges, los="z",
                basis="sugiyama", mask_edges_policy="default", cache={}, mpicomm=mpicomm,
            )
        pole = next(iter(spectrum))
        B = np.asarray(pole.value(), dtype=np.float64)
        k_tri = np.asarray(pole.coords("k"), dtype=np.float64)
        nmodes = np.asarray(pole.values("nmodes"))
        if B.shape[0] != 120:
            raise RuntimeError(f"expected 120 triangles, got {B.shape[0]}")
        # 几何门：k_tri 与 baseline 参照逐 bin 对齐（坐标系次序可能为 (k1,k2)）
        if k_tri.ndim == 1:
            k_tri = k_tri.reshape(-1, 1)
        if k_tri.shape[1] == 2 and not np.allclose(k_tri, ref_k, rtol=1e-3, atol=1e-7):
            # 尝试镜像次序 (k2,k1)
            if np.allclose(k_tri[:, ::-1], ref_k, rtol=1e-3, atol=1e-7):
                k_tri = k_tri[:, ::-1]
            else:
                raise RuntimeError(f"b000 geometry gate failed for {name}")
        out = output_dir / f"{name}_b000.npz"
        tmp = out.with_name(out.name + ".tmp.npz")
        np.savez_compressed(
            tmp, B=B, k_tri=k_tri, nmodes=nmodes, edges=np.asarray(edges, dtype=np.float64),
            n_halo=np.array(n, dtype=np.int64), meshsize=np.array(MESH), nran=np.array(NRAN),
        )
        os.replace(tmp, out)
        records[name] = {"n_halo": n, "sha256": sha256_file(out),
                         "B_min": float(np.nanmin(B)), "B_max": float(np.nanmax(B)),
                         "all_finite": bool(np.all(np.isfinite(B)))}
        del spectrum, pole

    (output_dir / "b000_metadata.json").write_text(json.dumps({
        "schema": "envclass-b000-product-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cosmology": args.cosmology, "realization": real,
        "geometry": {"bins": B_NBIN, "kmin": B_KMIN, "kmax": B_KMAX,
                     "basis": "sugiyama", "mask_edges_policy": "empty",
                     "triangles": 120, "meshsize": MESH, "nran": NRAN, "los": "z"},
        "classes": records,
        "inputs": {"positions": {"sha256": sha256_file(position_path)},
                   "env_labels": {"sha256": sha256_file(label_path)}},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output_dir),
                      "classes": {k: v["n_halo"] for k, v in records.items()}}))


if __name__ == "__main__":
    main()
