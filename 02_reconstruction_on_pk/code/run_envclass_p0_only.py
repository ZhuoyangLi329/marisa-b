#!/usr/bin/env python3
"""按 STO 环境类别（void/sheet/filament，丢 knot）逐类测量 P0 的生产任务。

代码大纲
========
1. 读取 halo 位置与 STO 生产的环境标签（DM 潮汐分类 lam1_dm_nenv），断言
   逐 halo 对齐；按类别拆成独立 catalog。
2. 复用旧仓库认证过的测量工具（jaxrecon_jaxpower_b000 的
   make_pk_edges/measure_pk/write_pk_npz 与 pre_p0_only 的几何参照门），
   与五方法 P0 完全同一几何：mesh 256、12 bins、kmax=0.08、los=z、
   shifted uniform randoms（nran=30 每类独立确定性种子）。
3. 每类输出 {class}_pk.npz + metadata.json（类别计数、输入 sha、几何门）
   + DONE；数据向量阶段只用自谱，互协方差由 realization 堆叠获得。
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
GEOMETRY_REFERENCE = (
    OLD_ROOT / "analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100"
    / "matrices/baseline_pre_standard_matrix.npz"
)
OUTPUT_ROOT = V2_ROOT / "02_reconstruction_on_pk/envclass_p0only/full"

SCHEMA = "quijote-halo-z0p5-envclass-p0only-product-v1"
CLASS_IDS = {"void": 0, "sheet": 1, "filament": 2, "knot": 3}
PK_KMIN = 0.006283185307179587
PK_KMAX = 0.08
PK_KSTEP = 0.006283185307179587
PK_MESH = 256
NRAN = 30
SEED_BASE = 20260917
# 路线 A：自建几何锚点（2026-09-17 login 节点 20260321 环境实测）。
# 现装 jaxpower 的 binning 语义（hermitian/interlacing 计数）与旧 GPU 生产
# 不同，历史 certified 参照 (k1=0.015079, nmodes[1]=66) 无法逐 bin 复现；
# 本分析自成一体：target/covariance/理论模板全部按本几何生成，
# 与旧几何的差异在 audit 中如实记录，不冒充同几何可比。
SELF_GEOMETRY = {
    "k": [0.0089, 0.016257, 0.022057, 0.027837, 0.034579, 0.040668,
          0.046767, 0.053235, 0.059556, 0.065895, 0.072131, 0.078453],
    "nmodes": [26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018],
    "note": "self-consistent geometry anchor measured on login26/20260321 env",
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
    parser.add_argument("--classes", default="void,sheet,filament")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    tools = import_path("envclass_tools_b000", CODES / "run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py")
    pre_mod = import_path("envclass_pre_p0", CODES / "run_quijote_halo_z0p5_pre_p0_only.py")
    tools2 = import_path(
        "envclass_tools_paperbin", CODES / "quijote_dm_2d_paperbin_runner_tools.py"
    )
    from clustering_statistics.tools import Catalog

    boxsize = float(tools.QUIJOTE_BOXSIZE_MPC_H)
    mpicomm = MPI.COMM_SELF

    real = int(args.realization)
    position_path = HALO_ROOT / args.cosmology / f"real{real:03d}" / "positions_mpc_h.npy"
    label_path = STO_SOURCE / args.cosmology / f"real{real:03d}" / "lam1_dm_nenv" / "env_labels.npy"
    for path in (position_path, label_path, GEOMETRY_REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(path)

    positions = tools.load_halo_positions(position_path, dtype=np.float32, max_rows=None)
    labels = np.load(label_path)
    if labels.shape[0] != positions.shape[0]:
        raise RuntimeError(
            f"env label count {labels.shape[0]} != halo count {positions.shape[0]}"
        )

    edges = tools.make_pk_edges(PK_KMIN, PK_KMAX, PK_KSTEP)
    if edges.size != 13:
        raise RuntimeError(f"expected 12 bins, got {edges.size - 1}")
    # 自建几何校验：pilot 实测的 k/nmodes 锚点（路线 A，见文件头注释）。
    def self_geometry_gate(npz_path: Path) -> dict:
        with np.load(npz_path, allow_pickle=False) as z:
            k = np.asarray(z["k"], dtype=float)
            nmodes = np.asarray(z["nmodes"])
            pk = np.asarray(z["pk"], dtype=float)
        k_expected = np.asarray(SELF_GEOMETRY["k"])
        n_expected = np.asarray(SELF_GEOMETRY["nmodes"])
        result = {
            "edges_exact": bool(np.array_equal(np.asarray(z["edges"]) if False else edges, edges)),
            "k_close": bool(np.allclose(k, k_expected, rtol=2e-4, atol=2e-7)),
            "nmodes_exact": bool(np.array_equal(nmodes, n_expected)),
            "k_max_abs_difference": float(np.max(np.abs(k - k_expected))),
            "pk_all_finite": bool(np.all(np.isfinite(pk))),
        }
        result["pass"] = bool(
            result["k_close"] and result["nmodes_exact"] and result["pk_all_finite"]
        )
        return result

    output_dir = OUTPUT_ROOT / args.cosmology / f"real{real:03d}"
    if args.check_only:
        counts = {name: int(np.sum(labels == cid)) for name, cid in CLASS_IDS.items()}
        print(json.dumps({
            "status": "check_only", "cosmology": args.cosmology, "realization": real,
            "n_halo": int(positions.shape[0]), "class_counts": counts,
            "edges_match": True,
        }))
        return

    class_list = [c.strip() for c in args.classes.split(",") if c.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for name in class_list:
        cid = CLASS_IDS[name]
        mask = labels == cid
        class_positions = np.ascontiguousarray(positions[mask])
        if class_positions.shape[0] < 2:
            raise RuntimeError(f"class {name} has too few halos: {class_positions.shape[0]}")
        data = tools.make_halo_catalog(class_positions, los="z", mpicomm=mpicomm, redshift=0.5)
        seed = SEED_BASE + 1_000_000 * real + 1_000 * cid + (0 if args.cosmology == "fiducial" else 500_000)
        random_positions, random_metadata = tools2.generate_rank_random_positions(
            local_data_count=int(class_positions.shape[0]),
            boxsize=boxsize,
            seed=seed,
            dtype=np.float32,
            nran=NRAN,
            mpicomm=mpicomm,
        )
        random_positions = tools2.wrap_positions_to_periodic_box(
            random_positions, boxsize=boxsize, dtype=np.float32,
            label=f"envclass_{name}_{args.cosmology}_real{real:03d}", mpicomm=mpicomm,
        )
        shifted = Catalog(
            {"POSITION": random_positions, "INDWEIGHT": np.ones(random_positions.shape[0], dtype="f4")},
            attrs=data.attrs, mpicomm=mpicomm,
        )
        print(
            f"[{datetime.now(timezone.utc).isoformat():.19}] {args.cosmology}/real{real:03d} "
            f"class={name} n={class_positions.shape[0]} measuring P0",
            flush=True,
        )
        with tools.jaxpower_runtime_compat(mpicomm):
            spectrum = tools.measure_pk(
                lambda: {"data": data, "shifted": shifted},
                boxsize=boxsize,
                meshsize=PK_MESH,
                edges=edges,
                los="z",
                cache={},
                mpicomm=mpicomm,
            )
        class_output = output_dir / f"{name}_pk.npz"
        summary = tools.write_pk_npz(
            class_output, spectrum, edges=edges, boxsize=boxsize, meshsize=PK_MESH, mpicomm=mpicomm,
        )
        del spectrum
        gate = self_geometry_gate(class_output)
        if not gate["pass"]:
            raise RuntimeError(f"geometry gate failed for {name}: {gate}")
        records[name] = {
            "n_halo": int(class_positions.shape[0]),
            "seed": seed,
            "randoms": random_metadata,
            "summary": summary,
            "geometry_gate": gate,
            "pk_sha256": sha256_file(class_output),
        }

    metadata = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cosmology": args.cosmology,
        "realization": real,
        "input_fNL": 0.0 if args.cosmology == "fiducial" else 100.0,
        "env_classification": "STO lam1_dm_nenv（DM 潮汐，λ_th=1，gsr=1，mesh128）",
        "classes": records,
        "geometry": {
            "meshsize": PK_MESH, "n_bins": 12,
            "kmin": PK_KMIN, "kmax": PK_KMAX, "kstep": PK_KSTEP, "los": "z",
            "nran_per_data": NRAN,
            "reference": str(GEOMETRY_REFERENCE),
        },
        "inputs": {
            "positions": {"path": str(position_path), "sha256": sha256_file(position_path)},
            "env_labels": {"path": str(label_path), "sha256": sha256_file(label_path)},
            "geometry_reference": {
                "mode": "self_consistent_v1",
                "anchor": SELF_GEOMETRY,
            },
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "DONE").write_text(
        json.dumps({"schema": SCHEMA, "cosmology": args.cosmology, "realization": real,
                    "created_utc": metadata["created_utc"]}) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output": str(output_dir),
                      "classes": {k: v["n_halo"] for k, v in records.items()}}))


if __name__ == "__main__":
    main()
