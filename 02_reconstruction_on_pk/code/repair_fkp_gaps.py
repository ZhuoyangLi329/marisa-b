#!/usr/bin/env python3
"""补产 FKP 自谱产品缺口（fid/LC_p 0-99 内 20 套组合，见 gap_audit.json）。

背景：这些套在原生产批次中因边界粒子出界 assert 崩溃遗留。本脚本复用
run_envclass_p0_only 的测量链（make_halo_catalog + 30 套 shifted randoms +
measure_pk + write_pk_npz，同 seed 公式），仅加 wrap v3 边界修复
（wrap 进 [1,999]，移动 <0.2% 盒长，谱影响可忽略）。产出与原产品同格式
（含 norm/num_shotnoise，供 fullspec v3 转换层使用）。
用法：python3 repair_fkp_gaps.py --shard i --nshard n（分片并行）。
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mpi4py import MPI

CODES = Path("/pscratch/sd/l/lzy/reconstruction-png/codes")
V2 = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
FULL = V2 / "envclass_p0only/full"
HALO = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions")
STO = Path("/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_source")
GAP_AUDIT = V2 / "envclass_p0only/fullspec/gap_audit.json"
CLASS_IDS = {"void": 0, "sheet": 1, "filament": 2, "knot": 3}
KMIN = 0.006283185307179587
NRAN = 30
SEED_BASE = 20260917
SELF_K = [0.0089, 0.016257, 0.022057, 0.027837, 0.034579, 0.040668,
          0.046767, 0.053235, 0.059556, 0.065895, 0.072131, 0.078453]
SELF_N = [26, 96, 128, 234, 440, 440, 738, 906, 1130, 1414, 1600, 2018]


def import_path(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    args = parser.parse_args()

    tools = import_path("rep_b000", CODES / "run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py")
    tools2 = import_path("rep_paperbin", CODES / "quijote_dm_2d_paperbin_runner_tools.py")
    from clustering_statistics.tools import Catalog

    mpicomm = MPI.COMM_SELF
    gaps = json.loads(GAP_AUDIT.read_text())["fkp"]
    tasks: dict[tuple[str, int], list[str]] = {}
    for item in gaps:
        cosmo, real_s, cls = item.split("/")
        tasks.setdefault((cosmo, int(real_s.replace("real", ""))), []).append(cls)

    keys = sorted(tasks)
    mine = keys[args.shard::args.nshard]
    print(f"shard {args.shard}/{args.nshard}: {len(mine)} reals: {mine}", flush=True)

    for cosmo, real in mine:
        output_dir = FULL / cosmo / f"real{real:03d}"
        todo = [c for c in sorted(tasks[(cosmo, real)])
                if not (output_dir / f"{c}_pk.npz").is_file()]
        if not todo:
            continue
        pos = tools.load_halo_positions(HALO / cosmo / f"real{real:03d}" / "positions_mpc_h.npy",
                                        dtype=np.float32, max_rows=None)
        labels = np.load(STO / cosmo / f"real{real:03d}" / "lam1_dm_nenv/env_labels.npy")
        # wrap v3：出界粒子移入 [1, 999] 安全区
        pos = np.mod(pos, np.float32(1000.0))
        pos[pos < np.float32(1.0)] += np.float32(998.0)
        pos[pos > np.float32(999.0)] -= np.float32(998.0)
        edges = tools.make_pk_edges(KMIN, 0.08, KMIN)
        for name in todo:
            cid = CLASS_IDS[name]
            pv = np.ascontiguousarray(pos[labels == cid])
            data = tools.make_halo_catalog(pv, los="z", mpicomm=mpicomm, redshift=0.5)
            seed = SEED_BASE + 1_000_000 * real + 1_000 * cid + (0 if cosmo == "fiducial" else 500_000)
            rpos, _meta = tools2.generate_rank_random_positions(
                local_data_count=int(pv.shape[0]), boxsize=1000.0, seed=seed,
                dtype=np.float32, nran=NRAN, mpicomm=mpicomm,
            )
            rpos = tools2.wrap_positions_to_periodic_box(
                rpos, boxsize=1000.0, dtype=np.float32, label=f"repair_{name}", mpicomm=mpicomm)
            shifted = Catalog({"POSITION": rpos,
                               "INDWEIGHT": np.ones(rpos.shape[0], dtype="f4")},
                              attrs=data.attrs, mpicomm=mpicomm)
            print(f"[{datetime.now(timezone.utc).isoformat():.19}] {cosmo}/real{real:03d} "
                  f"{name} n={pv.shape[0]} measuring", flush=True)
            with tools.jaxpower_runtime_compat(mpicomm):
                spectrum = tools.measure_pk(
                    lambda d=data, s=shifted: {"data": d, "shifted": s},
                    boxsize=1000.0, meshsize=256, edges=edges, los="z", cache={}, mpicomm=mpicomm)
            out = output_dir / f"{name}_pk.npz"
            tools.write_pk_npz(out, spectrum, edges=edges, boxsize=1000.0,
                               meshsize=256, mpicomm=mpicomm)
            with np.load(out) as z:
                k = np.asarray(z["k"], dtype=float)
                nm = np.asarray(z["nmodes"])
                pk = np.asarray(z["pk"], dtype=float)
            ok = (np.allclose(k, SELF_K, rtol=2e-4, atol=2e-7)
                  and np.array_equal(nm, np.asarray(SELF_N)) and np.all(np.isfinite(pk)))
            if not ok:
                raise RuntimeError(f"geometry gate failed: {cosmo}/real{real:03d}/{name}")
            print(f"   -> {name}_pk.npz pk[:3]={np.round(pk[:3], 1)}", flush=True)
    print("SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
