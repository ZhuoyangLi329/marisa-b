#!/usr/bin/env python3
"""功率收支分解（v2，三类）：裸涂场谱 vs FKP 差场谱的转换系数 C_c 测定。

v1（void 单类）发现：P_FKP = C × P_bare/nmodes − sn 近似闭合，C ≈ 2.467 ≈ π²/4。
100 套平均反推提示 C_c 逐类不同、疑似 ∝ 1/n_c²。本版本对 void/sheet/filament
逐类完整分解（每类：数据场裸自谱 + FKP 差场复现 + 随机场谱），输出
C_c(k) = (P_FKP + sn) / (P_bare/nmodes) 的 12 bin 表，判定：
  (a) C_c 逐 bin 是否恒定（归一化性质）
  (b) C_c 与 n_c 的幂律关系（n² 假说）
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

CODES = Path("/pscratch/sd/l/lzy/reconstruction-png/codes")
V2 = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk")
CACHE = V2 / "envclass_p0only/fullspec/bare_auto_cache"
FKP_ROOT = V2 / "envclass_p0only/full"
POS = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions/fiducial/real000/positions_mpc_h.npy")
CLASS_IDS = {"void": 0, "sheet": 1, "filament": 2, "knot": 3}
KMIN = 0.006283185307179587
NRAN = 30
SEED_BASE = 20260917


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
    parser.add_argument("--classes", default="void,sheet,filament")
    args = parser.parse_args()

    tools = import_path("t_b000", CODES / "run_quijote_halo_z0p5_jaxrecon_jaxpower_b000.py")
    tools2 = import_path("t_paperbin", CODES / "quijote_dm_2d_paperbin_runner_tools.py")
    from clustering_statistics.tools import Catalog

    pos_path = POS
    lab_path = Path("/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_source/fiducial/real000/lam1_dm_nenv/env_labels.npy")
    mpicomm = MPI.COMM_SELF
    pos = tools.load_halo_positions(pos_path, dtype=np.float32, max_rows=None)
    labels = np.load(lab_path)
    edges = tools.make_pk_edges(KMIN, 0.08, KMIN)
    assert edges.size == 13

    payload = {}
    for cls in [c.strip() for c in args.classes.split(",")]:
        cid = CLASS_IDS[cls]
        short = cls[0]
        pv = np.ascontiguousarray(pos[labels == cid])
        n = int(pv.shape[0])
        vn = 1.0e9 / n
        posw = np.mod(pv, np.float32(1000.0))
        posw[posw < np.float32(1.0)] += np.float32(998.0)
        posw[posw > np.float32(999.0)] -= np.float32(998.0)

        data = tools.make_halo_catalog(pv, los="z", mpicomm=mpicomm, redshift=0.5)
        seed = SEED_BASE + 1000 * cid
        rpos, _meta = tools2.generate_rank_random_positions(
            local_data_count=n, boxsize=1000.0, seed=seed, dtype=np.float32,
            nran=NRAN, mpicomm=mpicomm,
        )
        rpos = tools2.wrap_positions_to_periodic_box(
            rpos, boxsize=1000.0, dtype=np.float32, label=f"probe_{cls}", mpicomm=mpicomm,
        )
        shifted = Catalog(
            {"POSITION": rpos, "INDWEIGHT": np.ones(rpos.shape[0], dtype="f4")},
            attrs=data.attrs, mpicomm=mpicomm,
        )
        with tools.jaxpower_runtime_compat(mpicomm):
            spec = tools.measure_pk(
                lambda d=data, s=shifted: {"data": d, "shifted": s},
                boxsize=1000.0, meshsize=256, edges=edges, los="z", cache={}, mpicomm=mpicomm,
            )
            pole = next(iter(spec))
            pk_fkp = np.asarray(pole.value(), dtype=np.float64)
            norm_fkp = np.asarray(pole.values("norm"), dtype=np.float64)
            ns_fkp = np.asarray(pole.values("num_shotnoise"), dtype=np.float64)
            nmodes = np.asarray(pole.values("nmodes"))
            del spec, pole

            import jaxpower
            from clustering_statistics.spectrum2_tools import prepare_jaxpower_particles

            with jaxpower.create_sharding_mesh(meshsize=256):
                mattrs = {
                    "boxsize": np.full(3, 1000.0, "f4"),
                    "boxcenter": np.full(3, 500.0, "f4"),
                    "meshsize": 256,
                }

                def paint(positions):
                    cat = tools.make_halo_catalog(np.ascontiguousarray(positions), los="z",
                                                  mpicomm=mpicomm, redshift=0.5)
                    parts = prepare_jaxpower_particles(lambda c=cat: {"data": c}, mattrs=mattrs)
                    f = parts[0]["data"].paint(resampler="tsc", interlacing=3,
                                               compensate=True, out="real")
                    return f - f.mean()

                fd = paint(posw)
                binner = jaxpower.BinMesh2SpectrumPoles(fd.attrs, edges=edges, ells=(0,))
                s = jaxpower.compute_mesh2_spectrum(fd, fd, bin=binner, los="z")
                p = next(iter(s))
                vdd = np.asarray(p.value(), dtype=np.float64)
                nd = np.asarray(p.norm, dtype=np.float64)
                nmd = np.asarray(p.values("nmodes"))
                del fd, s, p

        pdd_bare = vdd * nmd * 1.0e9 / nd          # 缓存口径（应复现 v.npy）
        cached = np.load(CACHE / f"fiducial/real000/{short}.npy")
        pdd_bare_over_nmodes = pdd_bare / nmd
        sn = ns_fkp / norm_fkp
        C_k = (pk_fkp + sn) / pdd_bare_over_nmodes  # 待判定：逐 bin 恒定？
        with np.load(FKP_ROOT / "fiducial/real000" / f"{cls}_pk.npz") as z:
            pk_product = np.asarray(z["pk"], dtype=np.float64)

        print(f"\n===== {cls}  n={n}  V/n={vn:.1f}  nd[0]={nd[0]:.1f} (nd/n={nd[0]/n:.4f}) =====")
        print(f"cache==bare : {np.allclose(cached, pdd_bare, rtol=1e-6)}")
        print(f"fkp==product: {np.allclose(pk_fkp, pk_product, rtol=1e-6)}")
        print(f"C(k)=(P_FKP+sn)/(P_bare/nmodes): {np.array2string(C_k, precision=4)}")
        print(f"  C mean(bins2-12)={np.mean(C_k[1:]):.4f}  std/mean={np.std(C_k[1:])/np.mean(C_k[1:]):.4f}")
        print(f"  C*n^2/1e10 = {np.mean(C_k[1:]) * n * n / 1e10:.4f}")
        print(f"  nmodes: {nmodes}")
        payload[cls] = {
            "n": n, "V_over_n": vn, "nd_over_n": float(nd[0] / n),
            "C_k": C_k.tolist(), "nmodes": nmodes.tolist(),
            "pdd_bare": pdd_bare.tolist(), "pk_fkp": pk_fkp.tolist(),
            "sn": sn.tolist(), "norm_fkp": norm_fkp.tolist(), "nd": nd.tolist(),
        }

    out = V2 / "envclass_p0only/fullspec/exp_decompose_v2_result.json"
    existing = json.loads(out.read_text()) if out.is_file() else {}
    existing.update(payload)
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
