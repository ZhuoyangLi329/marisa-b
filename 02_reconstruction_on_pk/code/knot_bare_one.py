"""knot 裸自谱缓存 worker（自包含版）。

不 import run_envclass_fullspec_mcmc（v8 的模块级 h5py/emcee 链与 mpi4py
冲突：h5py 并行构建初始化 MPICH 后 mpi4py 报 before/after finalizing）。
涂场逻辑与 build_cache / crosspk 生产完全同款：wrap v3 + tsc + interlacing=3
+ compensate=True，P = value*nmodes*V/norm。
"""
import os
import sys
from pathlib import Path

for _n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"

import numpy as np
from mpi4py import MPI

sys.path.insert(0, "/pscratch/sd/l/lzy/reconstruction-png/codes")
from run_quijote_halo_z0p5_jaxrecon_jaxpower_b000 import (
    make_halo_catalog, jaxpower_runtime_compat, load_halo_positions,
)

CACHE = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/envclass_p0only/fullspec/bare_auto_cache")
HALO = Path("/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions")
STO = Path("/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100_source")
EDGES = np.arange(0.006283185307179587, 0.0817, 0.006283185307179587)[:13]

real, cosmo = int(sys.argv[1]), sys.argv[2]
out = CACHE / cosmo / f"real{real:03d}" / "k.npy"
if out.is_file():
    print(f"exists: {out}", flush=True)
    sys.exit(0)

pos = load_halo_positions(HALO / cosmo / f"real{real:03d}" / "positions_mpc_h.npy",
                          dtype=np.float32, max_rows=None)
labels = np.load(STO / cosmo / f"real{real:03d}" / "lam1_dm_nenv/env_labels.npy")
pos = np.mod(pos, np.float32(1000.0))
pos[pos < np.float32(1.0)] += np.float32(998.0)
pos[pos > np.float32(999.0)] -= np.float32(998.0)
pk_pos = np.ascontiguousarray(pos[labels == 3])
out.parent.mkdir(parents=True, exist_ok=True)
with jaxpower_runtime_compat(MPI.COMM_SELF):
    import jaxpower
    from clustering_statistics.spectrum2_tools import prepare_jaxpower_particles
    with jaxpower.create_sharding_mesh(meshsize=256):
        mattrs = {"boxsize": np.full(3, 1000.0, "f4"),
                  "boxcenter": np.full(3, 500.0, "f4"), "meshsize": 256}
        cat = make_halo_catalog(pk_pos, los="z", mpicomm=MPI.COMM_SELF, redshift=0.5)
        parts = prepare_jaxpower_particles(lambda c=cat: {"data": c}, mattrs=mattrs)
        f = parts[0]["data"].paint(resampler="tsc", interlacing=3, compensate=True, out="real")
        f = f - f.mean()
        b = jaxpower.BinMesh2SpectrumPoles(f.attrs, edges=EDGES, ells=(0,))
        s = jaxpower.compute_mesh2_spectrum(f, f, bin=b, los="z")
        pole = next(iter(s))
        P = (np.asarray(pole.value()) * np.asarray(pole.values("nmodes"))
             * 1e9 / np.asarray(pole.norm))
tmp = out.with_name(out.name + ".tmp.npy")
np.save(tmp, np.asarray(P, dtype=np.float64))
os.replace(tmp, out)
print(f"done: {out} n={pk_pos.shape[0]} P[:3]={np.round(P[:3], 1)}", flush=True)
