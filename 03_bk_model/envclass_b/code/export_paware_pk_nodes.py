#!/usr/bin/env python3
"""模块大纲：复用旧 jaxpower 分箱器，导出完整 Cartesian P 格点与原始分母。"""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
if hasattr(os,"sched_getaffinity"):
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
import hashlib
import json
from pathlib import Path
import numpy as np
from jaxpower.mesh import MeshAttrs
from jaxpower.mesh2 import BinMesh2SpectrumPoles


def main():
    root=Path("/pscratch/sd/l/lzy/reconstuction-png-v2")
    reference=root/"02_reconstruction_on_pk/inputs/p0_finite_local_png_lattice.npz"
    out=root/"03_bk_model/envclass_b/outputs/paware_pb_ir_brecfix_k012_k014_v2/templates/pk_nodes.npz"
    if out.exists():raise FileExistsError(out)
    with np.load(reference) as z:
        edges=z["pk_edges"];expected=z["pk_nmodes"]
        box=float(z["boxsize_mpc_h"]);size=int(z["meshsize"])
    mattrs=MeshAttrs(boxsize=box*np.ones(3,dtype="f4"),boxcenter=.5*box*np.ones(3,dtype="f4"),meshsize=size)
    binner=BinMesh2SpectrumPoles(mattrs,edges=edges,ells=(0,))
    coords=[np.asarray(x,dtype=np.float64) for x in mattrs.kcoords(kind="separation",sparse=True)]
    kmag=np.sqrt(sum(x*x for x in coords))
    ids=np.asarray(binner.ibin,dtype=np.int64).reshape(kmag.shape)
    good=(ids>=1)&(ids<=12)
    where=np.nonzero(good)
    vectors=np.column_stack([coords[a].ravel()[where[a]] for a in range(3)])
    ids=ids[good]-1
    counts=np.bincount(ids,minlength=12)
    assert np.array_equal(counts,expected),(counts,expected)
    assert np.allclose(np.linalg.norm(vectors,axis=1),kmag[good],rtol=1.e-15,atol=0.)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,vectors=vectors,bin_ids=ids,counts=counts,edges=edges)
    report={"status":"pass","mode_counts":counts.tolist(),"nodes":len(vectors),
        "geometry":"original jaxpower BinMesh2SpectrumPoles + MeshAttrs float32 kcoords, evaluated in float64",
        "source_sha256":hashlib.sha256(reference.read_bytes()).hexdigest(),
        "export_sha256":hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
