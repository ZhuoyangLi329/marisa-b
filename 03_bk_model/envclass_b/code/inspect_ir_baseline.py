#!/usr/bin/env python3
"""模块大纲：只读核验 p-aware MCMC 参数与实际模板；不拟合、不修改旧模型或结果。"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b")
sys.path.insert(0,str(ROOT/"code"))
import run_joint_pb_matched250 as run


def record(path):
    path=Path(path)
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    low,scan,runtime=run.configure_models()
    cfg=run.CASES["sto_marisa"]
    low.bphi_from_b1=lambda b:2*low.DELTA_C*(float(b)-cfg["p"])
    model=scan.DirectFiniteScanB000Model(runtime,"marisa")
    result=json.loads((ROOT/"outputs/paware_pb_mcmc_k012_k014_v1/sto_marisa_k0p12/RESULTS.json").read_text())
    mapping={k:v["median"] for k,v in result["posterior"].items() if k in model.names}
    def parts(brec):
        low.B_REC=brec;low.P_REC=cfg["p_rec"]
        low.BPHI_REC=2*low.DELTA_C*(brec-low.P_REC)
        return model.parts(mapping)
    old_brec=float(runtime.marisa_finite["b_rec"])
    old=parts(old_brec);configured=parts(cfg["b_rec"])
    idx=scan.selected_indices(runtime.geometry,.12)
    differences={name:{"max_abs":float(np.max(np.abs(configured[name][idx]-old[name][idx]))),
                       "relative_l2":float(np.linalg.norm(configured[name][idx]-old[name][idx])/
                                           max(1.,np.linalg.norm(old[name][idx])))} for name in configured}
    fixed_sector={}
    for name,adaptive_names in (("gaussian",()),("linear_coefficient",("adaptive_linear",)),
                               ("quadratic_coefficient",("adaptive_quadratic","adaptive_cross"))):
        first=old[name].copy();second=configured[name].copy()
        for key in adaptive_names:
            first-=old[key];second-=configured[key]
        fixed_sector[name]=float(np.max(np.abs(first[idx]-second[idx])))
    backend=runtime.high_runtime.modules["baseline"]
    assets=runtime.high_runtime.modules["assets"]
    adaptive_path=run.VENDOR/"data/marisa_finite_kmax0p20.npz"
    report={"status":"baseline_reconstruction_parameter_mismatch_confirmed",
            "configured_sto_marisa":cfg,
            "actual_standard_loader_b_rec":float(backend.B_REC),
            "actual_png_cache_b_rec":float(assets.png.b_rec_h),
            "actual_adaptive_cache":{name:float(runtime.marisa_finite[name]) for name in ("b_rec","p_rec","bphi_rec")},
            "runtime_change_low_B_REC":differences,
            "nonadaptive_sector_max_abs_change":fixed_sector,
            "interpretation":"The runner changes adaptive prefactors, but not the fixed-brec source templates or the 1/b_rec already inside adaptive raw bases.",
            "no_new_mcmc_submitted":True,
            "files":{"runner":record(run.__file__),"scan":record(scan.__file__),"backend":record(backend.__file__),
                     "adaptive_cache":record(adaptive_path),"gaussian_cache":record(backend.TREE_CACHE),
                     "png_cache":record(backend.PNG_CACHE),"old_mcmc_summary":record(ROOT/"outputs/paware_pb_mcmc_k012_k014_v1/SUMMARY.json")},
            "source_closure_names":sorted(inspect.getclosurevars(model.base_context.evaluate_model).nonlocals)}
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
