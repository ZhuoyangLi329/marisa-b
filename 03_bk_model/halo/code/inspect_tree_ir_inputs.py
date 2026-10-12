#!/usr/bin/env python3
"""只读检查现有 B-only 基准并冻结对照输入。

模块大纲：加载旧模型/数据；记录先验和投影；提取仿射 nuisance 模板；
全部新输出写入当前任务 outputs，历史仓库仅作为冻结输入读取。
"""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import sys
from pathlib import Path
import json
import inspect
import numpy as np

ROOT=Path('/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model')
OLD=Path('/pscratch/sd/l/lzy/reconstruction-png')
OUT=ROOT/'halo/outputs/local_png_tree_ir_20260930'
sys.path.insert(0,str(OLD/'codes'))
import quijote_halo_z0p5_direct_finite_fnl_kmax_scan_models as scan

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runtime=scan.load_runtime()
    model=scan.DirectFiniteScanB000Model(runtime,'pre')
    context=model.base_context
    arrays={"geometry":runtime.geometry,"certified_indices":scan.CERTIFIED_INDICES,
            "prior_mean":np.asarray(context.prior.mean),
            "prior_whitener":np.asarray(context.prior.whitener)}
    matrix=runtime.low_runtime.b_matrices['standard']
    for truth,key in scan.low.TRUTH_KEYS.items():
        arrays[f"data_{truth:+.0f}"]=matrix[f'{key}_pre_B000'][:500 if truth==0 else 250]
    zero={"b1":1.93,"b2":0.,"gamma2":0.,"Ashot_residual":0.,"Bshot_residual":0.}
    parts0=model.parts(zero)
    for name,value in parts0.items(): arrays[f'legacy_zero_{name}']=value
    for nuisance in context.names[1:]:
        mapped=dict(zero);mapped[nuisance]=1.
        parts=model.parts(mapped)
        for name,value in parts.items():
            arrays[f'legacy_{nuisance}_{name}']=value-parts0[name]
    closure=runtime.low_runtime.pre_closure
    free=inspect.getclosurevars(context.evaluate_model).nonlocals
    info={"names":list(context.names),"nbar":float(closure['nbar']),
          "cuts":list(scan.CUTS),"p_legacy":scan.low.P_BPHI,
          "p_current":1.1788659191136683,
          "context_input_hashes":context.input_hashes,
          "closure_keys":list(free),"low_closure_keys":list(closure),
          "pre_tree_fields":list(runtime.pre_tree)}
    for name,obj in free.items():
        if hasattr(obj,'__dict__'):
            info[f'object_{name}']={k:str(v)[:500] for k,v in vars(obj).items()
                                  if k in ('path','cache_path','metadata','manifest','paths','config')}
    np.savez_compressed(OUT/'frozen_inputs.npz',**arrays)
    (OUT/'frozen_inputs.json').write_text(json.dumps(info,indent=2,default=str)+'\n')
    print(json.dumps(info,indent=2,default=str))

if __name__=='__main__':main()
