"""模块大纲：生产 FFT 投影的 f→0 Gaussian 极限对照实空间缓存；不改拟合。

raw native 随机方向测试之外，再核对真正两腿 window 与 bias 基底映射。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from rsd_b000_fft_projection import FFTProjection,source_terms
from rsd_b000_contract import PART,reference,digest,save_json

def main():
    ref=reference();indices=[int(i) for i in ref['certified_indices'] if np.max(ref['B000_k'][i])<=.14]
    # 覆盖两个低壳 mixed、两个 diagonal 和高壳 mixed；不是用恢复真值挑输入。
    indices=[indices[i] for i in (0,5,8,13,19,26)]
    path=PART.parent/'03_bk_model/halo/outputs/local_png_tree_ir_20260930/ir_model_designs_convergence_v1.npz'
    with np.load(path) as z:real=np.array(z['current__fixed_order'])[0,indices,:3]
    rows=[];native=[];b1=1.93
    for index in indices:
        projection=FFTProjection(96,ref['edges'],ref['shell_pairs'][index],b1,f_growth=0.)
        g=np.zeros(3)
        for weights,shift,(x,y,z) in source_terms(b1,0.):
            if shift or not np.any(weights):continue
            g+=weights*projection.integral(x,0,y,0,z,0)
        native.append(g);rows.append({'index':index,'pair':ref['shell_pairs'][index].tolist(),
                                      'k1_k2_mean':ref['B000_k'][index].tolist(),'native_gaussian_base_b2_K':g.tolist()})
        print('ZERO_GROWTH_PROBE',index,g.tolist(),flush=True)
        del projection
    native=np.array(native);base_delta=(native[:,0]-real[:,0])/np.maximum(np.abs(real[:,0]),1.)
    # 不把 gamma2 与 K2 的记号差异误判为物理错误：独立求两列之间的基底映射。
    mapping=np.linalg.lstsq(native[:,1:],real[:,1:],rcond=None)[0]
    residual=native[:,1:]@mapping-real[:,1:]
    denom=np.maximum(np.abs(real[:,1:]),1.)
    result={'status':'PASS' if np.max(np.abs(base_delta))<2e-10 and np.max(np.abs(residual/denom))<2e-10 else 'REQUIRES_INVESTIGATION',
            'scope':'Gaussian f_growth=0 end-to-end two-leg window, deterministic columns only; no data fitting or model changes',
            'fixed_b1':b1,'mesh':96,'indices':indices,'base_max_relative_error':float(np.max(np.abs(base_delta))),
            'realspace_columns':['b2_EFT','gamma2'],'native_columns':['b2_native','bK2'],
            'realspace_columns_from_native_matrix':mapping.tolist(),'basis_map_max_relative_error':float(np.max(np.abs(residual/denom))),
            'unmapped_claim_is_not_made':True,'realspace_source_sha256':digest(path),'entry_sha256':digest(Path(__file__)),
            'native_projection_sha256':digest(PART/'code/rsd_b000_fft_projection.py'),
            'allowed_CPUs':sorted(os.sched_getaffinity(0)),'rows':rows}
    save_json(PART/'outputs/realspace_vs_rsd_diagnosis_20261002/ZERO_GROWTH_GAUSSIAN.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
