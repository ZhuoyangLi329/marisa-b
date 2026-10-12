"""模块大纲：实际最低/最高B-bin双网格，独立整数pair-count对vector算子。

短时单核理论探针，不是44-bin模板完成、physical-p校准或STO模型验收。
只读已冻结的窗口/谱/alpha/源码；对原source/fNL²/零腿/full denominator逐项核验。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time
import numpy as np
from rsd_b000_contract import PART,reference,digest
from post_rsd_ir_inputs import ROOT,frozen_reconstruction
from post_rsd_poisson_projection import PoissonShiftFFTProjection
from post_rsd_noise_transport import integrate_noise_transport
from audit_post_rsd_noise_transport_coefficients import scalar_oracle,integer_count_oracle


def main():
    if len(os.sched_getaffinity(0))!=1:raise ValueError('single-core short scalar probe required')
    out=ROOT/'NOISE_SCALAR_TWO_BIN_PROBE_AUDIT.json'
    if out.exists():raise RuntimeError('preserve previous scalar probe')
    cp=ROOT/'noise_transport_templates_v1/CONTRACT.json';contract=json.loads(cp.read_text())
    for path,want in contract['immutable_sources'].items():
        if digest(path)!=want:raise ValueError('frozen operator source/input drift: '+path)
    ref=reference();rec=frozen_reconstruction(256)
    with np.load(ROOT/'estimator_B_window/B000_WINDOW.npz',allow_pickle=False) as z:counts=np.array(z['measured_pair_counts'])
    # 未校准的物理p仅作diagnostic probe，不使用p_rec或将此值固定到MCMC。
    b,ph=2.11,1.417;powers=np.array([1.,500.,250000.])[None,:];rows=[];start=time.time()
    for index in (contract['indices'][0],contract['indices'][-1]):
        for mesh in contract['meshes'][str(index)]:
            field=PoissonShiftFFTProjection(mesh,ref['edges'],ref['shell_pairs'][index],b,contract['f_growth'],
                p_halo=ph,reconstruction=rec)
            if field.denominator!=counts[index]:raise ValueError('actual estimator window count mismatch')
            vector=integrate_noise_transport(field)[1:];scalar=scalar_oracle(field)
            independent,count_meta=integer_count_oracle(field)
            scale=np.maximum(1.,np.max(np.abs(independent)*powers,axis=1,keepdims=True))
            vector_error=float(np.max(np.abs(vector-independent)*powers/scale))
            scalar_error=float(np.max(np.abs(scalar-independent)*powers/scale))
            row=dict(bin_index=index,mesh=mesh,vector_design=vector.tolist(),scalar_integral_design=scalar.tolist(),
                independent_integer_design=independent.tolist(),vector_scaled_error=vector_error,
                scalar_scaled_error=scalar_error,cache_peak_bytes=field.cache_peak_bytes,counts=count_meta)
            rows.append(row);print(json.dumps(row),flush=True)
            del field;gc.collect()
    maximum=max(max(row['vector_scaled_error'],row['scalar_scaled_error']) for row in rows)
    passed=maximum<=contract['scaled_tolerance']
    result=dict(status='ACTUAL_TWO_BIN_INTEGER_SCALAR_ORACLE_PASS_NOT_ALL44_OR_STO_CLOSURE' if passed else 'REQUIRES_INVESTIGATION',
        created_UTC=datetime.now(timezone.utc).isoformat(),rows=rows,max_scaled_error=maximum,
        original_scaled_tolerance=contract['scaled_tolerance'],probe_b1=b,probe_p_halo=ph,
        physical_p_calibrated=False,actual_STO_noise_validated=False,all44_oracle_completed=False,
        production_model_modified=False,MCMC_released=False,IR=False,
        contract_sha256=digest(cp),source_sha256=digest(__file__),
        independent_oracle_source_sha256=digest(PART/'code/audit_post_rsd_noise_transport_coefficients.py'),
        CPU_affinity=sorted(os.sched_getaffinity(0)),elapsed_seconds=time.time()-start)
    with out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
