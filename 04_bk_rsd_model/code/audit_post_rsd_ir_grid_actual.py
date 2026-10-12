"""模块大纲：同一真实谱 benchmark 的 exact grid 加速对照；不复制数值流程。

仅在本进程替换 projector factory 和输出目录，原未加速进程/文件不改。
保留原门槛与全部源设置，再独立比较两个 engine 的实际低/高 bins。
"""
from pathlib import Path
import json
import sys
from datetime import datetime,timezone
from rsd_b000_contract import PART,digest
import audit_post_rsd_ir_actual as benchmark
from post_rsd_ir_grid import FastIRFFTProjection


def main():
    root=PART/'outputs/sto_marisa_post_rsd_mean100_v1';out=root/'fast_ir_projection_audit'
    if '--compare-only' in sys.argv:
        compare_reports(root);return
    out.mkdir(exist_ok=True);records=[]
    class RecordedFastProjection(FastIRFFTProjection):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            row={'meshsize':int(args[0]),'pair':[int(x) for x in args[2]],**self.damping_grid_statistics}
            records.append(row);print('DAMPING_GRID '+json.dumps(row),flush=True)
    benchmark.ROOT=out;benchmark.IRFFTProjection=RecordedFastProjection
    target=out/'IR_ACTUAL_SPECTRUM_AUDIT.json'
    try:benchmark.main()
    finally:
        if target.exists():
            report=json.loads(target.read_text());report['exact_grid_acceleration']={
                'records':records,'engine':'FastIRFFTProjection',
                'wrapper_sha256':digest(Path(__file__)),'engine_sha256':digest(PART/'code/post_rsd_ir_grid.py'),
                'same_benchmark_source_sha256':digest(PART/'code/audit_post_rsd_ir_actual.py'),
                'approximate_angular_average_used':False,'physical_parameters_changed':False}
            target.write_text(json.dumps(report,indent=2)+'\n')
    compare_reports(root)


def compare_reports(root):
    target=root/'fast_ir_projection_audit/IR_ACTUAL_SPECTRUM_AUDIT.json'
    baseline=root/'IR_ACTUAL_SPECTRUM_AUDIT.json'
    if baseline.exists():
        import numpy as np
        a=json.loads(baseline.read_text());b=json.loads(target.read_text());rows=[]
        if a['numerical_probe_only']!=b['numerical_probe_only'] or a['spectrum']!=b['spectrum']:raise ValueError('physical context mismatch')
        for aa,bb in zip(a['bins'],b['bins']):
            if aa['index']!=bb['index']:raise ValueError('bin order mismatch')
            for x,y in zip(aa['rows'],bb['rows']):
                if x['mesh']!=y['mesh'] or x['counts']!=y['counts']:raise ValueError('grid geometry mismatch')
                xa=np.asarray(x['design']);ya=np.asarray(y['design']);scale=np.maximum(1.,np.max(np.abs(xa),axis=1,keepdims=True))
                rows.append({'bin':aa['index'],'mesh':x['mesh'],'scaled_error':float(np.max(np.abs(xa-ya)/scale)),
                             'original_seconds':x['elapsed_seconds'],'accelerated_seconds':y['elapsed_seconds']})
        maximum=max(x['scaled_error'] for x in rows)
        result={'status':'ACTUAL_EXACT_GRID_ACCELERATION_PASS' if maximum<1e-9 else 'REQUIRES_INVESTIGATION',
                'created_UTC':datetime.now(timezone.utc).isoformat(),'rows':rows,'max_scaled_error':maximum,
                'predefined_tolerance':1e-9,'baseline_sha256':digest(baseline),'accelerated_sha256':digest(target),
                'comparison_source_sha256':digest(Path(__file__)),
                'whitened_error':None,'full_registry_verified':False}
        (root/'IR_GRID_ACCELERATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
        if maximum>=1e-9:raise SystemExit(1)
    else:print('Original baseline still running; cross-engine comparison pending.',flush=True)


if __name__=='__main__':main()
