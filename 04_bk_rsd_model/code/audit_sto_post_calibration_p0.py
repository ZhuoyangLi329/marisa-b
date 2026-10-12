"""模块大纲：独立150盒校准输入完整性/来源验收与均值；绝不创建替代协方差。

只汇总已完成的 weighted-input RSD P0。物理 p 拟合需要另行冻结模型/校准
likelihood；此脚本不提供 p、fNL 后验或以150盒散布替代主 Fid500。
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
from rsd_b000_contract import PART,digest


def main():
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('single CPU audit only')
    out=PART/'outputs/sto_marisa_post_rsd_mean100_v1/calibration_holdout_p0'
    status=json.loads((out/'queue/STATUS.json').read_text());contract=json.loads((out/'CONTRACT.json').read_text())
    ids=list(range(100,250));sha=digest(out/'CONTRACT.json')
    if status['state']!='COMPLETE' or status['completed_realizations']!=ids:raise RuntimeError('holdout queue incomplete')
    if contract['realizations']!=ids or set(ids)&(set(range(100))|set(range(1000,1500))):raise ValueError('holdout independence broken')
    if (out/'CALIBRATION_P0_HOLDOUT.npz').exists():raise RuntimeError('aggregate exists; inspect without overwriting')
    for path,expected in contract['immutable_sources'].items():
        if digest(path)!=expected:raise ValueError('measurement code drift: '+path)
    window=json.loads((out/'POST_P0_WINDOW_AUDIT.json').read_text())
    if window['status']!='PASS' or window['window_sha256']!=digest(out/'P0_WINDOW.npz'):raise ValueError('P window drift')
    with np.load(out/'P0_WINDOW.npz') as z:
        nmodes=np.array(z['nmodes']);k=np.array(z['k']);edges=np.array(z['edges'])
    rows=[];matrices={key:[] for key in ('P0','P0_raw','P0_shot')};n2=[];n3=[]
    for index in ids:
        directory=out/'rows'/f'LC_p_real{index:03d}';done=json.loads((directory/'DONE.json').read_text())
        if done['state']!='COMPLETE' or done['contract_sha256']!=sha or done['realization']!=index:raise ValueError('measurement contract mismatch')
        if done['measurement_sha256']!=digest(directory/'P0.npz'):raise ValueError('P0 changed')
        source=Path(done['source_STO'])
        if done['source_STO_sha256']!=digest(source/'post_sto_halo.npz'):raise ValueError('STO source changed')
        if done['source_diagnostics_sha256']!=digest(source/'diagnostics.json'):raise ValueError('source recipe changed')
        if done['raw_fof']['source_raw_sha256']!=digest(Path(done['raw_fof']['source_raw'])):raise ValueError('raw FoF changed')
        if len(done['affinity'])!=1:raise ValueError('worker was not single core')
        with np.load(directory/'P0.npz') as z:
            np.testing.assert_array_equal(z['P0_nmodes'],nmodes);np.testing.assert_array_equal(z['edges'],edges)
            np.testing.assert_allclose(z['P0_k'],k,rtol=3e-6,atol=2e-8)
            for key in matrices:matrices[key].append(np.asarray(z[key]))
        n2.append(done['N2_self_moment']);n3.append(done['N3_self_moment'])
        rows.append({'realization':index,'measurement_sha256':done['measurement_sha256'],
                     'source_STO_sha256':done['source_STO_sha256'],'raw_sha256':done['raw_fof']['source_raw_sha256'],
                     'DONE_sha256':digest(directory/'DONE.json'),'host':done['host'],'affinity':done['affinity']})
    arrays={key:np.asarray(value) for key,value in matrices.items()}
    if arrays['P0'].shape!=(150,12) or not all(np.isfinite(x).all() for x in arrays.values()):raise ValueError('invalid aggregate matrix')
    np.testing.assert_allclose(arrays['P0_raw']-arrays['P0_shot'],arrays['P0'],rtol=3e-14,atol=1e-9)
    dest=out/'CALIBRATION_P0_HOLDOUT.npz'
    np.savez_compressed(dest,**arrays,P0_mean=arrays['P0'].mean(axis=0),realizations=ids,
                        N2_self=n2,N3_self=n3,k=k,edges=edges,nmodes=nmodes)
    report={'status':'INDEPENDENT_GSR2_HOLDOUT_P0_INPUTS_PASS_NO_P_FIT',
            'created_UTC':datetime.now(timezone.utc).isoformat(),'cosmology':'LC_p','truth':100,
            'realizations':ids,'n_data_mean':150,'n_P0':12,'kmax_P':.08,'z':.5,'LOS':'z',
            'tracer':'real-space matter-assisted GSR2 STO weights carried into RSD before Marisa',
            'role':'independent physical-p calibration input only, not new STO-only comparison',
            'rows':rows,'aggregate_sha256':digest(dest),'contract_sha256':sha,'entry_sha256':digest(Path(__file__)),
            'no_covariance_created':True,'main_Fid500_replaced':False,'main_mean100_replaced':False,
            'main_V_Gpc_h3':1,'physical_p_fit_or_MCMC_run':False,
            'next_gate':'freeze matched model and independent calibration likelihood; do not reuse historical GSR1 p or substitute this holdout for Fid500'}
    (out/'HOLDOUT_COMPLETION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','realizations')},indent=2),flush=True)


if __name__=='__main__':main()
