"""模块大纲：p校准正负PNG/校准范围敏感性诊断，不改已冻结主p。

每个sign都固定其真实fNL，仅作独立blind MAP；不会为各真值分别调主p。
全部C仍单盒，Hartlap按各自P点数；辅助cut不能代替原.08校准结果。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
from iminuit import Minuit
from rsd_b000_contract import PART,digest,save_json
from audit_all_available_rsd_pcal_v1 import direct_p
import run_all_available_rsd_pcal_v1 as calibration

def read(path):return json.loads(Path(path).read_text())
def fit(target,covariance,templates,fnl):
    n=len(target);alpha=(500-n-2)/499;chol=np.linalg.cholesky(covariance/alpha);candidates=[]
    def objective(p,b1,Pshot):
        white=np.linalg.solve(chol,direct_p(fnl,b1,p,templates)+Pshot-target)
        return float(white@white)
    for p in (.5,1.5,2.5):
        for b in (1.25,1.93,2.8):
            m=Minuit(objective,p,b,0.);m.errordef=1;m.tol=1e-5
            for name,bound,error in zip(m.parameters,calibration.BOUNDS,(.1,.1,1000.)):
                m.limits[name]=bound;m.errors[name]=error
            m.migrad(ncall=12000)
            if not m.valid:m.simplex(ncall=6000);m.migrad(ncall=12000)
            m.hesse();candidates.append(m)
    good=[m for m in candidates if m.valid and m.covariance is not None and np.isfinite(m.fval)]
    if not good:raise ValueError('all diagnostic p/sign/range blind starts failed')
    m=min(good,key=lambda x:x.fval)
    return dict(physical_p_MAP=float(m.values[0]),b1_MAP=float(m.values[1]),Pshot_MAP=float(m.values[2]),
        chi2_single_box_forecast=float(m.fval),P_bins=n,Hartlap_factor=alpha,
        p_HESSE_sigma_single_box=float(np.sqrt(np.array(m.covariance)[0,0])),optimizer_valid=bool(m.valid))

def main():
    out=calibration.OUT/'P_CALIBRATION_SIGN_RANGE_DIAGNOSTIC.json'
    if out.exists():raise ValueError('diagnostic already exists; preserve it')
    report_path=calibration.STO/'ensemble_measurements/AUDIT.json';report=read(report_path)
    matrix_path=Path(report['ensemble_files']['matched']['path'])
    if digest(matrix_path)!=report['ensemble_files']['matched']['sha256']:raise ValueError('actual C500 changed')
    with np.load(matrix_path,allow_pickle=False) as z:
        matrix=z['matrix'].copy();cov=z['covariance_single_box'].copy()
        keys=[tuple(json.loads(str(x))) for x in z['measurement_keys_json']]
    pos={key:i for i,key in enumerate(keys)};checks=[];sources=[Path(__file__),report_path,matrix_path]
    for stage in ('pre','post_input','post'):
        field='pre' if stage=='pre' else 'post'
        cal=calibration.stage_root(field)/'cases'/calibration.case_name(field)
        with np.load(cal/'likelihood.npz',allow_pickle=False) as z:templates=z['templates'].copy()
        with np.load(calibration.PRE/'P0_LATTICE.npz',allow_pickle=False) as z:k=z['k'].copy()
        sources.append(cal/'likelihood.npz')
        for truth,fnl in (('LC_p',100),('LC_m',-100)):
            if stage=='pre':
                d=calibration.OUT/'pre_P0_only_all_available_p/cases'/(truth+'_pre_P0_all_available_p')
                with np.load(d/'likelihood.npz',allow_pickle=False) as z:target=z['target'].copy();covariance=z['covariance'].copy()
                sources.extend([d/'contract.json',d/'likelihood.npz'])
            else:
                columns=np.arange(12) if stage=='post_input' else np.arange(12,24)
                rows=[pos[truth,i] for i in range(100)];target=matrix[rows][:,columns].mean(axis=0)
                covariance=cov[np.ix_(columns,columns)]
            for cut in (.04,.06,.08):
                mask=k<=cut+1e-12
                r=fit(target[mask],covariance[np.ix_(mask,mask)],templates[:,mask],fnl)
                checks.append(dict(r,stage=stage,truth_fNL=fnl,target_count=100,
                    matched_target_ids=list(range(100)),kmax_P_calibration_diagnostic=cut))
    # 正校准全500/250的范围控制，不用matched100子样本替代主p。
    allpositive=[]
    for stage in ('pre','post'):
        d=calibration.stage_root(stage)/'cases'/calibration.case_name(stage);meta=read(d/'contract.json')
        with np.load(d/'likelihood.npz',allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
        with np.load(calibration.PRE/'P0_LATTICE.npz',allow_pickle=False) as z:k=z['k'].copy()
        for cut in (.04,.06,.08):
            mask=k<=cut+1e-12;r=fit(a['target'][mask],a['covariance'][np.ix_(mask,mask)],a['templates'][:,mask],100)
            allpositive.append(dict(r,stage=stage,target_count=meta['calibration_target_count'],kmax_P=cut))
    result=dict(created_UTC=calibration.stamp(),status='ACTUAL_SIGN_AND_RANGE_MAP_DIAGNOSTICS_NOT_MAIN_P_REPLACEMENT',
        matched100_controls=checks,all_available_positive_controls=allpositive,
        calibration_fNL_is_fixed_not_fitted=True,main_p_not_changed=True,
        main_mean100_Csingle500_V1_noIR_not_changed=True,no_covariance_averaging=True,
        no_per_truth_p_tuning_or_cut_selection_to_force_fNL_recovery=True,
        fNL_zero_does_not_calibrate_p=True,sources_sha256={str(p):digest(p) for p in dict.fromkeys(sources)})
    save_json(out,result)
    print(json.dumps(dict(status=result['status'],all_positive=allpositive,
        sign_controls_at_original_cut=[x for x in checks if x['kmax_P_calibration_diagnostic']==.08])),flush=True)

if __name__=='__main__':main()
