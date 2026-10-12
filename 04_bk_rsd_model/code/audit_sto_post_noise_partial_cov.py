"""模块大纲：与p校准并行的实际权重/零腿白化/leading噪声幅度敏感度检查。

复用已核验44-bin算子和实际测量；不把N2认作Nraw，不把幅度扫描当校准。
CPU14单核，小内存；同一冻结粗协方差cohort，原500盒流程和B噪声门不改。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime,timezone
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import traceback
import numpy as np
from rsd_b000_contract import PART,digest,save_json
from audit_sto_post_fid20_pilots import load_pilot
from run_sto_post_compute_queue import directory
from run_sto_post_covariance_queue import source_row
from build_post_rsd_noise_transport_templates import evaluate
from post_rsd_general_noise_transport import combine_white_operators

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'noise_actual_partial_cov_v1'
CAL=ROOT/'diagnostic_partial_cov_p_mcmc_20261010_v2'
OPS=ROOT/'noise_transport_templates_v1'


def read(p):return json.loads(Path(p).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()


def main():
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14}:raise ValueError('login26 CPU14 required')
    if OUT.exists():raise ValueError('preserve previous actual noise diagnostic')
    cal=read(CAL/'CONTRACT.json');ids=cal['covariance_ids'];n=len(ids)
    if n!=cal['n_covariance'] or len(set(ids))!=n:raise ValueError('exact frozen partial covariance cohort required')
    for path,want in cal['immutable_inputs'].items():
        if digest(path)!=want:raise ValueError('partial calibration source drift')
    completion=read(OPS/'COMPLETION_AUDIT.json');independent=read(OPS/'INDEPENDENT_COEFFICIENT_AUDIT.json')
    op_path=OPS/'OPERATOR_COEFFICIENTS.npz'
    if (completion['bins']!=44 or independent['bins']!=44
        or digest(op_path)!=completion['coefficients_sha256']
        or digest(op_path)!=independent['coefficients_sha256']):raise ValueError('actual full44 operator coefficients required')
    for rel in ['noise_scalar_all44_v1/AUDIT.json','noise_cross_all44_v1/AUDIT.json']:
        q=read(ROOT/rel)
        if q['bins']!=44 or q['maximum_scaled_error']>q['original_scaled_tolerance']:
            raise ValueError('independent full44 integer/convolution check missing')
    with np.load(op_path,allow_pickle=False) as z:coeff=z['coefficients'].copy();indices=z['indices'].copy()
    with np.load(PART/'outputs/realspace_binning_reference/REALSPACE_BINNING.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(indices,z['certified_indices']);bk=z['B000_k'][indices].copy()
    target_audit=read(ROOT/'target300_completion_v1/AUDIT.json');target_path=Path(target_audit['target_matrix_path'])
    if digest(target_path)!=target_audit['target_matrix_sha256']:raise ValueError('original300 target means changed')
    with np.load(target_path,allow_pickle=False) as z:
        target=z['matched_target_vectors'].copy();counter=z['nonzero_diagnostic_target_vectors'].copy()
        moments=z['data_N2_N3_nhalo'].copy()
    OUT.mkdir();state=dict(state='RUNNING',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[14],
        started_UTC=stamp(),completed_fid=0,n_covariance=n)
    def update(**values):
        state.update(values,updated_UTC=stamp());save_json(OUT/'STATUS.json',state)
    update();matched=[];nonzero=[];weight_checks=[]
    try:
        for index in ids:
            data,meta,done=load_pilot(directory(('fiducial',index)))
            with np.load(directory(('fiducial',index))/'POST_HALOS.npz',allow_pickle=False) as z:
                weights=np.asarray(z['weights'],dtype=float)
            total=weights.sum();n2=float(1e9*np.sum(weights**2)/total**2);n3=float(1e18*np.sum(weights**3)/total**3)
            if index<100:
                expected=moments[index,:2]
            else:
                row=source_row(index)[1];expected=[row['N2_data_self'],row['N3_data_self']]
            np.testing.assert_allclose([n2,n3],expected,rtol=8*np.finfo(float).eps,atol=0)
            common=np.r_[data['P_weighted_input'],data['P_post']]
            matched.append(np.r_[common,data['B_post'][indices]])
            nonzero.append(np.r_[common,data['B_post_nonzero_only_counterfactual'][indices]])
            weight_checks.append(dict(realization=index,N2=n2,N3=n3,measurement_sha256=done['measurement_sha256']))
            if len(matched)%25==0:update(completed_fid=len(matched))
        c=np.cov(np.array(matched),rowvar=False,ddof=1)
        cnon=np.cov(np.array(nonzero),rowvar=False,ddof=1)
        with np.load(CAL/'FROZEN_PARTIAL_P_ENSEMBLE.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(c[:24,:24],z['covariance_single_box'],rtol=2e-13,atol=1e-7)
        # 数值探针p不替代正在MCMC校准的物理p；这里只量化Nraw假设的敏感度。
        probe_b=2.2280546353635335;probe_p=1.31
        operators=np.stack([evaluate(coeff[...,j],probe_b,probe_p) for j in range(44)],axis=-1)
        rows=[];cuts=(.08,.10,.12,.14,.18,.20)
        for t,(cosmo,fnl) in enumerate([('fiducial',0),('LC_p',100),('LC_m',-100)]):
            n2_values=moments[t*100:(t+1)*100,0]
            delta=counter[t,24:]-target[t,24:];powers=np.array([1.,fnl,fnl*fnl])
            for cut in cuts:
                mask=np.max(bk,axis=1)<=cut+1e-12;bidx=24+np.flatnonzero(mask)
                alpha=(n-len(bidx)-2)/(n-1);chol=np.linalg.cholesky(c[np.ix_(bidx,bidx)]/alpha)
                cholnon=np.linalg.cholesky(cnon[np.ix_(bidx,bidx)]/alpha)
                zero_norm=float(np.linalg.norm(np.linalg.solve(chol,delta[mask])))
                responses=[]
                for ratio in (.5,1.,1.5,2.):
                    # 平均逐盒组合，保留mean(N2²)，不使用square(mean(N2))。
                    designs=np.mean([combine_white_operators(operators,noise_data_raw=ratio*v,n2_data_self=v)
                        for v in n2_values],axis=0)
                    signal=powers@designs
                    responses.append(dict(assumed_Nraw_over_N2=ratio,
                        single_box_whitened_norm=float(np.linalg.norm(np.linalg.solve(cholnon,signal[mask])))))
                rows.append(dict(cosmology=cosmo,kmax_B=cut,B_bins=int(mask.sum()),Hartlap_factor=alpha,
                    zero_atom_single_box_whitened_norm=zero_norm,white_Nraw_hypothesis_sensitivity=responses))
        report=dict(status='ACTUAL_WEIGHT_AND_PARTIAL_COV_NOISE_DIAGNOSTICS_COMPLETE_NOT_STO_CLOSURE',
            created_UTC=stamp(),n_covariance=n,covariance_ids=ids,weight_checks=weight_checks,rows=rows,
            target_count_per_truth=100,covariance_is_of_mean=False,covariance_multiplier=1.,volume_Gpc_h3=1.,
            probe_b1=probe_b,probe_p_not_calibration=probe_p,
            noise_raw_power_measured_or_calibrated=False,white_noise_ratios_are_hypotheses_only=True,
            noise_formula='Nraw*Kcross+(Nraw-N2)*Ksame+Nraw*(Nraw-N2)*KNN',
            formal_B_definition_selected=False,connected_stochastic_response_validated=False,
            noise_IR_dressing_validated=False,B_MCMC_released=False,
            next_required=['actual raw STO noise spectrum and anisotropy calibration',
                'connected weighted stochastic response and PNG dependence',
                'final physical-p-dependent projected model check'],
            operator_coefficients_sha256=digest(op_path),partial_p_contract_sha256=digest(CAL/'CONTRACT.json'),
            source_sha256=digest(Path(__file__)),CPU_affinity=[14],
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        save_json(OUT/'DIAGNOSTICS.json',report);update(state='COMPLETE',completed_fid=n)
        print(json.dumps({k:v for k,v in report.items() if k not in ['rows','weight_checks','covariance_ids']}),flush=True)
    except Exception:update(state='FAILED_PRESERVED',error=traceback.format_exc());raise


if __name__=='__main__':
    if sys.argv[1:]==['--detach']:
        log=ROOT/'noise_actual_partial_cov_v1.log';launch=ROOT/'noise_actual_partial_cov_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect existing noise diagnostic')
        if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14}:raise ValueError('single CPU14 launcher required')
        with log.open('x') as stream:
            p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(launch,dict(pid=p.pid,host=socket.gethostname(),CPU_affinity=[14],created_UTC=stamp(),log=str(log)))
        print(json.dumps(read(launch)),flush=True)
    elif not sys.argv[1:]:main()
    else:raise ValueError('only --detach or direct audit supported')
