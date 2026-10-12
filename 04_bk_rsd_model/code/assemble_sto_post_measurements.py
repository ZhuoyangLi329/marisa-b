"""模块大纲：严格验收700盒；两种B self定义分别汇总；未平均的Fid500 C。

复用唯一统计实现和逐盒测量验收；无缺行回退、协方差缩放或正则化。
同时保留matched及nonzero诊断，不擅自决定正式B定义或放行MCMC。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import numpy as np
from audit_sto_post_fid20_pilots import load_pilot,sha
from post_rsd_ensemble import EnsembleSelection,assemble_blocks,marginal_likelihood_arrays

PART=Path(__file__).resolve().parents[1]
ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT=ROOT/'ensemble_measurements'
CUTS=(.08,.10,.12,.14,.18,.20)
COUNTS=(9,14,20,27,35,44)


def selection_from_record(record):
    if (record['covariance_id_range_inclusive']!=[0,499] or record['n_covariance']!=500
        or record['target_id_range_inclusive']!=[0,99] or record['n_target_per_truth']!=100
        or record['overlap_policy']!='allow_explicit_target_overlap'
        or record['volume_Gpc_h3']!=1 or record['covariance_multiplier']!=1
        or record['covariance_is_of_mean'] or record['covariance_ddof']!=1):
        raise ValueError('explicit human-approved original ensemble and unaveraged C_single required')
    return EnsembleSelection(tuple(range(500)),overlap_policy='allow_explicit_target_overlap')


def directory(key):
    cosmo,index=key
    return ROOT/'pilots'/f'{cosmo}_real{index:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'


def readiness(selection):
    complete=[key for key in selection.measurement_keys if (directory(key)/'DONE.json').exists()]
    states=[]
    for queue,total in (('target300_measurement_queue',300),('fid100_499_measurement_queue',400)):
        path=ROOT/queue/'STATUS.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        states.append(dict(queue=queue,state=state.get('state'),completed=state.get('completed_count'),
            expected=total,workers=state.get('workers'),host=state.get('host'),pid=state.get('pid')))
    compute_path=ROOT/'compute32_measurement_queue/STATUS.json'
    migration_path=ROOT/'compute32_migration/MIGRATION.json'
    compute=json.loads(compute_path.read_text()) if compute_path.exists() else {}
    migrated=json.loads(migration_path.read_text()) if migration_path.exists() else {}
    via_compute=bool(compute)
    if via_compute:
        states=[dict(queue='compute32_measurement_queue',state=compute.get('state'),
            completed=compute.get('completed_count'),expected=700,workers=compute.get('workers'),
            host=compute.get('host'),pid=compute.get('pid'))]
    login_name='login_measurement_queue_v3' if (ROOT/'login_measurement_queue_v3/STATUS.json').exists() else 'login_measurement_queue_v2'
    login_path=ROOT/login_name/'STATUS.json'
    via_login=login_path.exists()
    if via_login:
        login=json.loads(login_path.read_text())
        proof=json.loads((ROOT/'login_resumption_20261009/MIGRATION.json').read_text())
        if not proof.get('compute_job_cancelled_before_running') or not proof.get('old_controller_confirmed_absent'):
            raise ValueError('new login resumption must not overlap original compute/control')
        if via_compute and (compute.get('state') not in ('COMPLETE','DRAINED_INCOMPLETE','FAILED') or compute.get('workers')):
            raise ValueError('compute dispatcher may still own measurement outputs')
        states=[dict(queue=login_name,state=login.get('state'),
            completed=login.get('completed_count'),expected=700,workers=login.get('workers'),
            host=login.get('host'),pid=login.get('pid'),starttime_ticks=login.get('starttime_ticks'))]
        via_compute=False
    ready=(len(complete)==700 and all(s['state']=='COMPLETE' and s['completed']==s['expected']
        and not s['workers'] for s in states)
        and (not via_compute or (migrated.get('state')=='LOGIN_MEASUREMENT_PARENTS_TERMINATED_WORKERS_COMPLETE'
            and migrated.get('original_parent_processes_confirmed_absent') is True)))
    return dict(artifact_ready=ready,complete_measurements=len(complete),required=700,queues=states,
        process_termination_verified_by_readiness=False,via_compute_migration=via_compute,via_login_resumption=via_login,
        login_resumption_queue=login_name if via_login else None)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check-ready',action='store_true');args=ap.parse_args()
    selected_path=ROOT/'ENSEMBLE_SELECTION.json'
    selection=selection_from_record(json.loads(selected_path.read_text()));selection_sha=sha(selected_path)
    ready=readiness(selection)
    if args.check_ready:print(json.dumps(ready),flush=True);return
    if not ready['artifact_ready']:raise ValueError('all700 complete measurements required; no partial covariance')
    if len(os.sched_getaffinity(0))!=1:raise ValueError('single-core aggregation only')
    for state in ready['queues']:
        if state['host']!=socket.gethostname():raise ValueError('verify original queue processes on original host')
        if state.get('starttime_ticks'):
            path=Path('/proc')/str(state['pid'])
            try:start=(path/'stat').read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:continue
            if start!=state['starttime_ticks']:continue
        try:os.kill(state['pid'],0)
        except ProcessLookupError:continue
        raise ValueError('queue process still exists; do not reuse its CPUs or declare production complete')
    queues=['target300_measurement_queue','fid100_499_measurement_queue']
    if ready['via_compute_migration']:queues.append('compute32_measurement_queue')
    if ready['via_login_resumption']:queues.append(ready['login_resumption_queue'])
    for queue in queues:
        contract=json.loads((ROOT/queue/'CONTRACT.json').read_text())
        for path,expected in contract['immutable_sources'].items():
            if sha(path)!=expected:raise ValueError('frozen measurement source changed: '+path)
    if OUT.exists():raise ValueError('existing ensemble must be inspected, never overwritten')
    source_audit=json.loads((ROOT/'VALIDATION_SOURCE_AUDIT.json').read_text())
    sources={(r['cosmology'],r['realization']):r for r in source_audit['source_rows']}
    reference=PART/'outputs/realspace_binning_reference/REALSPACE_BINNING.npz'
    with np.load(reference,allow_pickle=False) as z:ref={k:np.array(z[k]) for k in z.files}
    indices=ref['certified_indices']
    if tuple(int(np.sum(np.max(ref['B000_k'][indices],axis=1)<=cut+1e-12)) for cut in CUTS)!=COUNTS:
        raise ValueError('original six-cut B registry changed')
    window=ROOT/'calibration_holdout_p0/P0_WINDOW.npz'
    with np.load(window,allow_pickle=False) as z:p_edges=z['edges'].copy();p_modes=z['nmodes'].copy()
    pilot_sha=sha(PART/'code/run_sto_post_rsd_pilot.py')
    matched_rows={};nonzero_rows={};records=[];first=None;moments=[]
    for key in selection.measurement_keys:
        data,contract,done=load_pilot(directory(key))
        cosmo,index=key
        expected=dict(cosmology=cosmo,realization=index,recon_mesh=256,measurement_mesh=256,nran=30,
            self_contraction='matched',zero_atom_diagnostics=True,nested_randoms=True,grid_origin_fraction=0.)
        if any(contract['arguments'][name]!=value for name,value in expected.items()):raise ValueError('mixed measurement settings')
        if (done['entry_sha256']!=pilot_sha or len(done['CPU_affinity'])!=1
            or contract['shared_measure_source_sha256']!=sha(PART.parent/'01_reconstruction_methods/common/halo_b000_pipeline.py')):
            raise ValueError('source/resource drift')
        if index>=100:
            row_path=Path(contract['input_identity_row'])
            if sha(row_path)!=contract['input_identity_row_sha256']:raise ValueError('source row evidence drift')
            source=json.loads(row_path.read_text())
            if source['realization']!=index or source['status']!='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS':
                raise ValueError('unaudited covariance input')
        else:source=sources[key]
        if (source['source_sha256']!=contract['source_STO_sha256']
            or source['diagnostics_sha256']!=contract['source_diagnostics_sha256']
            or source['raw_fof']['source_raw_sha256']!=contract['raw_fof']['source_raw_sha256']):
            raise ValueError('measured inputs differ from audited source')
        if first is None:first=(data,contract)
        if contract['fixed_reconstruction']!=first[1]['fixed_reconstruction']:raise ValueError('mixed fixed reconstruction')
        np.testing.assert_array_equal(data['B_edges'],ref['edges']);np.testing.assert_array_equal(data['P_edges'],p_edges)
        np.testing.assert_allclose(data['B_post_k'],ref['B000_k'],rtol=0.,atol=5e-5)
        np.testing.assert_array_equal(data['B_post_nmodes'],first[0]['B_post_nmodes'])
        for label in ('P_weighted_input','P_post'):
            if data[label].shape!=(12,):raise ValueError('P bin dimension drift')
            np.testing.assert_array_equal(data[label+'_nmodes'],p_modes)
        if data['B_post'].shape!=(120,):raise ValueError('B registry dimension drift')
        common=dict(P_weighted_input=data['P_weighted_input'],P_post=data['P_post'])
        matched_rows[key]={**common,'B_post':data['B_post'][indices]}
        nonzero_rows[key]={**common,'B_post':data['B_post_nonzero_only_counterfactual'][indices]}
        moments.append([source['N2_data_self'],source['N3_data_self'],contract['raw_fof']['n_halo']])
        records.append(dict(cosmology=cosmo,realization=index,contract_sha256=done['contract_sha256'],
            measurement_sha256=done['measurement_sha256'],post_halos_sha256=done['post_halos_sha256'],
            source_STO_sha256=source['source_sha256'],raw_fof_sha256=source['raw_fof']['source_raw_sha256']))
    sizes=dict(P_weighted_input=12,P_post=12,B_post=44)
    ensembles={label:assemble_blocks(selection,rows,sizes) for label,rows in
        (('matched',matched_rows),('nonzero_diagnostic',nonzero_rows))}
    # 校准C_PP、各B-only和P+B都从C子块构造，Hartlap按各自数据维数；不截取precision。
    checks=[]
    for label,ensemble in ensembles.items():
        p_start,p_stop=ensemble['blocks']['P_post'];b_start,_=ensemble['blocks']['B_post']
        for cut,count in zip(CUTS,COUNTS):
            mask=np.max(ref['B000_k'][indices],axis=1)<=cut+1e-12
            b_indices=tuple((b_start+np.flatnonzero(mask)).tolist())
            for name,selected in (('B_only',b_indices),('P_plus_B',tuple(range(p_start,p_stop))+b_indices)):
                arrays=marginal_likelihood_arrays(ensemble,selected,'fiducial')
                checks.append(dict(B_definition=label,observable=name,kmax_B=cut,bins=len(selected),
                    Hartlap_factor=arrays['Hartlap_factor'],cholesky_pass=True))
        marginal_likelihood_arrays(ensemble,tuple(range(12)),'LC_p')
    if sha(selected_path)!=selection_sha:raise ValueError('selection changed during aggregation')
    OUT.mkdir()
    paths={}
    for label,ensemble in ensembles.items():
        path=OUT/f'{label.upper()}_ENSEMBLE.npz'
        np.savez_compressed(path,matrix=ensemble['matrix'],covariance_single_box=ensemble['covariance_single_box'],
            target_vectors=np.array([ensemble['targets'][c] for c in ('fiducial','LC_p','LC_m')]),
            measurement_keys_json=np.array([json.dumps(k) for k in ensemble['measurement_keys']]),
            covariance_indices=ensemble['covariance_indices'],B_indices=indices,B_k=ref['B000_k'][indices],
            P_edges=p_edges,B_edges=ref['edges'],data_N2_N3_nhalo=np.array(moments))
        paths[label]=dict(path=str(path),sha256=sha(path))
    report=dict(status='ALL700_POST_MEASUREMENTS_AND_C500_SINGLEBOX_ARRAYS_VERIFIED_NOT_MCMC_RELEASE',
        created_UTC=datetime.now(timezone.utc).isoformat(),statistics=selection.metadata(),blocks=sizes,
        block_slices=ensembles['matched']['blocks'],ensemble_files=paths,rows=records,marginal_checks=checks,
        formal_B_definition_chosen=False,physical_p_calibrated=False,noise_model_closed=False,
        primary_MCMC_released=False,paired_B_definitions_not_combined_into_one_likelihood=True,
        covariance_ddof=1,covariance_multiplier=1.,volume_Gpc_h3=1.,covariance_is_of_mean=False,
        selection_sha256=selection_sha,reference_sha256=sha(reference),P_window_sha256=sha(window),
        entry_sha256=sha(Path(__file__)),fixed_reconstruction=first[1]['fixed_reconstruction'])
    with (OUT/'AUDIT.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
