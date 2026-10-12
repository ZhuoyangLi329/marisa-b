"""模块大纲：当前目标逐项完成审计；数值核验不替代缺失输入、协方差或MCMC。

复算44-bin双grid误差，核验源/旧结果保护和主统计合同；只输出未完成项证据。
不替代goal状态工具，也不自行改变样本或关闭任何科学验收门。
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
from rsd_b000_contract import PART,digest,selected

ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'


def read(path):return json.loads(path.read_text())


def main():
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('single CPU audit only')
    target=ROOT/'INPUT_DEPENDENCY_GATE_AUDIT.json'
    if target.exists():raise RuntimeError('preserve existing dependency-gate evidence')
    core=read(ROOT/'SOURCE_AUDIT.json')
    if core['status']!='SOURCE_TESTS_PASS_PRODUCTION_NOT_RELEASED' or core['failures'] or core['errors']:raise ValueError('core audit failed')
    for path,sha in core['sources'].items():
        if digest(path)!=sha:raise ValueError('core source changed since unit audit: '+path)
    for path,hashes in core['protected_old_outputs'].items():
        if digest(PART/path)!=hashes['expected']:raise ValueError('protected old result/source changed')
    if digest(PART/'work/build/libmarisa_tree_rsd.so')!=core['old_library_sha256']:raise ValueError('old library changed')
    if digest(PART/'work/build/libmarisa_tree_rsd_fixedrec.so')!=core['new_library_sha256']:raise ValueError('new library changed')
    directory=ROOT/'ir_full_registry_audit';ir=read(directory/'REGISTRY_AUDIT.json')
    status=read(directory/'STATUS.json');contract=read(directory/'CONTRACT.json');csha=digest(directory/'CONTRACT.json')
    indices=selected(.20).tolist()
    if status['state']!='COMPLETE' or status['completed_bins']!=indices or len(indices)!=44:raise ValueError('full registry not complete')
    if ir['contract_sha256']!=csha or contract['indices']!=indices:raise ValueError('IR contract changed')
    for path,sha in contract['immutable_sources'].items():
        if digest(path)!=sha:raise ValueError('IR source changed: '+path)
    errors=[];powers=np.asarray(contract['fNL_coefficient_scaling'])
    for row,index in zip(ir['rows'],indices):
        if row['index']!=index or row!=read(directory/'bins'/f'bin{index:03d}.json'):raise ValueError('IR row mismatch')
        levels=[read(directory/'grids'/f'bin{index:03d}_m{n:03d}.json') for n in contract['pair_meshes'][str(index)]]
        if any(r['contract_sha256']!=csha for r in levels):raise ValueError('grid contract mismatch')
        a,b=(np.asarray(r['design']) for r in levels)
        if a.shape!=(3,3) or b.shape!=(3,3) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid IR matrices')
        if levels[0]['counts']!=levels[1]['counts'] or levels[0]['denominator']!=levels[1]['denominator']:raise ValueError('mode count differs')
        plain=float(np.max(np.abs(b-a)/np.maximum(1.,np.max(np.abs(b),axis=1,keepdims=True))))
        weighted=float(np.max(np.abs(b-a)*powers/np.maximum(1.,np.max(np.abs(b)*powers,axis=1,keepdims=True))))
        if max(plain,weighted)>=1e-9:raise ValueError('independent double-grid verification failed')
        np.testing.assert_array_equal(b,np.asarray(row['fine_design']));errors.append((plain,weighted))
    if len(ir['rows'])!=44:raise ValueError('missing or extra IR rows')
    validation=read(ROOT/'VALIDATION_SOURCE_AUDIT.json')
    if validation['source_counts']!={'fiducial':100,'LC_p':100,'LC_m':100}:raise ValueError('validation source counts changed')
    holdout=read(ROOT/'calibration_holdout_p0/HOLDOUT_COMPLETION_AUDIT.json')
    if len(holdout['rows'])!=150 or digest(ROOT/'calibration_holdout_p0/CALIBRATION_P0_HOLDOUT.npz')!=holdout['aggregate_sha256']:
        raise ValueError('holdout source proof changed')
    missing=read(ROOT/'COVARIANCE_INPUT_RECHECK.json');progress=read(ROOT/'GOAL_PROGRESS.json')
    if missing['missing_count']!=500 or missing['correct_input_count']!=0:raise ValueError('input state changed; reassess blocker')
    stats=progress['statistics']
    if stats['covariance_count']!=500 or stats['covariance_of_mean'] or stats['volume_Gpc_h3']!=1:raise ValueError('main covariance contract changed')
    if stats['target_mean_count_per_truth']!=100:raise ValueError('main target changed')
    chains=list(ROOT.rglob('SUMMARY.json'))
    if chains or progress['primary_MCMC_cases_started']!=0:raise ValueError('new inference artifacts exist; inspect rather than assuming no MCMC')
    report={'status':'OBJECTIVE_INCOMPLETE_REQUIRED_INPUT_DEPENDENCY',
            'created_UTC':datetime.now(timezone.utc).isoformat(),'goal_complete':False,'entry_sha256':digest(Path(__file__)),
            'source_tests':core['tests_run'],'protected_old_outputs_and_libraries_unchanged':True,
            'IR_double_grid_independently_recomputed':{'bins':44,'max_plain_error':max(x[0] for x in errors),
                  'max_fNL_range_weighted_error':max(x[1] for x in errors),'scope':'one numerical parameter point, not full model or science validation'},
            'validation_STO_source_counts':validation['source_counts'],'independent_calibration_P0_count':150,
            'physical_p_fit_complete':False,'post_ensemble_and_covariance_complete':False,
            'post_stochastic_and_statistical_numerical_gates_complete':False,
            'main_MCMC_required':{'B_only':18,'P0_only':3,'P_plus_B':18},'main_MCMC_completed':0,
            'same_data_IR_effect_control_complete':False,'covariance_correct_input_count':0,'covariance_required_count':500,
            'known_alternate_same_recipe_count':missing['alternate_same_recipe_count'],'statistics_preserved':stats,
            'required_next_input':'fiducial1000..1499 DM or matching matter-assisted GSR2 STO cache path; otherwise explicit authority for a revised data/sample plan',
            'not_authorized_or_performed':['fewer covariance mocks','different covariance ICs','halo-only substitution','large new data acquisition','covariance averaging'],
            'evidence_hashes':{name:digest(ROOT/name) for name in ('SOURCE_AUDIT.json','VALIDATION_SOURCE_AUDIT.json',
                  'calibration_holdout_p0/HOLDOUT_COMPLETION_AUDIT.json','ir_full_registry_audit/REGISTRY_AUDIT.json',
                  'COVARIANCE_INPUT_RECHECK.json','IR_PNG_TRUNCATION_DIAGNOSTIC.json')}}
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
