"""模块大纲：原实空间Fid0..499候选输入只读盘点；不冻结或变更主拟合样本。

仅用标准库先验收raw FoF完整文件结构、同配方STO产物及元数据。
后续需完整halo身份检查；样本重叠须由用户明确选择，不能标为严格独立。
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import socket
import struct
import time

PART=Path(__file__).resolve().parents[1]
ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
SOURCE=Path('/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_source/fiducial')
RAW=Path('/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5/fiducial')
RECIPE={'env_source':'dm','weight_source':'dm','gsr_mpc_h':2.,'thr_mpc_h':15.,'lambda_th':1.,'env_mesh':128,'recon_mesh':128}


def main():
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('single core read-only audit only')
    print(json.dumps({'stage':'started','host':socket.gethostname(),'pid':os.getpid(),
                      'affinity':sorted(os.sched_getaffinity(0))}),flush=True)
    start=time.time();rows=[]
    for index in range(500):
        raw=RAW/f'real{index:03d}/groups_003/group_tab_003.0'
        directory=SOURCE/f'real{index:03d}/env_lam1_gsr2_thr15'
        item={'realization':index,'raw_path':str(raw),'source_path':str(directory)}
        item['raw_exists']=raw.is_file();item['raw_header_and_length_valid']=False
        if item['raw_exists']:
            item['raw_is_symlink']=raw.is_symlink();item['raw_size']=raw.stat().st_size
            with raw.open('rb') as stream:header=stream.read(24)
            if len(header)==24:
                ng,nt,ni,tid,nf=struct.unpack('<iiiQI',header)
                item['raw_groups']=ng
                item['raw_header_and_length_valid']=ng>0 and ng==nt and nf==1 and item['raw_size']==24+84*ng
        item['complete_STO_outputs']=all((directory/name).is_file() for name in ('post_sto_halo.npz','diagnostics.json','DONE'))
        item['same_recipe']=False
        diag=directory/'diagnostics.json'
        if diag.is_file():
            meta=json.loads(diag.read_text())
            differences={k:{'expected':v,'actual':meta['config'].get(k)} for k,v in RECIPE.items() if meta['config'].get(k)!=v}
            if meta['component_config'].get('method')!='nenv' or meta['component_config'].get('base_weight')!='number':
                differences['components']=meta['component_config']
            item['recipe_differences']=differences;item['same_recipe']=not differences
            item['diagnostics_sha256']=hashlib.sha256(diag.read_bytes()).hexdigest()
        rows.append(item)
        if index%50==49:print(json.dumps({'stage':'checked','count':len(rows),'last':index,'elapsed_seconds':time.time()-start}),flush=True)
    complete=all(r['raw_header_and_length_valid'] and r['complete_STO_outputs'] and r['same_recipe'] for r in rows)
    report={'status':'FID0_499_CANDIDATE_FILE_AND_RECIPE_AUDIT_PASS' if complete else 'FID0_499_CANDIDATE_REQUIRES_INSPECTION',
            'created_UTC':datetime.now(timezone.utc).isoformat(),'source_root':str(SOURCE),'rows':rows,
            'raw_complete_count':sum(r['raw_header_and_length_valid'] for r in rows),
            'same_recipe_complete_STO_count':sum(r['complete_STO_outputs'] and r['same_recipe'] for r in rows),
            'recipe':RECIPE,'candidate_covariance_ids':[0,499],
            'exact_catalog_overlap_with_fiducial_validation':[0,99],
            'possible_shared_phase_overlap_with_PNG_validation':[0,99],
            'possible_shared_phase_overlap_with_LCp_calibration':[100,249],
            'entire_velocity_and_halo_row_identity_verified':False,'sample_choice_approved_here':False,
            'main_contract_or_model_changed':False,'new_data_downloaded':False,'new_reconstruction_or_MCMC_run':False,
            'CPU_affinity':sorted(os.sched_getaffinity(0)),'elapsed_seconds':time.time()-start,
            'entry_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target=ROOT/'FID0_499_CANDIDATE_INPUT_AUDIT.json'
    if target.exists():raise RuntimeError('preserve prior candidate audit')
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
