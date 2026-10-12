"""模块大纲：只读盘点目标 STO 配方缓存和 DM；区分路径存在与科学验收。

不下载 DM、不改变 IC、不改用 halo-only；只在第四部分写小型输入盘点。
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import socket
from tree_png_rsd_native import PART

DM=Path('/pscratch/sd/l/lzy/quijote-png/dm')
RAW=Path('/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5')
SOURCE=Path('/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_z0p5_r10_stovar_env_lam1_gsr2_thr15_fid500_lcp100_source')
ALT=Path('/pscratch/sd/l/lzy/reconstruction-png/result_quijote_halo_fullprod_nenv_fid4000_lcp50_lcm50_b1one_source')


def main():
    rows=[];counts={}
    for cosmo,ids,role in [('fiducial',range(100),'validation'),('LC_p',range(100),'validation'),
                           ('LC_m',range(100),'validation'),('fiducial',range(1000,1500),'covariance')]:
        label=f'{role}_{cosmo}';counts[label]={'total':len(ids),'halo_raw_present':0,
            'DM_directory_present':0,'DM_files_present':0,'matching_STO_cache_candidate':0,'missing_both_DM_and_cache':0}
        for real in ids:
            raw=RAW/cosmo/f'real{real:03d}/groups_003/group_tab_003.0'
            dm=DM/cosmo/str(real)/'snapdir_003'
            cache=SOURCE/cosmo/f'real{real:03d}/env_lam1_gsr2_thr15'
            candidate=all((cache/name).is_file() for name in ('post_sto_halo.npz','diagnostics.json','DONE'))
            dmfiles=bool(list(dm.glob('*.hdf5'))) if dm.is_dir() else False
            row={'cosmology':cosmo,'realization':real,'role':role,'halo_raw_path':str(raw),
                'halo_raw_present':raw.is_file(),'DM_path':str(dm),'DM_directory_present':dm.is_dir(),
                'DM_files_present':dmfiles,'STO_cache_path':str(cache),'matching_STO_cache_candidate':candidate,
                'missing_both_DM_and_cache':not(candidate or dmfiles)}
            rows.append(row)
            for key in counts[label]:
                if key!='total':counts[label][key]+=int(row[key])
    mismatched=[]
    for real in (0,1000,1499):
        p=ALT/'fiducial'/f'real{real:03d}/full_mesh128_nenv_haloonly_lam3/post_sto_halo.npz'
        mismatched.append({'path':str(p),'exists':p.is_file(),'eligible':False,
                           'reason':'halo-only lam3 is not matter-assisted env_lam1_gsr2_thr15'})
    missing=[r for r in rows if r['missing_both_DM_and_cache'] or not r['halo_raw_present']]
    payload={'status':'INPUT_ACTION_REQUIRED' if missing else 'PATH_INVENTORY_PASS_PENDING_CONTENT_AUDIT',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),
        'CPU_affinity':sorted(os.sched_getaffinity(0)),'counts':counts,'rows':rows,
        'ineligible_alternative_cache':mismatched,'missing_count':len(missing),
        'scope':'file/path presence only; cache rows/weights/config and DM headers must still be audited',
        'required_STO_recipe':{'method':'nenv','base_weight':'number','env_source':'dm','weight_source':'dm',
                               'lambda_th':1.,'gsr':2.,'density_radius':15.,'env_mesh':128,'recon_mesh':128},
        'covariance_ICs_changed':False,'covariance_divided_by_N':False,'new_data_downloaded':False}
    dest=PART/'outputs/sto_marisa_post_rsd_mean100_v1/INPUT_INVENTORY.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
