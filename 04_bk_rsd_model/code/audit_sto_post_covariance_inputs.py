"""模块大纲：独立Fid500输入缺口复查；逐盒检查正确配方与已知不合格替代。

只读输入，只在第四部分写证据；不改IC、不改STO方法、不下载、不缩放协方差。
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
from rsd_b000_contract import PART,digest
from inventory_sto_post_inputs import SOURCE,ALT,RAW,DM
from prepare_sto_post_validation_sources import DEST,RECIPE


def main():
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('single core read-only audit')
    rows=[];alternative=[]
    for index in range(1000,1500):
        raw=RAW/'fiducial'/f'real{index:03d}/groups_003/group_tab_003.0'
        dm=DM/'fiducial'/str(index)/'snapdir_003'
        candidates=[]
        for root in (SOURCE,DEST):
            path=root/'fiducial'/f'real{index:03d}/env_lam1_gsr2_thr15'
            complete=all((path/name).is_file() for name in ('post_sto_halo.npz','diagnostics.json','DONE'))
            matches=False
            if complete:
                meta=json.loads((path/'diagnostics.json').read_text())
                matches=all(meta['config'].get(k)==v for k,v in RECIPE.items())
                matches=matches and meta['component_config'].get('method')=='nenv' and meta['component_config'].get('base_weight')=='number'
            candidates.append({'path':str(path),'complete_candidate':complete,'recipe_matches':matches})
        row={'realization':index,'raw_fof_exists':raw.is_file(),'DM_path':str(dm),
             'DM_files_exist':bool(list(dm.glob('*.hdf5'))),'candidate_sources':candidates}
        row['required_input_available']=row['DM_files_exist'] or any(c['recipe_matches'] for c in candidates)
        rows.append(row)
        path=ALT/'fiducial'/f'real{index:03d}/full_mesh128_nenv_haloonly_lam3'
        record={'realization':index,'path':str(path),'source_exists':(path/'post_sto_halo.npz').is_file(),
                'diagnostics_exists':(path/'diagnostics.json').is_file()}
        if record['diagnostics_exists']:
            meta=json.loads((path/'diagnostics.json').read_text())
            record['different_recipe_fields']={k:{'required':v,'actual':meta['config'].get(k)}
                                               for k,v in RECIPE.items() if meta['config'].get(k)!=v}
            record['eligible_same_recipe']=not record['different_recipe_fields']
            record['diagnostics_sha256']=digest(path/'diagnostics.json')
        else:record['eligible_same_recipe']=False
        alternative.append(record)
    missing=[r['realization'] for r in rows if not r['required_input_available']]
    report={'status':'REQUIRED_INDEPENDENT_FID500_INPUTS_MISSING' if missing else 'PATHS_PRESENT_PENDING_FULL_SOURCE_IDENTITY_AUDIT',
            'created_UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),
            'CPU_affinity':sorted(os.sched_getaffinity(0)),'entry_sha256':digest(Path(__file__)),
            'required_realizations':[1000,1499],'required_count':500,'required_recipe':RECIPE,
            'raw_fof_count':sum(r['raw_fof_exists'] for r in rows),'correct_input_count':500-len(missing),
            'missing_count':len(missing),'missing_realizations':missing,'rows':rows,
            'alternate_source_count':sum(r['source_exists'] for r in alternative),
            'alternate_same_recipe_count':sum(r['source_exists'] and r['eligible_same_recipe'] for r in alternative),
            'ineligible_alternative_rows':alternative,
            'new_data_downloaded':False,'covariance_ICs_or_count_changed':False,
            'covariance_divided_by_realization_count':False,'volume_Gpc_h3':1,
            'action_needed':'provide the DM or matching matter-assisted GSR2 source path for fiducial1000..1499, or explicitly authorize a new data/sample plan; do not silently substitute halo-only or fewer covariance mocks'}
    root=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
    path=root/'COVARIANCE_INPUT_RECHECK.json'
    if path.exists():raise RuntimeError('preserve previous input recheck before a new revision')
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','ineligible_alternative_rows','missing_realizations')},indent=2),flush=True)


if __name__=='__main__':main()
