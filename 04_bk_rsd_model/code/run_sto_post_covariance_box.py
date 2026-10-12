"""模块大纲：已验收Fid100..499单盒测量；原STO/raw身份；原P/B/重构数值入口。

Fid0..99由target队列测一次、承担两个统计角色。本入口不平均协方差，
只将来源审计后的同字节数据交给原pilot数值程序，并明确metadata恢复路径。
"""
import argparse
import json
import os
from pathlib import Path
import sys
from run_sto_post_target_box import PART,ROOT,sha,stage_source


def read_row(path,index):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path.parent.name!='rows':raise ValueError('owned source audit row required')
    blob=path.read_bytes();row=json.loads(blob)
    if row['realization']!=index or row['status']!='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS':
        raise ValueError('selected source row has not passed complete identity/recipe audit')
    row['cosmology']='fiducial'
    return row,blob


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--realization',type=int,required=True)
    ap.add_argument('--source-row',required=True);args=ap.parse_args()
    if len(os.sched_getaffinity(0))!=1 or not 100<=args.realization<500:
        raise ValueError('single-core Fid100..499 only; target0..99 measured by its own queue')
    selection=json.loads((ROOT/'ENSEMBLE_SELECTION.json').read_text())
    if selection['covariance_id_range_inclusive']!=[0,499] or selection['overlap_policy']!='allow_explicit_target_overlap':
        raise ValueError('human-confirmed original ensemble selection required')
    row,row_blob=read_row(args.source_row,args.realization)
    from post_rsd_blob_catalog import read_blob,catalog_from_blobs
    import run_sto_post_rsd_pilot as pilot
    old_diag=Path(row['source'])/'diagnostics.json'
    consumed_diag=Path(row.get('diagnostics_consumed_path',old_diag))
    def read(path):return read_blob(consumed_diag if Path(path)==old_diag else path)
    previous=pilot.SOURCE,pilot.catalog,pilot.save_json
    with stage_source(row,read) as (temporary,raw):
        def catalog(cosmology,index):
            if (cosmology,index)!=('fiducial',args.realization):raise ValueError('unexpected input identity')
            return catalog_from_blobs(cosmology,index,raw,raw_source=row['raw_fof']['source_raw'])
        def save(path,value):
            if path.name=='CONTRACT.json':
                value.update(source_STO=row['source'],source_diagnostics_consumed_path=str(consumed_diag),
                    input_identity_row=str(args.source_row),input_identity_row_sha256=sha(row_blob),
                    input_staging='original bytes through unchanged reader; local temporary storage',
                    IO_adapter_sha256=sha(Path(__file__).read_bytes()),
                    ensemble_selection_sha256=sha((ROOT/'ENSEMBLE_SELECTION.json').read_bytes()),
                    original_inputs_modified=False,covariance_is_of_mean=False,volume_Gpc_h3=1.)
            return previous[2](path,value)
        pilot.SOURCE,pilot.catalog,pilot.save_json=temporary,catalog,save
        sys.argv=[pilot.__file__,'--cosmology','fiducial','--realization',str(args.realization),
            '--recon-mesh','256','--measurement-mesh','256','--nran','30','--self-contraction','matched',
            '--zero-atom-diagnostics','--nested-randoms']
        try:pilot.main()
        except Exception:
            import traceback
            if pilot.PILOT_STATUS_PATH is not None:
                state=json.loads(pilot.PILOT_STATUS_PATH.read_text())
                state.update(state='FAILED',error=traceback.format_exc());previous[2](pilot.PILOT_STATUS_PATH,state)
            raise
        finally:pilot.SOURCE,pilot.catalog,pilot.save_json=previous


if __name__=='__main__':main()
