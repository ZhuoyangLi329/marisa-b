"""模块大纲：独立验收全500来源报告及实际measurement准入，不创建协方差。

逐行核对原STO/FoF/diagnostics哈希和N2/N3、目标0..99回归、425无标记字节、
471同payload路由及原核验进程退出。保留样本重叠，不宣称独立mock recovery。
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket

PART=Path(__file__).resolve().parents[1]
ROOT=PART/'outputs/sto_marisa_post_rsd_mean100_v1'
AUDIT=ROOT/'fid0_499_raw_source_audit_routed_v3'
ADMISSION=ROOT/'fid0_499_raw_source_audit_routed_v1/rows'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def science_equal(a,b,*,derived_moment_roundoff=False):
    for key in ('source','source_sha256','diagnostics_sha256','n_halo'):
        if a[key]!=b[key]:raise ValueError('admission science input changed: '+key)
    if a['raw_fof']['source_raw_sha256']!=b['raw_fof']['source_raw_sha256']:
        raise ValueError('admission FoF bytes changed')
    # 仅跨旧target报告允许有限float64舍入；全源文件SHA必须先逐位相同。
    for key in ('N2_data_self','N3_data_self'):
        if derived_moment_roundoff:
            equal=math.isclose(a[key],b[key],rel_tol=8*math.ulp(1.),abs_tol=0.)
        else:equal=a[key]==b[key]
        if not equal:raise ValueError('admission weighted moment changed: '+key)


def validate_moments(row):
    # 二次矩必须正；有符号STO的三次矩只要求有限，不另加原审计没有的正性先验。
    if (not math.isfinite(row['N2_data_self']) or row['N2_data_self']<=0
        or not math.isfinite(row['N3_data_self']) or row['n_halo']<=0
        or row['n_halo']!=row['raw_fof']['n_halo']):raise ValueError('invalid weighted moments/halo count')


def main():
    if len(os.sched_getaffinity(0))!=1:raise ValueError('single-core independent admission audit required')
    status=json.loads((AUDIT/'STATUS.json').read_text())
    if (status['state']!='COMPLETE' or status['completed']!=list(range(500)) or status['failed']
        or status['host']!=socket.gethostname()):raise ValueError('original complete500/0fail status required')
    try:os.kill(status['pid'],0)
    except ProcessLookupError:pass
    else:raise ValueError('original input auditor still exists; verify actual termination')
    report=json.loads((AUDIT/'AUDIT.json').read_text());contract=json.loads((AUDIT/'CONTRACT.json').read_text())
    if (report['status']!='ALL500_EXISTING_REALSPACE_SOURCES_PAYLOAD_IDENTITY_PASS'
        or report['passed_count']!=500 or report['failed_realizations']
        or report['contract_sha256']!=sha(AUDIT/'CONTRACT.json')
        or contract['candidate_realizations']!=list(range(500))
        or report['metadata_replay_realizations']!=[127,293,471]
        or not report['original_STO_weights_unchanged']):raise ValueError('full source report/contract mismatch')
    for path,expected in contract['immutable_sources'].items():
        if sha(path)!=expected:raise ValueError('frozen source audit evidence changed: '+path)
    rows=report['rows']
    if [row['realization'] for row in rows]!=list(range(500)):raise ValueError('missing/duplicated source IDs')
    targets=json.loads((ROOT/'VALIDATION_SOURCE_AUDIT.json').read_text())
    prior={r['realization']:r for r in targets['source_rows'] if r['cosmology']=='fiducial'}
    if set(prior)!=set(range(100)):raise ValueError('prior mean100 target identities incomplete')
    selected=json.loads((ROOT/'ENSEMBLE_SELECTION.json').read_text())
    if (selected['covariance_id_range_inclusive']!=[0,499] or selected['n_covariance']!=500
        or selected['overlap_policy']!='allow_explicit_target_overlap'
        or selected['covariance_is_of_mean'] or selected['covariance_multiplier']!=1.
        or selected['volume_Gpc_h3']!=1.):raise ValueError('approved sample/volume/covariance contract drift')
    admission=[]
    for row in rows:
        index=row['realization']
        if row['status']!='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS':raise ValueError('not every input passed')
        validate_moments(row)
        path=AUDIT/'rows'/f'fiducial_real{index:03d}.json'
        if json.loads(path.read_text())!=row:raise ValueError('aggregate report differs from row artifact')
        accepted_path=ADMISSION/path.name;accepted=json.loads(accepted_path.read_text())
        if accepted['realization']!=index or accepted['status']!='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS':
            raise ValueError('measurement admission row is not PASS')
        science_equal(row,accepted)
        if index<100:
            # 旧target报告无顶层n_halo，reader metadata保存其计数；仅补形状标签，不改数值。
            old=dict(prior[index]);old.setdefault('n_halo',old['raw_fof']['n_halo'])
            science_equal(row,old,derived_moment_roundoff=True)
        if index in (127,293,471):
            if accepted.get('diagnostics_consumed_path')!=row['diagnostics_consumed_path']:
                raise ValueError('accepted replay metadata route changed')
            if sha(accepted['diagnostics_consumed_path'])!=row['diagnostics_sha256']:
                raise ValueError('actual replay diagnostics hash differs from admitted evidence')
        if index==425 and (row['DONE_sha256'] is not None or row['completion_marker_consumed'] is not False):
            raise ValueError('unused marker bytes must not be fabricated')
        admission.append(dict(realization=index,row_path=str(accepted_path),row_sha256=sha(accepted_path),
            source_STO_sha256=row['source_sha256'],raw_FoF_sha256=row['raw_fof']['source_raw_sha256'],
            diagnostics_sha256=row['diagnostics_sha256']))
    replay=json.loads((ROOT/'input_recipe_replay_fid471/AUDIT.json').read_text())
    if (rows[471]['source_sha256']!=replay['original_payload_sha256']
        or replay['original_payload_sha256']!=replay['regenerated_payload_sha256']
        or sha(rows[471]['diagnostics_consumed_path'])!=replay['fresh_diagnostics_sha256']):
        raise ValueError('real471 original/replayed payload routing proof drift')
    value=dict(status='ALL500_FIDUCIAL_SOURCES_AND_MEASUREMENT_ADMISSION_INDEPENDENTLY_VERIFIED',
        created_UTC=datetime.now(timezone.utc).isoformat(),passed_count=500,failed_count=0,
        original_source_auditor_PID_confirmed_absent=status['pid'],admission_rows=admission,
        source_report_sha256=sha(AUDIT/'AUDIT.json'),source_contract_sha256=sha(AUDIT/'CONTRACT.json'),
        entry_sha256=sha(Path(__file__)),selection_sha256=sha(ROOT/'ENSEMBLE_SELECTION.json'),
        reused_target_ids=list(range(100)),strict_independent_mock_recovery=False,
        target_derived_moment_relative_roundoff_gate=8*math.ulp(1.),all_source_SHA_comparisons_exact=True,
        original_weights_not_replaced=True,metadata_replay_realizations=[127,293,471],
        real246_same_source_FoF_repair_preserved=True,real425_completion_marker_not_consumed=True,
        n_covariance_required=500,covariance_generated=False,MCMC_released=False,
        volume_Gpc_h3=1.,covariance_is_of_mean=False,covariance_multiplier=1.)
    path=ROOT/'FID500_ADMISSION_COMPLETION_AUDIT.json'
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='admission_rows'}),flush=True)


if __name__=='__main__':main()
