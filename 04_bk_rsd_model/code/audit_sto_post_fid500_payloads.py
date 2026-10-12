"""模块大纲：完整读取候选Fid0..499的原字节；同配方STO与原FoF/RSD行身份验收。

只读输入，绕开源端stat/fstat；原数值reader处理相同字节的本地临时副本。
不更换主样本、不下载/生成新STO或DM、不生成协方差、不运行拟合。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import unittest
import zipfile
import numpy as np
from post_rsd_blob_catalog import read_blob, optional_blob, catalog_from_blobs, adapter
from post_rsd_catalog_identity import validate_periodic_identity
from prepare_sto_post_validation_sources import RECIPE
from inventory_sto_post_inputs import SOURCE
from rsd_b000_contract import PART

ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'
OUT = ROOT / 'fid0_499_payload_audit'


def sha(value): return hashlib.sha256(value).hexdigest()


def validate_rsd_before_painting(rsd):
    """原reader仅对LOS取模；其他轴允许精确0=L，由原production wrapper统一wrap。

    这里只验收输入，不改坐标、不放宽任何ULP容差；L之外的值仍拒绝。
    """
    values = np.asarray(rsd)
    if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1000.)):
        raise ValueError('invalid RSD catalog before periodic painting')
    endpoints = np.argwhere(values == 1000.)
    return {'input_domain': 'closed [0,L]; exact L=0 periodic identification, no tolerance',
            'exact_L_endpoint_components': len(endpoints), 'endpoint_indices_first8': endpoints[:8].tolist(),
            'coordinates_modified_by_audit': False,
            'production_periodic_wrapping_still_required': True}


def save(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--detach', action='store_true')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('single-core input audit only')
    if args.detach:
        log = ROOT / 'fid0_499_payload_audit.log'; launch = ROOT / 'fid0_499_payload_audit.launch.json'
        if log.exists() or launch.exists() or OUT.exists(): raise RuntimeError('prior execution must be inspected, not restarted')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        record = {'host': socket.gethostname(), 'pid': process.pid, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
                  'new_session': True, 'log': str(log), 'command': command,
                  'created_UTC': datetime.now(timezone.utc).isoformat()}
        with launch.open('x') as stream: json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True); return
    OUT.mkdir(exist_ok=False); (OUT / 'rows').mkdir()
    state = {'state': 'PREFLIGHT', 'host': socket.gethostname(), 'pid': os.getpid(),
             'CPU_affinity': sorted(os.sched_getaffinity(0)), 'completed': [], 'failed': [],
             'total': 500, 'started_UTC': datetime.now(timezone.utc).isoformat(),
             'sample_choice_approved_here': False, 'primary_MCMC_released': False}
    save(OUT / 'STATUS.json', state); start = time.time()
    try:
        tests = unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_blob_catalog')
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        if not result.wasSuccessful() or result.testsRun != 3: raise RuntimeError('blob/reader regressions failed')
        old = json.loads(read_blob(ROOT / 'VALIDATION_SOURCE_AUDIT.json'))
        prior = {r['realization']: r for r in old['source_rows'] if r['cosmology'] == 'fiducial'}
        sources = [Path(__file__), PART / 'code/post_rsd_blob_catalog.py', PART / 'code/test_post_rsd_blob_catalog.py',
                   PART / 'code/post_rsd_catalog.py', PART / 'code/post_rsd_catalog_identity.py',
                   PART / 'code/run_rsd_bonly_workflow.py', PART / 'code/export_pre_rsd.py',
                   PART / 'code/vendor/readfof.py', PART / 'code/inventory_sto_post_inputs.py',
                   PART / 'code/prepare_sto_post_validation_sources.py']
        frozen = {str(p): sha(read_blob(p)) for p in sources}
        contract = {'cosmology': 'fiducial', 'candidate_realizations': list(range(500)), 'recipe': RECIPE,
            'source_root': str(SOURCE), 'mass_threshold_Msun_h': 1e13, 'z': .5, 'LOS': 'z',
            'boxsize_Mpc_h': 1000., 'original_reader_reused_on_identical_bytes': True,
            'source_stat_or_fstat_used_by_blob_reader': False, 'immutable_sources': frozen,
            'sample_choice_approved_here': False, 'main_covariance_or_MCMC_created': False}
        save(OUT / 'CONTRACT.json', contract); csha = sha(read_blob(OUT / 'CONTRACT.json'))
        rows = []
        for index in range(500):
            state.update(state='RUNNING', current_realization=index, stage='read_source_payloads',
                         elapsed_seconds=time.time()-start, updated_UTC=datetime.now(timezone.utc).isoformat())
            save(OUT / 'STATUS.json', state)
            if index % 20 == 0:
                for path, expected in frozen.items():
                    if sha(read_blob(path)) != expected: raise RuntimeError('numeric reader/source changed during audit')
            row = {'realization': index, 'status': 'UNVERIFIED'}
            try:
                directory = SOURCE / 'fiducial' / f'real{index:03d}' / 'env_lam1_gsr2_thr15'
                raw_source = adapter.original.raw_path('fiducial', index)
                raw_blob = read_blob(raw_source)
                sto_blob = read_blob(directory / 'post_sto_halo.npz', max_bytes=64*1024*1024)
                diag_blob = read_blob(directory / 'diagnostics.json', max_bytes=8*1024*1024)
                marker = read_blob(directory / 'DONE', max_bytes=1024*1024)
                diag = json.loads(diag_blob)
                if any(diag['config'].get(k) != v for k, v in RECIPE.items()): raise ValueError('STO recipe mismatch')
                if diag['component_config']['method'] != 'nenv' or diag['component_config']['base_weight'] != 'number':
                    raise ValueError('wrong tracer mark definition')
                cache = adapter.original.OLD_POS / 'fiducial' / f'real{index:03d}'
                cached_pos = optional_blob(cache / 'positions_mpc_h.npy', max_bytes=128*1024*1024)
                cached_mass = optional_blob(cache / 'masses_msun_h.npy', max_bytes=128*1024*1024)
                state['stage'] = 'original_reader_and_identity'; save(OUT / 'STATUS.json', state)
                rsd, positions, meta = catalog_from_blobs('fiducial', index, raw_blob, raw_source=raw_source,
                    position_cache_blob=cached_pos, mass_cache_blob=cached_mass)
                # 控制受信任输入解压的内存，所需数组最多float64三坐标与单权重。
                with zipfile.ZipFile(io.BytesIO(sto_blob)) as archive:
                    if archive.getinfo('positions_mpc_h.npy').file_size > 24*len(positions)+65536:
                        raise ValueError('STO position array exceeds the matching raw halo size')
                    if archive.getinfo('weights.npy').file_size > 8*len(positions)+65536:
                        raise ValueError('STO weight array exceeds the matching raw halo size')
                with np.load(io.BytesIO(sto_blob), allow_pickle=False) as data:
                    identity = validate_periodic_identity(data['positions_mpc_h'], positions)
                    weights = np.asarray(data['weights'], dtype=float)
                if weights.shape != (len(positions),) or not np.isfinite(weights).all() or weights.sum() <= 0:
                    raise ValueError('invalid original STO weights')
                rsd_domain = validate_rsd_before_painting(rsd)
                if index in prior:
                    previous = prior[index]
                    if (sha(sto_blob) != previous['source_sha256'] or sha(diag_blob) != previous['diagnostics_sha256']
                            or sha(raw_blob) != previous['raw_fof']['source_raw_sha256']):
                        raise ValueError('previously audited source bytes changed')
                    np.testing.assert_allclose(meta['rms_ds_Mpc_h'], previous['raw_fof']['rms_ds_Mpc_h'], rtol=2e-14)
                row.update(status='PAYLOAD_RECIPE_AND_FOF_ROW_IDENTITY_PASS', source=str(directory),
                    source_sha256=sha(sto_blob), diagnostics_sha256=sha(diag_blob), DONE_sha256=sha(marker),
                    raw_fof=meta, source_position_identity=identity, RSD_input_domain=rsd_domain, n_halo=len(positions),
                    N2_data_self=float(1e9*np.sum(weights*weights)/weights.sum()**2),
                    N3_data_self=float(1e18*np.sum(weights**3)/weights.sum()**3),
                    negative_weights=int(np.count_nonzero(weights < 0)), prior_identity_audit_reproduced=index in prior,
                    source_payload_bytes=len(sto_blob), raw_payload_bytes=len(raw_blob))
                state['completed'].append(index)
            except Exception:
                row.update(status='INPUT_REQUIRES_INSPECTION', error=traceback.format_exc()); state['failed'].append(index)
            rows.append(row); save(OUT / 'rows' / f'fiducial_real{index:03d}.json', row)
            state.update(stage='row_recorded', elapsed_seconds=time.time()-start,
                         updated_UTC=datetime.now(timezone.utc).isoformat()); save(OUT / 'STATUS.json', state)
            if index % 20 == 19: print(json.dumps({'visited': index+1, 'passed': len(state['completed']), 'failed': state['failed']}), flush=True)
        passed = len(state['completed']) == 500 and not state['failed']
        report = {'status': 'ALL500_EXISTING_REALSPACE_SOURCES_PAYLOAD_IDENTITY_PASS' if passed else 'FID500_INPUTS_REQUIRE_INSPECTION',
            'created_UTC': datetime.now(timezone.utc).isoformat(), 'rows': rows, 'passed_count': len(state['completed']),
            'failed_realizations': state['failed'], 'contract_sha256': csha, 'source_root': str(SOURCE),
            'candidate_covariance_ids': [0, 499], 'exact_fiducial_target_overlap_if_selected': [0, 99],
            'possible_PNG_target_phase_overlap_if_selected': [0, 99],
            'possible_LCp_calibration_phase_overlap_if_selected': [100, 249],
            'sample_choice_approved_here': False, 'old_inputs_modified': False,
            'new_DM_or_STO_generated_or_downloaded': False, 'main_covariance_or_MCMC_created': False,
            'CPU_affinity': sorted(os.sched_getaffinity(0)), 'elapsed_seconds': time.time()-start}
        save(OUT / 'AUDIT.json', report)
        state.update(state='COMPLETE' if passed else 'COMPLETE_WITH_INPUT_ERRORS', stage='complete',
                     updated_UTC=datetime.now(timezone.utc).isoformat()); save(OUT / 'STATUS.json', state)
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    except Exception:
        state.update(state='FAILED', error=traceback.format_exc(), updated_UTC=datetime.now(timezone.utc).isoformat())
        save(OUT / 'STATUS.json', state); raise


if __name__ == '__main__': main()
