"""模块大纲：real127旧diagnostics读取等待；用现有DM/原producer独立重现同一STO。

原producer的源副本写第一部分outputs；本部分保留审计记录，不覆盖旧输入或下载DM。
FoF的原质量选择提供同一halo行；必须与已有STO逐数组且payload SHA完全一致。
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
import numpy as np
from post_rsd_blob_catalog import read_blob, catalog_from_blobs, adapter
from post_rsd_catalog_identity import validate_periodic_identity
from prepare_sto_post_validation_sources import check_code, RECIPE, PRODUCER, DM, REQUIRED
from inventory_sto_post_inputs import SOURCE
from rsd_b000_contract import PART

ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'
OUT = ROOT / 'input_recipe_replay_fid127_v2'
PRODUCED_ROOT = PART.parent / '01_reconstruction_methods/outputs/sto_post_input_recipe_replays_20261009'


def sha(blob): return hashlib.sha256(blob).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--detach', action='store_true')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('single logical CPU required')
    check_code()
    if args.detach:
        launch = ROOT / 'input_recipe_replay_fid127_v2.launch.json'; log = ROOT / 'input_recipe_replay_fid127_v2.log'
        if launch.exists() or log.exists() or OUT.exists(): raise RuntimeError('prior replay requires inspection')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        record = {'host': socket.gethostname(), 'pid': proc.pid, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
            'command': command, 'log': str(log), 'new_session': True,
            'created_UTC': datetime.now(timezone.utc).isoformat()}
        with launch.open('x') as stream: json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True); return
    OUT.mkdir(exist_ok=False); start = time.time()
    state = {'state': 'RUNNING', 'stage': 'verify_original_source_bytes', 'host': socket.gethostname(),
        'pid': os.getpid(), 'CPU_affinity': sorted(os.sched_getaffinity(0)), 'child_pid': None,
        'started_UTC': datetime.now(timezone.utc).isoformat()}
    save(OUT / 'STATUS.json', state)
    try:
        old_path = SOURCE / 'fiducial/real127/env_lam1_gsr2_thr15/post_sto_halo.npz'
        old_blob = read_blob(old_path); original_sha = sha(old_blob)
        raw_path = adapter.original.raw_path('fiducial', 127); raw_blob = read_blob(raw_path)
        _, real, raw_meta = catalog_from_blobs('fiducial', 127, raw_blob, raw_source=raw_path)
        with np.load(io.BytesIO(old_blob), allow_pickle=False) as z:
            old = {k: np.array(z[k]) for k in z.files}
        identity = validate_periodic_identity(old['positions_mpc_h'], real)
        # 真实FoF质量选择产生的位置，不伪造质量缓存；nenv/number不需要质量导出。
        staged = OUT / 'original_fof_selected_positions/fiducial/real127'
        staged.mkdir(parents=True); np.save(staged / 'positions_mpc_h.npy', real)
        sources = [Path(__file__), PRODUCER, PART / 'code/post_rsd_blob_catalog.py',
                   PART / 'code/post_rsd_catalog.py', PART / 'code/run_rsd_bonly_workflow.py',
                   PART / 'code/post_rsd_catalog_identity.py', PART / 'code/prepare_sto_post_validation_sources.py']
        frozen = {str(p): sha(read_blob(p)) for p in sources}
        command = [sys.executable, '-B', '-u', str(PRODUCER), '--cosmology', 'fiducial', '--realization', '127',
            '--halo-root', str(staged.parents[1]), '--dm-root', str(DM), '--method', 'nenv',
            '--base-weight', 'number', '--env-source', 'dm', '--weight-source', 'dm',
            '--env-mesh', '128', '--recon-mesh', '128', '--gsr', '2', '--thr', '15', '--lambda-th', '1',
            '--dtype', 'f4', '--result-root', str(PRODUCED_ROOT), '--label', 'env_lam1_gsr2_thr15']
        contract = {'cosmology': 'fiducial', 'realization': 127, 'recipe': RECIPE, 'command': command,
            'original_STO_path': str(old_path), 'original_STO_payload_sha256': original_sha,
            'raw_FoF': raw_meta, 'original_STO_position_identity': identity,
            'selected_positions_sha256': sha(read_blob(staged / 'positions_mpc_h.npy')),
            'immutable_sources': frozen, 'numeric_threads': 1,
            'old_diagnostics_read_here': False, 'original_inputs_modified': False,
            'new_DM_downloaded_or_Nbody_generated': False, 'main_sample_choice_changed': False,
            'previous_failed_replay_preserved': str(ROOT / 'input_recipe_replay_fid127'),
            'recovery_reason': 'producer output guard requires source products under part1 outputs',
            'producer_or_output_guard_modified': False}
        save(OUT / 'CONTRACT.json', contract)
        with (OUT / 'PRODUCER.log').open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            state.update(stage='same_recipe_source_replay', child_pid=child.pid, updated_UTC=datetime.now(timezone.utc).isoformat())
            save(OUT / 'STATUS.json', state); code = child.wait()
        state['child_pid'] = None
        if code: raise RuntimeError('original producer failed; retain its log and partial outputs')
        check_code()
        for path, expected in frozen.items():
            if sha(read_blob(path)) != expected: raise ValueError('source drift during replay: ' + path)
        generated = PRODUCED_ROOT / 'fiducial/real127/env_lam1_gsr2_thr15'
        if not all((generated / name).is_file() for name in REQUIRED): raise ValueError('incomplete producer outputs')
        fresh_blob = read_blob(generated / 'post_sto_halo.npz'); fresh_sha = sha(fresh_blob)
        diagnostics = json.loads(read_blob(generated / 'diagnostics.json'))
        if any(diagnostics['config'].get(k) != v for k, v in RECIPE.items()): raise ValueError('recipe changed')
        if diagnostics['component_config']['method'] != 'nenv' or diagnostics['component_config']['base_weight'] != 'number':
            raise ValueError('tracer mark definition changed')
        with np.load(io.BytesIO(fresh_blob), allow_pickle=False) as z:
            if set(z.files) != set(old): raise ValueError('STO payload keys differ')
            equality = {key: bool(np.array_equal(old[key], z[key])) for key in z.files}
        if not all(equality.values()) or fresh_sha != original_sha:
            save(OUT / 'PAYLOAD_MISMATCH.json', {'arrays': equality, 'old_sha256': original_sha, 'new_sha256': fresh_sha})
            raise ValueError('recipe replay differs from existing STO; do not substitute it')
        if sha(read_blob(old_path)) != original_sha: raise ValueError('original STO payload changed during replay')
        report = {'status': 'EXISTING_REAL127_STO_PAYLOAD_REPRODUCED_FROM_SAME_RECIPE_AND_EXISTING_DM',
            'created_UTC': datetime.now(timezone.utc).isoformat(), 'all_arrays_exactly_equal': equality,
            'original_payload_sha256': original_sha, 'regenerated_payload_sha256': fresh_sha,
            'fresh_diagnostics_path': str(generated / 'diagnostics.json'),
            'fresh_diagnostics_sha256': sha(read_blob(generated / 'diagnostics.json')),
            'contract_sha256': sha(read_blob(OUT / 'CONTRACT.json')),
            'original_diagnostics_read_or_repaired': False, 'original_inputs_modified': False,
            'new_DM_downloaded_or_Nbody_generated': False, 'main_covariance_or_MCMC_created': False,
            'does_not_claim_all500_inputs_verified': True, 'elapsed_seconds': time.time()-start}
        save(OUT / 'AUDIT.json', report)
        state.update(state='COMPLETE', stage='payload_and_recipe_reproduced', updated_UTC=datetime.now(timezone.utc).isoformat())
        save(OUT / 'STATUS.json', state); print(json.dumps(report, indent=2), flush=True)
    except Exception:
        state.update(state='FAILED', error=traceback.format_exc(), updated_UTC=datetime.now(timezone.utc).isoformat())
        save(OUT / 'STATUS.json', state); raise


if __name__ == '__main__': main()
