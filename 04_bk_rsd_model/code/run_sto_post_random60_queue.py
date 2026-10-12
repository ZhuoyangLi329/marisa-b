"""模块大纲：同20盒的nested random30/60配对数值控制；复用原pilot入口。

先验收完整nran30 baseline，复用已有real000 nran60，只补剩余19盒。
固定STO权重和重构参数，要求post halo坐标不变；不选择主covariance样本。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


PART = Path(__file__).resolve().parents[1]
ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpus', default='10,11')
    ap.add_argument('--detach', action='store_true')
    args = ap.parse_args(); cpus = tuple(int(c) for c in args.cpus.split(','))
    if len(cpus) != 2 or len(set(cpus)) != 2 or not set(cpus) <= os.sched_getaffinity(0):
        raise RuntimeError('exactly two allowed logical CPUs required')
    audit_path = ROOT / 'pilot_fid20_diagnostics/AUDIT.json'
    baseline = json.loads(audit_path.read_text())
    if (baseline['status'] != 'FID20_PILOT_MEASUREMENTS_VERIFIED_NOT_MAIN_COVARIANCE'
            or baseline['realizations'] != list(range(20))
            or baseline['matrix_sha256'] != sha(audit_path.parent / 'PILOT_MEASUREMENTS.npz')):
        raise RuntimeError('complete same-box baseline must be audited first')
    baseline_rows = {r['realization']: r for r in baseline['rows']}
    if args.detach:
        work = ROOT / 'pilot_fid20_random60_queue'
        launch = ROOT / 'pilot_fid20_random60_queue.launch.json'
        if work.exists() or launch.exists(): raise RuntimeError('existing launch requires inspection')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())] + [
            value for value in sys.argv[1:] if value != '--detach']
        log = ROOT / 'pilot_fid20_random60_queue.launch.log'
        with log.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        record = {'host': socket.gethostname(), 'pid': proc.pid, 'new_session': True,
                  'log': str(log), 'CPU_pool': cpus, 'command': command,
                  'launched_UTC': datetime.now(timezone.utc).isoformat()}
        with launch.open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True); return
    os.sched_setaffinity(0, {cpus[0]})
    work = ROOT / 'pilot_fid20_random60_queue'
    work.mkdir(exist_ok=False)
    state = {'state': 'PREFLIGHT', 'host': socket.gethostname(), 'pid': os.getpid(),
             'CPU_pool': cpus, 'parent_affinity': sorted(os.sched_getaffinity(0)),
             'maximum_workers': 2, 'realizations': list(range(20)),
             'started_UTC': datetime.now(timezone.utc).isoformat(), 'completed': [],
             'workers': [], 'primary_covariance_or_MCMC_released': False,
             'role': 'same-box finite-random paired controls only, not primary covariance or inference'}
    save(work / 'STATUS.json', state); print(json.dumps(state), flush=True)
    active = {}; pending = list(range(20)); started = time.time()
    paths = [Path(__file__), audit_path, audit_path.parent / 'PILOT_MEASUREMENTS.npz',
             PART / 'code/run_sto_post_rsd_pilot.py',
             PART / 'code/run_sto_post_rsd_pilot.sh', PART / 'code/post_rsd_estimator.py',
             ROOT / 'VALIDATION_SOURCE_AUDIT.json', ROOT / 'SOURCE_AUDIT.json',
             ROOT / 'MATCHED_SELF_AUDIT.json', ROOT / 'ZERO_ATOM_AUDIT.json',
             ROOT / 'OPERATOR_AUDIT.json', ROOT / 'RECON_ALPHA_COMPARISON.npz',
             PART.parent / '01_reconstruction_methods/common/halo_b000_pipeline.py',
             PART.parent / '01_reconstruction_methods/common/nersc_env.sh']
    frozen = {str(path): sha(path) for path in paths}
    pilot_sha = frozen[str(PART / 'code/run_sto_post_rsd_pilot.py')]
    source_audit = json.loads((ROOT / 'VALIDATION_SOURCE_AUDIT.json').read_text())
    if source_audit['status'] != 'ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE':
        raise RuntimeError('validated target source identities required')
    sources = {row['realization']: row for row in source_audit['source_rows']
               if row['cosmology'] == 'fiducial' and row['realization'] < 20}
    if set(sources) != set(range(20)):
        raise RuntimeError('not all twenty fiducial sources have been identity audited')
    contract = {'role': state['role'], 'cosmology': 'fiducial', 'realizations': list(range(20)),
                'immutable_sources': frozen, 'recon_mesh': 256, 'measurement_mesh': 256,
                'nran': 60, 'self_contraction': 'matched', 'zero_atom_diagnostics_only': True,
                'nested_randoms': True, 'source_identity_audit': str(ROOT / 'VALIDATION_SOURCE_AUDIT.json'),
                'physical_p_fitted': False, 'main_covariance_ids_changed': False,
                'main_covariance_count': 500, 'main_covariance_is_of_mean': False,
                'main_volume_Gpc_h3': 1., 'worker_CPUs': cpus, 'numeric_threads': 1}
    save(work / 'CONTRACT.json', contract)

    def output(index):
        return ROOT / 'pilots' / f'fiducial_real{index:03d}_rec256_meas256_ran60_selfmatched_zeroaudit_nested'

    def finished(index):
        path = output(index) / 'DONE.json'
        if not path.exists(): return False
        done = json.loads(path.read_text())
        if (done['status'] != 'PILOT_COMPLETE_NOT_SCIENTIFIC_VALIDATION'
                or done['entry_sha256'] != pilot_sha
                or done['measurement_sha256'] != sha(path.parent / 'MEASUREMENTS.npz')
                or done['contract_sha256'] != sha(path.parent / 'CONTRACT.json')
                or done['post_halos_sha256'] != sha(path.parent / 'POST_HALOS.npz')):

            raise RuntimeError('pilot completion provenance mismatch: ' + str(path))
        expected = baseline_rows[index]
        base = ROOT / 'pilots' / f'fiducial_real{index:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'
        if sha(base / 'DONE.json') != expected['DONE_sha256']:
            raise RuntimeError('paired baseline provenance changed')
        if done['post_halos_sha256'] != expected['post_halos_sha256']:
            raise RuntimeError('random-density control changed post halo coordinates')
        paired = json.loads((path.parent / 'CONTRACT.json').read_text())
        original = json.loads((base / 'CONTRACT.json').read_text())
        for field in ('source_STO_sha256', 'source_diagnostics_sha256', 'random_seed',
                      'fixed_reconstruction', 'shared_measure_source_sha256'):
            if paired[field] != original[field]:
                raise RuntimeError('unpaired control source: ' + field)
        return True

    def update():
        state.update(workers=[{k: v for k, v in row.items() if k not in ('process', 'stream')}
                              for row in active.values()], pending=pending,
                     updated_UTC=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time() - started)
        save(work / 'STATUS.json', state)

    error = None
    try:
        for i in list(pending):
            if finished(i): state['completed'].append(i); pending.remove(i)
            elif output(i).exists():
                raise RuntimeError('incomplete pilot output exists; inspect the original execution, do not restart')
        state['state'] = 'RUNNING'; update()
        while pending or active:
            for path, expected in frozen.items():
                if sha(path) != expected: raise RuntimeError('source changed during queue: ' + path)
            for cpu in cpus:
                if cpu in active or not pending or error: continue
                index = pending[0]; row = sources[index]
                state.update(stage='source_recheck', checking_realization=index); update()
                for path, expected in ((Path(row['source']) / 'post_sto_halo.npz', row['source_sha256']),
                                       (Path(row['source']) / 'diagnostics.json', row['diagnostics_sha256']),
                                       (Path(row['raw_fof']['source_raw']), row['raw_fof']['source_raw_sha256'])):
                    if sha(path) != expected: raise RuntimeError('validated source changed: ' + str(path))
                pending.pop(0); log = work / f'fiducial_real{index:03d}.log'; stream = log.open('x')
                command = ['bash', str(PART / 'code/run_sto_post_rsd_pilot.sh'), '--cosmology', 'fiducial',
                           '--realization', str(index), '--recon-mesh', '256', '--measurement-mesh', '256',
                           '--nran', '60', '--self-contraction', 'matched', '--zero-atom-diagnostics', '--nested-randoms']
                proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                        preexec_fn=lambda c=cpu: os.sched_setaffinity(0, {c}))
                active[cpu] = {'cpu': cpu, 'pid': proc.pid, 'realization': index, 'log': str(log),
                               'process': proc, 'stream': stream}
                state['stage'] = 'pilot_measurement'; update()
            time.sleep(2.)
            for cpu, row in list(active.items()):
                code = row['process'].poll()
                if code is None: continue
                row['stream'].close(); del active[cpu]
                if code or not finished(row['realization']):
                    error = f"pilot {row['realization']} failed, returncode={code}; preserve log and partial output"
                else:
                    state['completed'].append(row['realization']); state['completed'].sort()
                    print(json.dumps({'completed': state['completed'], 'total': 20}), flush=True)
            update()
            if error and not active: raise RuntimeError(error)
        if state['completed'] != list(range(20)): raise RuntimeError('pilot set incomplete')
        state['state'] = 'COMPLETE'; state['stage'] = 'complete'; update(); print(json.dumps(state), flush=True)
    except Exception:
        # 不误报/重启仍活着的worker，也不杀死它们；状态保留PID和部分产物。
        state.update(state='FAILED_WORKERS_MAY_STILL_BE_RUNNING' if active else 'FAILED',
                     error=traceback.format_exc()); update(); raise


if __name__ == '__main__': main()

