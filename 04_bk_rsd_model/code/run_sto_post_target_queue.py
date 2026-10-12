"""模块大纲：三真值mean100输入批量测量；四单核worker；复用完成盒。

使用原重构/P/B入口和新的同字节IO适配；不改变任何旧结果、协方差编号或
科学门。独立会话和文件日志允许对话结束后继续，失败盒保留而不偷偷重跑。
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
ROOT = PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT = ROOT/'target300_measurement_queue'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def output(key):
    cosmo, index = key
    return ROOT/'pilots'/f'{cosmo}_real{index:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpus', default='7,8,9,10'); ap.add_argument('--detach', action='store_true')
    args = ap.parse_args(); cpus = tuple(int(c) for c in args.cpus.split(','))
    if len(cpus) != 4 or len(set(cpus)) != 4 or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('four distinct permitted CPUs required')
    if args.detach:
        if OUT.exists() or (ROOT/'target300_measurement_queue.launch.json').exists():
            raise ValueError('existing launch must be inspected, not replaced')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--cpus', args.cpus]
        log = ROOT/'target300_measurement_queue.launch.log'
        with log.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(host=socket.gethostname(), pid=proc.pid, command=command, CPU_pool=cpus,
                      log=str(log), created_UTC=datetime.now(timezone.utc).isoformat(), new_session=True)
        with (ROOT/'target300_measurement_queue.launch.json').open('x') as f:
            json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(record), flush=True); return
    os.sched_setaffinity(0, {cpus[0]})
    OUT.mkdir(exist_ok=False)
    source_audit = ROOT/'VALIDATION_SOURCE_AUDIT.json'
    audit = json.loads(source_audit.read_text())
    if audit['status'] != 'ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE':
        raise ValueError('target sources not validated')
    rows = {(r['cosmology'],r['realization']):r for r in audit['source_rows']}
    keys = [(c,i) for i in range(100) for c in ('fiducial','LC_p','LC_m')]
    if set(rows) != set(keys): raise ValueError('not exactly 300 source rows')
    sources = [Path(__file__), PART/'code/run_sto_post_target_box.py',
        PART/'code/run_sto_post_rsd_pilot.py', PART/'code/post_rsd_blob_catalog.py',
        PART/'code/post_rsd_estimator.py', source_audit,
        PART.parent/'01_reconstruction_methods/common/halo_b000_pipeline.py',
        PART.parent/'01_reconstruction_methods/common/nersc_env.sh']
    frozen = {str(p):sha(p) for p in sources}
    contract = dict(role='complete mean100 target inputs, not covariance or MCMC release',
        n_target_per_truth=100, truths=[0,100,-100], recon_mesh=256, measurement_mesh=256,
        nran=30, matched_self=True, nested_randoms=True, zero_atom_counterfactual_saved=True,
        b_rec=2.2280546353635335, p_rec=1.2822, fNL_rec=100., CPU_pool=cpus,
        numeric_threads=1, covariance_ids_changed=False, covariance_is_of_mean=False,
        volume_Gpc_h3=1., immutable_sources=frozen)
    save(OUT/'CONTRACT.json', contract)
    state = dict(state='PREFLIGHT', host=socket.gethostname(), pid=os.getpid(), CPU_pool=cpus,
        maximum_workers=4, started_UTC=datetime.now(timezone.utc).isoformat(), completed=[], workers=[])
    save(OUT/'STATUS.json', state)
    active = {}; pending = list(keys); error = None

    def finished(key):
        path = output(key)
        if not (path/'DONE.json').exists(): return False
        done = json.loads((path/'DONE.json').read_text()); original = json.loads((path/'CONTRACT.json').read_text())
        row = rows[key]
        if (done['measurement_sha256'] != sha(path/'MEASUREMENTS.npz')
                or done['post_halos_sha256'] != sha(path/'POST_HALOS.npz')
                or done['contract_sha256'] != sha(path/'CONTRACT.json')
                or done['entry_sha256'] != frozen[str(PART/'code/run_sto_post_rsd_pilot.py')]
                or original['source_STO_sha256'] != row['source_sha256']
                or original['source_diagnostics_sha256'] != row['diagnostics_sha256']
                or original['raw_fof']['source_raw_sha256'] != row['raw_fof']['source_raw_sha256']):
            raise ValueError('completed target provenance drift: '+str(key))
        return True

    def update():
        state.update(completed_count=len(state['completed']), total_count=300, pending_count=len(pending),
            workers=[{k:v for k,v in r.items() if k not in ('process','stream')} for r in active.values()],
            updated_UTC=datetime.now(timezone.utc).isoformat())
        save(OUT/'STATUS.json', state)

    try:
        for key in list(pending):
            if finished(key): state['completed'].append(key); pending.remove(key)
            elif output(key).exists(): raise ValueError('incomplete old output requires inspection: '+str(key))
        state['state'] = 'RUNNING'; update()
        while pending or active:
            for path, expected in frozen.items():
                if sha(path) != expected: raise ValueError('immutable measurement source changed: '+path)
            for cpu in cpus:
                if cpu in active or not pending or error: continue
                key = pending.pop(0); cosmo,index = key
                log = OUT/f'{cosmo}_real{index:03d}.log'; stream = log.open('x')
                # 同一环境设置，仅将原shell入口最后的Python命令换成IO适配入口。
                command = ['bash','-c',
                    'source ../01_reconstruction_methods/common/nersc_env.sh\n'
                    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
                    'export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1\n'
                    'export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"\n'
                    'exec python -B -u code/run_sto_post_target_box.py "$@"',
                    'target-box','--cosmology',cosmo,'--realization',str(index)]
                proc = subprocess.Popen(command, cwd=PART, stdout=stream, stderr=subprocess.STDOUT,
                    preexec_fn=lambda c=cpu: os.sched_setaffinity(0,{c}))
                active[cpu] = dict(cpu=cpu, pid=proc.pid, key=key, log=str(log), process=proc, stream=stream)
            update(); time.sleep(2.)
            for cpu, worker in list(active.items()):
                code = worker['process'].poll()
                if code is None: continue
                worker['stream'].close(); del active[cpu]
                if code or not finished(worker['key']):
                    error = f"target {worker['key']} failed ({code}); preserve log, no automatic restart"
                else: state['completed'].append(worker['key'])
            update()
            if error and not active: raise RuntimeError(error)
        if len(state['completed']) != 300: raise ValueError('incomplete target set')
        state['state'] = 'COMPLETE'; update()
    except Exception:
        state.update(state='FAILED_WORKERS_MAY_STILL_RUN' if active else 'FAILED', error=traceback.format_exc())
        update(); raise


if __name__ == '__main__': main()
