"""模块大纲：冻结旧dispatcher；当前盒自然完成；核验后释放登录节点。

只向经host/cmdline/starttime核实的两个本任务parent发信号，不终止worker。
旧状态、源码及结果不改；独立迁移证据记录真实进程终止和逐盒DONE哈希。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

PART = Path(__file__).resolve().parents[1]
ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'
OUT = ROOT / 'compute32_migration'
QUEUES = ('target300_measurement_queue', 'fid100_499_measurement_queue')
ENTRY = ('run_sto_post_target_queue.py', 'run_sto_post_covariance_queue.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def parse_stat(value):
    fields = value.rsplit(')', 1)[1].split()
    return dict(state=fields[0], ppid=int(fields[1]), starttime=int(fields[19]))


def process(pid):
    """starttime防PID复用；不扫描args/打开源数据，也不以状态文件推断进程存活。"""
    root = Path('/proc') / str(pid)
    try:
        fields = parse_stat((root / 'stat').read_text())
        cmd = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        return dict(pid=pid, command=cmd, **fields)
    except FileNotFoundError:
        return None


def children(parents):
    result = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) not in parents:
                continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        info = process(int(item.name))
        if info is not None:
            result.append(info)
    return result


def freeze():
    if OUT.exists():
        raise ValueError('existing migration must be inspected, not replaced')
    records = []
    # 两个身份均通过后才开始发信号，且完全保留旧队列快照。
    for queue, entry in zip(QUEUES, ENTRY):
        state_path = ROOT / queue / 'STATUS.json'
        state = json.loads(state_path.read_text())
        info = process(state['pid'])
        if state['host'] != socket.gethostname() or info is None:
            raise ValueError('original live queue must be verified on its original host')
        if str(PART / 'code' / entry) not in info['command']:
            raise ValueError('parent PID does not belong to expected task entry')
        records.append(dict(queue=queue, original_status=state, process=info,
                            original_contract_sha256=sha(ROOT / queue / 'CONTRACT.json')))
    OUT.mkdir()
    report = dict(state='PREPARED', host=socket.gethostname(), parents=records,
                  created_UTC=datetime.now(timezone.utc).isoformat(),
                  user_authorization='CPU compute nodes shared32; login MCMC stays <=8 cores',
                  entry_sha256=sha(Path(__file__)), workers_not_signalled=True)
    save(OUT / 'MIGRATION.json', report)
    for record in records:
        info = process(record['process']['pid'])
        if info is None or info['starttime'] != record['process']['starttime']:
            raise ValueError('parent identity changed before freeze')
        os.kill(info['pid'], signal.SIGSTOP)
    for _ in range(50):
        if all(process(r['process']['pid'])['state'] == 'T' for r in records):
            break
        time.sleep(.02)
    else:
        raise ValueError('dispatcher stop not confirmed')
    report.update(state='DISPATCHERS_STOPPED_WORKERS_DRAINING',
                  workers=children({r['process']['pid'] for r in records}),
                  stopped_UTC=datetime.now(timezone.utc).isoformat())
    save(OUT / 'MIGRATION.json', report)
    print(json.dumps(report), flush=True)


def finish():
    report = json.loads((OUT / 'MIGRATION.json').read_text())
    if report['host'] != socket.gethostname():
        raise ValueError('drain check must run on original login host')
    if report['entry_sha256'] != sha(Path(__file__)):
        raise ValueError('migration implementation changed')
    if report['state'] == 'LOGIN_MEASUREMENT_PARENTS_TERMINATED_WORKERS_COMPLETE':
        for r in report['parents']:
            if process(r['process']['pid']) is not None:
                raise ValueError('recorded parent PID exists again; inspect actual process identity')
        print(json.dumps(report), flush=True)
        return
    active = []
    for expected in report['workers']:
        now = process(expected['pid'])
        if now is not None and now['starttime'] != expected['starttime']:
            raise ValueError('worker PID reused; inspect without signalling it')
        if now is not None and now['state'] != 'Z':
            active.append(now)
    if active:
        print(json.dumps(dict(state='WAITING_FOR_ORIGINAL_WORKERS', workers=active)), flush=True)
        return
    completions = []
    for expected in report['workers']:
        args = expected['command'].split()
        index = int(args[args.index('--realization') + 1])
        cosmo = args[args.index('--cosmology') + 1] if '--cosmology' in args else 'fiducial'
        directory = ROOT / 'pilots' / f'{cosmo}_real{index:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'
        done = json.loads((directory / 'DONE.json').read_text())
        for field, name in (('contract_sha256', 'CONTRACT.json'),
                            ('measurement_sha256', 'MEASUREMENTS.npz'),
                            ('post_halos_sha256', 'POST_HALOS.npz')):
            if done[field] != sha(directory / name):
                raise ValueError('drained worker output provenance mismatch')
        completions.append(dict(key=[cosmo, index], directory=str(directory),
                                DONE_sha256=sha(directory / 'DONE.json')))
    for record in report['parents']:
        expected = record['process']
        now = process(expected['pid'])
        if now is None:
            continue
        if now['starttime'] != expected['starttime'] or now['state'] != 'T':
            raise ValueError('original dispatcher is not stopped with matching identity')
        # SIGTERM在stop状态挂起，SIGCONT后优先递送默认终止信号，不恢复dispatch。
        os.kill(now['pid'], signal.SIGTERM)
        os.kill(now['pid'], signal.SIGCONT)
    for _ in range(100):
        if all(process(r['process']['pid']) is None for r in report['parents']):
            break
        time.sleep(.1)
    else:
        raise ValueError('parent termination not yet confirmed; do not launch replacement')
    report.update(state='LOGIN_MEASUREMENT_PARENTS_TERMINATED_WORKERS_COMPLETE',
                  worker_completion_evidence=completions,
                  original_parent_processes_confirmed_absent=True,
                  finished_UTC=datetime.now(timezone.utc).isoformat())
    save(OUT / 'MIGRATION.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('freeze', 'finish'))
    args = ap.parse_args()
    freeze() if args.command == 'freeze' else finish()
