"""模块大纲：用户改选三噪声后，按实际PID身份停止旧12参数/自动128k；保留链。

只操作已核验的本轮旧链进程，不操作任何700测量、Slurm或p/P0生产进程。
失败/状态/源/HDF均不覆盖；人工改方向另存收据，不把SIGINT写成科学失败。
"""
import os
import signal
import socket
import time
from pathlib import Path
import json
from datetime import datetime, timezone

ROOT = Path('/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model/outputs/sto_marisa_post_rsd_mean100_v1')
OLD = ROOT / 'diagnostic_latest_cov_pb_20261010_v1'
OUT = ROOT / 'user_three_noise_redirect_20261010_v1'


def read(path):
    return json.loads(Path(path).read_text())


def actual(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return dict(pid=pid, terminal=True, state='ABSENT')
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], starttime_ticks=fields[19],
                    terminal=fields[0] in ('Z', 'X'), cmdline=(p / 'cmdline').read_bytes().decode().replace('\0', ' '))
    except FileNotFoundError:
        return dict(pid=pid, terminal=True, state='ABSENT')


def owner(pid, ticks, entry):
    a = actual(pid)
    if not a['terminal'] and (a['starttime_ticks'] != ticks or entry not in a['cmdline']):
        raise ValueError('exact superseded process identity required; no unrelated signals')
    return a


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {12}:
        raise ValueError('original login26 CPU12 control only')
    if OUT.exists():
        raise ValueError('inspect previous direction-change receipt before retry')
    state = read(OLD / 'STATUS.json')
    standby = read(OLD / 'audited_continuation_v1/STATUS.json')
    if state['pid'] != 1209663 or standby['pid'] != 1414034 or standby.get('child_pid'):
        raise ValueError('inspect changed actual ownership or already dispatched child before signaling')
    a = owner(1414034, '152265006', 'continue_sto_latest_pb_audited_v1.py')
    b = owner(1209663, '151936506', 'resume_sto_latest_pb_progress_v1.py')
    OUT.mkdir()
    save('USER_REQUEST.json', dict(direct_human_goal='按旧realspace只保留3个噪声相关参数，重新拟合并与pre-recon比较',
        created_UTC=datetime.now(timezone.utc).isoformat(), old12_products_preserved=True,
        stop_old12_and_automatic128k=True, original700_measurements_continue=True))
    save('OLD_STATUS_BEFORE.json', state)
    save('OLD_CONTINUATION_BEFORE.json', standby)
    save('ACTUAL_BEFORE.json', [a, b])
    if not a['terminal']:
        os.kill(a['pid'], signal.SIGTERM)
    # 先终止旧加长入口，防止旧chain退出触发自动128k。
    limit = time.monotonic() + 10
    while not actual(a['pid'])['terminal'] and time.monotonic() < limit:
        time.sleep(.2)
    if not actual(a['pid'])['terminal']:
        raise ValueError('automatic old12 continuation still live; no further signal')
    if not b['terminal']:
        os.kill(b['pid'], signal.SIGINT)
    limit = time.monotonic() + 35
    while not actual(b['pid'])['terminal'] and time.monotonic() < limit:
        time.sleep(.2)
    if not actual(b['pid'])['terminal']:
        raise ValueError('old12 owner still live; do not restart or touch HDF')
    pids = set(state.get('worker_pids', []))
    pids.update(read(OLD / 'STATUS.json').get('worker_pids', []))
    for progress in (OLD / 'pb_cases').glob('*/PROGRESS_ensemble_*.json'):
        pids.add(read(progress)['pid'])
    for pid in sorted(pids):
        worker = actual(pid)
        if not worker['terminal']:
            if 'resume_sto_latest_pb_progress_v1.py' not in worker['cmdline'] or '--worker' not in worker['cmdline']:
                raise ValueError('unexpected writer identity; preserve and inspect')
            os.kill(pid, signal.SIGINT)
    limit = time.monotonic() + 20
    while any(not actual(pid)['terminal'] for pid in pids) and time.monotonic() < limit:
        time.sleep(.2)
    final = [actual(pid) for pid in [1414034, 1209663] + sorted(pids)]
    if not all(x['terminal'] for x in final):
        raise ValueError('actual old writers remain live; no replacement launch')
    result = dict(status='USER_SUPERSEDED_EFT12_AND_AUTO128K_ACTUALLY_TERMINATED_PRODUCTS_PRESERVED',
                  created_UTC=datetime.now(timezone.utc).isoformat(), actual_processes_after=final,
                  original700_measurement_coordinator=actual(1278399),
                  HDF_read_or_modified=False, files_deleted=False)
    save('STOP_RECEIPT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
