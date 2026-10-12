"""模块大纲：8核短批次的唯一资源/Slurm/原source验收辅助；不改数值。

暂停原v3 dispatcher而非结束它，原C500/p/P0衔接继续等同一真实PID。
计算节点只接管独占暂停期间的新盒，全部完成后原parent恢复并验hash复用。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess
from run_sto_post_compute_queue import PART, ROOT, keys, directory, verified_finished, source_row, worker_environment
from run_sto_post_target_queue import sha, save

OUT = ROOT/'compute8_measurement_series'
QUEUE = ROOT/'login_measurement_queue_v3'


def memory_capacity(memory_mib):
    # 单盒观测峰值5.02GiB，保守6GiB；parent仅metadata/I/O，另留2GiB。
    count = min(7, (int(memory_mib)-2048)//6144)
    if count < 1:
        raise ValueError('insufficient memory for unchanged numerical worker')
    return count


def enough_time(seconds_remaining):
    # 原131盒最长627秒；启动新盒必须还余16分钟，否则自然排空当前盒。
    return seconds_remaining >= 960


def read(path):
    return json.loads(Path(path).read_text())


def command(args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=30).stdout


def job_fields(jid):
    raw = command(['scontrol', 'show', 'job', '-o', str(jid)])
    return raw, dict(re.findall(r'\b([A-Za-z][A-Za-z0-9/]*)=([^\s]+)', raw))


def accounting(jid):
    raw = command(['sacct', '-X', '-n', '-P', '-j', str(jid), '-o', 'JobIDRaw,State,ExitCode'])
    matches = [line.split('|') for line in raw.splitlines() if line.startswith(str(jid)+'|')]
    if len(matches) != 1:
        return None, raw  # 观察缺失不是终止，不因此重复提交。
    cells = matches[0]
    return dict(job_id=str(jid), state=cells[1].split()[0].rstrip('+'), exit_code=cells[2]), raw


def original_identity(record, *, stopped=False):
    if record['host'] != socket.gethostname():
        raise ValueError('original process must be checked on original host')
    path = Path('/proc')/str(record['pid'])
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    if fields[19] != record['starttime_ticks']:
        return False
    if record['entry'] not in (path/'cmdline').read_bytes().replace(bytes([0]), b' ').decode():
        raise ValueError('original PID/starttime command identity differs')
    if stopped and fields[0] != 'T':
        raise ValueError('original dispatcher is not actually stopped')
    return True


def remote_pause_guard(reference):
    """每个batch从真实login26 /proc检查暂停owner，不只信一个状态文件。"""
    code = ("import json;from pathlib import Path;"
        f"p=Path('/proc/{reference['pid']}');s=(p/'stat').read_text().rsplit(')',1)[1].split();"
        "c=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();"
        "print(json.dumps(dict(starttime=s[19],state=s[0],command=c)))")
    observed = command(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'login26',
        'taskset', '-c', '12', '/usr/bin/python3', '-c', shlex.quote(code)])
    # ssh提示可能进入stdout；只取最后一个JSON，身份仍逐项严格核对。
    records = [line for line in observed.splitlines() if line.startswith('{')]
    if len(records) != 1:
        raise ValueError('actual original paused-process observation missing')
    value = json.loads(records[0])
    if (value['starttime'] != reference['starttime_ticks'] or value['state'] != 'T'
            or reference['entry'] not in value['command']):
        raise ValueError('paused delegation is no longer exclusive')
    return value


def target_rows():
    audit = read(ROOT/'VALIDATION_SOURCE_AUDIT.json')
    if audit['status'] != 'ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE':
        raise ValueError('actual target300 input admission required')
    return {(r['cosmology'], r['realization']): r for r in audit['source_rows']}


def row_for(key, targets):
    return (None, targets[key]) if key[1] < 100 else source_row(key[1])


def all_finished(targets, pilot_sha):
    complete = []
    for key in keys():
        _, row = row_for(key, targets)
        if verified_finished(key, row, pilot_sha):
            complete.append(key)
        elif directory(key).exists():
            raise ValueError('partial canonical output must be inspected: '+str(key))
    return complete
