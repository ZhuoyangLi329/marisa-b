"""模块大纲：原host检查同一批worker；自然完成后释放已验收的CPU32作业。

不重启、kill worker或绕过scratch许可证；只释放本任务具名held job。
等待进程每30秒核验具体/proc身份，终端断开不影响断点迁移。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
import migrate_sto_post_login_queue as migration

ROOT, OUT = migration.ROOT, migration.OUT


def check_job(value, job_id):
    fields = dict(re.findall(r'\b([A-Za-z][A-Za-z0-9/]*)=([^\s]+)', value))
    expected = dict(JobId=str(job_id), JobName='sm_post_rsd_cpu32', Account='desi', QOS='shared')
    if (any(fields.get(k) != v for k, v in expected.items())
        or fields.get('NumCPUs') not in ('32', '33') or fields.get('CPUs/Task') not in ('32', '33')
        or not fields.get('ReqTRES', '').startswith('cpu=32,mem=60G,')
        or fields.get('Command') != str(migration.PART / 'code/submit_sto_post_compute32.slurm')
        or fields.get('JobState') != 'PENDING' or fields.get('Reason') != 'JobHeldUser'):
        raise ValueError('job is not this task matching held CPU32 allocation')
    return fields


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job-id', required=True, type=int)
    ap.add_argument('--detach', action='store_true')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('single-core lightweight drain watcher required')
    report = json.loads((OUT / 'MIGRATION.json').read_text())
    if report['host'] != socket.gethostname():
        raise ValueError('original host required')
    if args.detach:
        launch = OUT / 'WATCHER.launch.json'; log = OUT / 'WATCHER.log'
        if launch.exists() or log.exists():
            raise ValueError('existing watcher must be inspected, never duplicated')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--job-id', str(args.job_id)]
        with log.open('x') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=proc.pid, host=socket.gethostname(), command=command, job_id=args.job_id,
                      CPU_affinity=sorted(os.sched_getaffinity(0)), entry_sha256=migration.sha(Path(__file__)),
                      created_UTC=datetime.now(timezone.utc).isoformat(), log=str(log))
        with launch.open('x') as f:
            json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(record), flush=True); return
    state = dict(state='WAITING_FOR_ORIGINAL_WORKERS', pid=os.getpid(), host=socket.gethostname(),
                 job_id=args.job_id, started_UTC=datetime.now(timezone.utc).isoformat())
    try:
        while True:
            migration.finish()
            report = json.loads((OUT / 'MIGRATION.json').read_text())
            state['updated_UTC'] = datetime.now(timezone.utc).isoformat()
            if report['state'] == 'LOGIN_MEASUREMENT_PARENTS_TERMINATED_WORKERS_COMPLETE':
                break
            migration.save(OUT / 'WATCHER_STATUS.json', state)
            time.sleep(30)
        job = subprocess.run(['scontrol', 'show', 'job', str(args.job_id)], check=True,
                             stdout=subprocess.PIPE, text=True).stdout
        fields = check_job(job, args.job_id)
        stored = subprocess.run(['scontrol', 'write', 'batch_script', str(args.job_id), '-'], check=True,
                                stdout=subprocess.PIPE, text=True).stdout
        current = (migration.PART / 'code/submit_sto_post_compute32.slurm').read_text()
        if stored != current:
            raise ValueError('Slurm stored script differs from current validated submission source')
        subprocess.run(['scontrol', 'release', str(args.job_id)], check=True)
        state.update(state='CPU32_SHARED_JOB_RELEASED_NOT_RUNNING_PROOF',
                     released_UTC=datetime.now(timezone.utc).isoformat(), validated_job_fields=fields,
                     stored_batch_script_sha256=migration.sha(migration.PART / 'code/submit_sto_post_compute32.slurm'),
                     migration_record_sha256=migration.sha(OUT / 'MIGRATION.json'))
        migration.save(OUT / 'WATCHER_STATUS.json', state)
        print(json.dumps(state), flush=True)
    except Exception:
        state.update(state='FAILED_PRESERVED_NO_RESTART', error=traceback.format_exc())
        migration.save(OUT / 'WATCHER_STATUS.json', state)
        raise


if __name__ == '__main__':
    main()
