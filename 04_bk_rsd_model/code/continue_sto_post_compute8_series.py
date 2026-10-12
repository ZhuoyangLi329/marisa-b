"""模块大纲：只在首个8核短作业实际RUNNING后接管；断点批次自动续提交。

暂停而不结束原v3 parent，等待全部在途盒自然完成；原C500/p/P0等待器不改。
全部700实际完成后恢复v3，它用原worker REUSE/哈希门补齐记账再原路衔接。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from sto_post_compute8_common import (PART, ROOT, OUT, QUEUE, read, sha, save, command, job_fields,
    accounting, original_identity, target_rows, row_for, all_finished, verified_finished)


def verify_job8(jid):
    raw, value = job_fields(jid)
    expected = dict(UserId='lzy(112513)', JobName='sto_measure8_short', Account='desi', QOS='shared',
        NumCPUs='8', **{'CPUs/Task': '8'}, TimeLimit='00:30:00',
        Command=str(PART/'code/submit_sto_post_compute8_batch.slurm'))
    if any(value.get(k) != want for k, want in expected.items()):
        raise ValueError('not the exact owned eight-core short measurement job: '+raw)
    if not value.get('ReqTRES', '').startswith('cpu=8,mem=14G,'):
        raise ValueError('memory must not cause CPU inflation')
    stored = command(['scontrol', 'write', 'batch_script', str(jid), '-'])
    if stored != (PART/'code/submit_sto_post_compute8_batch.slurm').read_text():
        raise ValueError('actual stored Slurm script differs')
    return value


def capture_children(reference):
    children = (Path('/proc')/str(reference['pid'])/'task'/str(reference['pid'])/'children').read_text().split()
    tracked = {w['pid']: w for w in read(QUEUE/'STATUS.json')['workers']}; rows = []
    for number in children:
        pid = int(number); path = Path('/proc')/number
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        args = [a.decode() for a in (path/'cmdline').read_bytes().split(bytes([0])) if a]
        if pid in tracked:
            key = tuple(tracked[pid]['key'])
        else:
            # 处理parent刚派发尚未写STATUS的短窗口；依据实际child argv，不猜样本。
            for _ in range(50):
                if '--realization' in args:
                    break
                time.sleep(.1)
                args = [a.decode() for a in (path/'cmdline').read_bytes().split(bytes([0])) if a]
            if '--realization' not in args:
                raise ValueError('child argv not yet identified; preserve paused owner for inspection')
            index = int(args[args.index('--realization')+1])
            cosmo = args[args.index('--cosmology')+1] if '--cosmology' in args else 'fiducial'
            key = cosmo, index
        rows.append(dict(pid=pid, starttime_ticks=fields[19], key=key,
                         actual_state_at_pause=fields[0], command=args))
    return rows


def children_terminal(rows, targets, pilot_sha):
    evidence = []
    for record in rows:
        path = Path('/proc')/str(record['pid']); terminal = not path.exists(); actual = None
        if not terminal:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            actual = fields[0]
            terminal = fields[19] != record['starttime_ticks'] or fields[0] == 'Z'
        if not terminal:
            return None
        _, row = row_for(tuple(record['key']), targets)
        if not verified_finished(tuple(record['key']), row, pilot_sha):
            raise ValueError('original in-flight worker ended without valid completed box')
        evidence.append(dict(record, actual_terminal_state=actual, complete_box_hash_verified=True,
            zombie_is_terminal_but_original_paused_parent_has_not_reaped_it=actual == 'Z'))
    return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--first-job', required=True, type=int)
    ap.add_argument('--detach', action='store_true'); args = ap.parse_args()
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {12}:
        raise ValueError('lightweight orchestration on original login26 CPU12 required')
    if args.detach:
        launch = ROOT/'compute8_measurement_series.launch.json'; log = ROOT/'compute8_measurement_series.log'
        if OUT.exists() or launch.exists() or log.exists():
            raise ValueError('existing series must be inspected, not duplicated')
        verify_job8(args.first_job)
        cmd = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--first-job', str(args.first_job)]
        with log.open('xb') as stream:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(host=socket.gethostname(), pid=proc.pid, command=cmd, first_job=args.first_job,
                     CPU_affinity=[12], launched_UTC=datetime.now(timezone.utc).isoformat(), log=str(log))
        with launch.open('x') as stream:
            import json
            json.dump(value, stream, indent=2); stream.write('\n')
        print(value, flush=True); return
    if OUT.exists():
        raise ValueError('preserve prior compute8 series')
    old = read(QUEUE/'STATUS.json')
    reference = dict(host=old['host'], pid=old['pid'], starttime_ticks=old['starttime_ticks'],
                     entry=str(PART/'code/run_sto_post_login_queue_v3.py'))
    if not original_identity(reference):
        raise ValueError('original v3 parent must still be actually live')
    verify_job8(args.first_job)
    sources = [Path(__file__), PART/'code/sto_post_compute8_common.py', PART/'code/run_sto_post_compute8_batch.py',
        PART/'code/run_sto_post_compute8_box.py', PART/'code/submit_sto_post_compute8_batch.slurm',
        QUEUE/'CONTRACT.json', ROOT/'compute8_transition_20261010/THEORY_JOB_TRANSITION.json']
    frozen = dict(read(QUEUE/'CONTRACT.json')['immutable_sources'])
    frozen.update({str(p): sha(p) for p in sources})
    OUT.mkdir(); lock = (OUT/'SERIES_OWNER.lock').open('x'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(OUT/'CONTRACT.json', dict(immutable_sources=frozen, original_queue_identity=reference,
        actual_compute_CPU_limit=8, walltime_per_batch='00:30:00', memory='14G', numeric_threads=1,
        pause_not_terminate_original_v3=True, original_C500_p_P0_continuations_preserved=True,
        main_MCMC_required=39, full_goal_reduced=False, IR_still_required=True,
        covariance_is_of_mean=False, covariance_multiplier=1., volume_Gpc_h3=1.))
    state = dict(state='WAITING_FOR_SPECIFIC_FIRST8_JOB_ORIGINAL_LOGIN_CONTINUES', pid=os.getpid(),
        host=socket.gethostname(), starttime_ticks=(Path('/proc/self/stat')).read_text().rsplit(')', 1)[1].split()[19],
        current_job_id=args.first_job, original_v3_paused=False, jobs=[],
        original_queue_identity=reference, started_UTC=datetime.now(timezone.utc).isoformat())
    def update(**values):
        state.update(values, updated_UTC=datetime.now(timezone.utc).isoformat()); save(OUT/'STATUS.json', state)
    def guard():
        for p, want in frozen.items():
            if sha(p) != want:
                raise ValueError('frozen eight-core transition source drift: '+p)
    update(); targets = target_rows(); pilot_sha = sha(PART/'code/run_sto_post_rsd_pilot.py')
    jid = args.first_job
    try:
        while True:
            guard(); observed, raw = accounting(jid)
            if observed is None or observed['state'] in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
                fields = verify_job8(jid)
                update(actual_job_state=fields['JobState'], actual_job_reason=fields.get('Reason'))
                if not state['original_v3_paused']:
                    if not original_identity(reference):
                        # 原login在排队期间已完成：不再启动独立测量，无数据替换。
                        done = read(QUEUE/'STATUS.json')
                        if done['state'] == 'COMPLETE' and done['completed_count'] == 700 and not done['workers']:
                            if fields['JobState'] == 'PENDING':
                                command(['scancel', str(jid)])
                            update(state='ORIGINAL_LOGIN_FINISHED700_BEFORE_COMPUTE_HANDOFF_NO_DUPLICATE',
                                   finished_UTC=datetime.now(timezone.utc).isoformat()); return
                        raise ValueError('original v3 ended without complete700')
                    if fields['JobState'] == 'RUNNING':
                        os.kill(reference['pid'], signal.SIGSTOP)
                        if not original_identity(reference, stopped=True):
                            raise ValueError('actual original dispatcher pause not confirmed')
                        rows = capture_children(reference)
                        save(OUT/'PAUSE_OBSERVATION.json', dict(original_queue_identity=reference,
                            in_flight_children=rows, checked_UTC=datetime.now(timezone.utc).isoformat()))
                        update(state='DRAINING_ORIGINAL_INFLIGHT_BOXES_WITH_PARENT_STOPPED', original_v3_paused=True)
                        while True:
                            guard()
                            if not original_identity(reference, stopped=True):
                                raise ValueError('original paused owner exited during in-flight drain')
                            evidence = children_terminal(rows, targets, pilot_sha)
                            if evidence is not None:
                                break
                            time.sleep(15)
                        save(OUT/'HANDOFF_READY.json', dict(
                            state='ORIGINAL_V3_ACTUALLY_STOPPED_INFLIGHT_BOXES_COMPLETED_DELEGATION_READY',
                            original_queue_identity=reference, original_children_terminal_evidence=evidence,
                            immutable_original_queue_contract_sha256=sha(QUEUE/'CONTRACT.json'),
                            ready_UTC=datetime.now(timezone.utc).isoformat(), original_sources_modified=False))
                        update(state='COMPUTE8_SHORT_BATCH_OWNS_NEW_MEASUREMENTS', original_inflight_completed=True)
                else:
                    if not original_identity(reference, stopped=True):
                        raise ValueError('paused original owner no longer exists')
                time.sleep(30); continue
            state['jobs'].append(dict(observed, accounting_raw=raw)); update()
            if observed['state'] not in ('COMPLETED', 'TIMEOUT') or (observed['state'] == 'COMPLETED' and observed['exit_code'] != '0:0'):
                raise ValueError('actual batch failed; preserve outputs, no silent numerical retry')
            batch = OUT/('job'+str(jid))/'STATUS.json'
            if observed['state'] == 'COMPLETED':
                value = read(batch)
                if value['workers'] or value['state'] not in ('SHORT_BATCH_DRAINED_INCOMPLETE_SAFE_TO_CONTINUE',
                    'ALL700_COMPLETE_PUBLISHED_NOT_C500_OR_MCMC_RELEASE', 'BATCH_DRAINED_BEFORE_DELEGATION_NO_MEASUREMENT'):
                    raise ValueError('successful batch exit without safe drained/complete state')
            completed = all_finished(targets, pilot_sha)
            update(completed_count=len(completed))
            if len(completed) == 700:
                if not original_identity(reference, stopped=True):
                    raise ValueError('only restore exact originally paused parent')
                os.kill(reference['pid'], signal.SIGCONT)
                update(state='ALL700_PUBLISHED_ORIGINAL_V3_RESUMED_FOR_EXISTING_HASH_REUSE', original_v3_paused=False)
                while original_identity(reference):
                    guard(); time.sleep(30)
                final = read(QUEUE/'STATUS.json')
                if final['state'] != 'COMPLETE' or final['completed_count'] != 700 or final['workers']:
                    raise ValueError('original queue did not finish its original full700 gate')
                update(state='COMPUTE8_MEASUREMENTS_COMPLETE_ORIGINAL_C500_P_P0_CONTINUATIONS_PRESERVED',
                    finished_UTC=datetime.now(timezone.utc).isoformat(), main_MCMC_released=False, IR_completed=False); return
            if not state['original_v3_paused']:
                raise ValueError('compute batch ended before delegation; original login preserved, inspect')
            # TIMEOUT仅重新计算未发布盒；所有staging保留，canonical已完成盒逐hash复用。
            update(state='SUBMITTING_NEXT_SHORT8_BATCH', submission_attempt_started_UTC=datetime.now(timezone.utc).isoformat())
            answer = command(['sbatch', '--parsable', str(PART/'code/submit_sto_post_compute8_batch.slurm')])
            jid = int(answer.strip().split(';')[0]); update(current_job_id=jid, state='WAITING_FOR_NEXT_SPECIFIC_SHORT8_BATCH')
            verify_job8(jid)
    except Exception:
        update(state='FAILED_PRESERVED_REQUIRES_INSPECTION', error=traceback.format_exc()); raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
