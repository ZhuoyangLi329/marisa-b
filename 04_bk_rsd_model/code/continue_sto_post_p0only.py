"""模块大纲：绑定实际v3校准parent，退出且收敛后自动跑三组主post-P0。

复用CPU7，旧校准parent/child仍活着时只等待；不重启测量、校准或失败链。
冻结独立源码，输出实际PID/阶段；B/P+B及IR门仍保持未放行。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from rsd_b000_contract import PART, digest, save_json
from post_rsd_p0only_inputs import calibration_continuation_gate

ROOT = PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT = ROOT/'p0only_continuation_v1'
PCAL = ROOT/'physical_p_continuation_login_v3'


def read(path):
    return json.loads(Path(path).read_text())


def identity(pid, entry):
    path = Path('/proc')/str(pid)
    start = (path/'stat').read_text().rsplit(')', 1)[1].split()[19]
    record = dict(pid=pid, starttime_ticks=start, entry=str(entry), host=socket.gethostname())
    if not specific_process_is_live(record):
        raise ValueError('required exact original p continuation is not currently live')
    return record


def specific_process_is_live(record):
    if record['host'] != socket.gethostname():
        raise ValueError('inspect original PID on original host')
    path = Path('/proc')/str(record['pid'])
    try:
        start = (path/'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return False
    if start != record['starttime_ticks']:
        return False
    command = (path/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
    if record['entry'] not in command:
        raise ValueError('exact PID/starttime command differs; inspect, do not assume exit')
    return True


def validate_calibration_exit():
    """freeze本身也要求真实退出证据，不只由自动controller保证。"""
    frozen = read(OUT/'CONTRACT.json'); reference = frozen['pcal_control_identity']
    record = read(PCAL/'STATUS.json')
    if record['pid'] != reference['pid'] or record['host'] != reference['host']:
        raise ValueError('p continuation identity drift')
    if not calibration_continuation_gate(record, specific_process_is_live(reference)):
        raise ValueError('p calibration continuation/child still owns CPU and mutable summary')
    summary = ROOT/'physical_p_calibration_mcmc/cases/physical_p_GSR2_RSD_fNLfixed100/SUMMARY.json'
    if digest(summary) != record['summary_sha256']:
        raise ValueError('completed calibration summary drift')
    return frozen


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--detach', action='store_true')
    ap.add_argument('--max-steps', type=int, default=64000); args = ap.parse_args()
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {7}:
        raise ValueError('original login26 CPU7 required; numeric libraries single-thread')
    if args.max_steps != 64000:
        raise ValueError('preserve original maximum64000 sampling budget')
    if args.detach:
        log = ROOT/'p0only_continuation_v1.log'; launch = ROOT/'p0only_continuation_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():
            raise ValueError('existing P0 continuation must be inspected, not replaced')
        pcal = read(PCAL/'STATUS.json')
        identity(pcal['pid'], PART/'code/continue_sto_post_login_pipeline_v3.py')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(host=socket.gethostname(), pid=proc.pid, CPU_affinity=[7], command=command,
            launched_UTC=datetime.now(timezone.utc).isoformat(), log=str(log), detached=True)
        with launch.open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
        print(json.dumps(value), flush=True); return
    if OUT.exists():
        raise ValueError('preserve existing continuation and inspect its exact process')
    pcal = read(PCAL/'STATUS.json')
    reference = identity(pcal['pid'], PART/'code/continue_sto_post_login_pipeline_v3.py')
    sources = [Path(__file__), PART/'code/run_sto_post_p0only.py', PART/'code/post_rsd_p0only_inputs.py',
        PART/'code/run_sto_post_noir_mcmc.py', PART/'code/run_rsd_p0only.py',
        PART/'code/run_rsd_b000_mcmc.py', PART/'code/post_rsd_noir_case_inputs.py',
        PART/'code/post_rsd_noir_polynomial.py', PART/'code/post_rsd_noir_likelihood.py',
        PART/'code/post_rsd_ensemble.py', PART/'code/rsd_pb_common.py', PART/'code/rsd_b000_contract.py',
        PART/'code/run_sto_physical_p_mcmc.py', PCAL/'CONTRACT.json',
        ROOT/'physical_p_continuation_login_v3.launch.json', ROOT/'ENSEMBLE_SELECTION.json',
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    frozen = {str(path): digest(path) for path in sources}
    OUT.mkdir()
    owner = (OUT/'OWNER.lock').open('x'); fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save_json(OUT/'CONTRACT.json', dict(pcal_control_identity=reference, immutable_sources=frozen,
        CPU_affinity=[7], numeric_threads=1, main_P0_case_count=3, remaining_main_B_PB_case_count=36,
        original_main_case_count=39, full_goal_reduced=False, IR_still_required=True,
        B_noise_model_released=False))
    state = dict(state='WAITING_FOR_SPECIFIC_P_CALIBRATION_CONTINUATION', pid=os.getpid(),
        host=socket.gethostname(), starttime_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')', 1)[1].split()[19],
        pcal_continuation_pid=reference['pid'], child_pid=None, completed_P0_cases=[],
        started_UTC=datetime.now(timezone.utc).isoformat(), B_noise_model_released=False, IR_completed=False)
    def update(**values):
        state.update(values, updated_UTC=datetime.now(timezone.utc).isoformat()); save_json(OUT/'STATUS.json', state)
    def guard():
        for path, want in frozen.items():
            if digest(path) != want:
                raise ValueError('new P0 continuation frozen source drift: '+path)
    update()
    try:
        while True:
            guard(); record = read(PCAL/'STATUS.json')
            if record['pid'] != reference['pid'] or record['host'] != reference['host']:
                raise ValueError('exact p continuation record differs')
            live = specific_process_is_live(reference)
            update(pcal_continuation_live=live, actual_pcal_state=record['state'],
                   completed_measurements=record.get('completed_measurements'))
            if calibration_continuation_gate(record, live):
                validate_calibration_exit(); break
            time.sleep(60)
        entry = PART/'code/run_sto_post_p0only.py'
        commands = [('FREEZING_THREE_MAIN_POST_P0', ['freeze'])]
        commands += [(f'SAMPLING_MAIN_POST_P0_CASE_{i}', ['sample', '--case-index', str(i)]) for i in range(3)]
        for stage, arguments in commands:
            guard(); update(state=stage, child_pid=None)
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(entry)]+arguments, cwd=PART)
            update(child_pid=proc.pid)
            if proc.wait():
                raise ValueError('post-P0 stage failed; preserve outputs/no automatic retry: '+stage)
            update(child_pid=None)
            if arguments[0] == 'sample':
                names = read(ROOT/'mcmc_noir_p0only/CASES.json'); name = names[int(arguments[-1])]
                path = ROOT/'mcmc_noir_p0only/cases'/name/'SUMMARY.json'; summary = read(path)
                state['completed_P0_cases'].append(dict(name=name, converged=summary['converged'],
                    summary_sha256=digest(path)))
                update()
        passed = all(r['converged'] for r in state['completed_P0_cases'])
        update(state='THREE_MAIN_POST_P0_CONVERGED_B_PB_AND_IR_STILL_REQUIRED' if passed
            else 'THREE_MAIN_POST_P0_SAMPLED_SOME_NOT_CONVERGED_REQUIRES_ATTENTION',
            finished_UTC=datetime.now(timezone.utc).isoformat(), main_case_count_required=39,
            full_goal_completed=False)
    except Exception:
        update(state='FAILED_PRESERVED_REQUIRES_INSPECTION', error=traceback.format_exc()); raise
    finally:
        owner.close()


if __name__ == '__main__':
    main()
