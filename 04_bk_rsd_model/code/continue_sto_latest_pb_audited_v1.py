"""模块大纲：单次后台衔接；真实原owner终止后，先审计，再必要预算续跑。

这不是重启活动链；等待只读取JSON和/proc，每20秒复核一次，登录CPU12。
若真实失败、原门不足或审计报错，保留全部产品并明确状态，不自动改门。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import extend_sto_latest_pb_budget_v1 as extension
import run_sto_post_latest_cov_pb_v1 as base
from post_rsd_owner_quiescence import owner_quiescent

OUT = base.OUT / 'audited_continuation_v1'


def pipeline():
    if OUT.exists():
        raise ValueError('preserve existing single continuation; inspect actual handle before recovery')
    base.guard()
    OUT.mkdir()
    sources = ['continue_sto_latest_pb_audited_v1.py', 'extend_sto_latest_pb_budget_v1.py',
               'post_rsd_budget_only.py', 'test_post_rsd_budget_only.py',
               'audit_sto_latest_pb_extension_v1.py', 'audit_sto_latest_pb_completion_v2.py',
               'audit_sto_latest_pb_inference_v1.py', 'post_rsd_owner_quiescence.py',
               'test_post_rsd_owner_quiescence.py']
    frozen = {str(base.PART / 'code' / name): base.digest(base.PART / 'code' / name) for name in sources}
    base.save_json(OUT / 'CONTRACT.json', dict(created_UTC=base.stamp(), immutable_sources=frozen,
        original_parent_sha256=base.digest(base.OUT / 'PB_CONTRACT.json'),
        status='ONE_SHOT_WAIT_FOR_ACTUAL_OWNER_THEN_AUDIT_AND_BUDGET_ONLY_RESUME',
        C_single=True, n_covariance=402, CPU_affinity=[12], diagnostic_only=True,
        model_data_covariance_p_priors_moves_RNG_gates_changed=False,
        original700_measurements_continue=True, full_STO_noise_closure_certified=False))
    state = dict(state='WAITING_ACTUAL_ORIGINAL_OWNER_OR_WRITER', pid=os.getpid(),
                 host=socket.gethostname(), CPU_affinity=[12], started_UTC=base.stamp())
    state['starttime_ticks'] = (Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    def update(**values):
        state.update(values, updated_UTC=base.stamp())
        base.save_json(OUT / 'STATUS.json', state)
    def check_sources():
        base.guard()
        for path, expected in frozen.items():
            if base.digest(path) != expected:
                raise ValueError('single continuation frozen source drift: ' + path)
    def execute(name):
        check_sources()
        process = subprocess.Popen([sys.executable, '-B', '-u', str(base.PART / 'code' / name)],
                                   stdin=subprocess.DEVNULL)
        update(child_pid=process.pid, child_entry=name)
        result = process.wait()
        if result:
            raise ValueError('actual continuation subprocess failed: {} return {}'.format(name, result))
        update(child_pid=None)
    try:
        while True:
            current = base.read(base.OUT / 'STATUS.json')
            original_terminal = owner_quiescent(current)
            writers_terminal = all(owner_quiescent(base.read(path))
                for path in base.BOUT.glob('*/PROGRESS_ensemble_*.json'))
            update(original_owner_pid=current['pid'], original_owner_terminal=original_terminal,
                   original_writers_terminal=writers_terminal, original_case=current.get('case'),
                   original_state=current['state'])
            if original_terminal and writers_terminal:
                break
            time.sleep(20)
        extension.quiescent_original()
        update(state='AUDITING_ACTUAL_ORIGINAL64K_FULL_HDF')
        execute('audit_sto_latest_pb_completion_v2.py')
        result = base.read(base.OUT / 'MCMC_COMPLETION_AUDIT.json')
        if result['all3_converged']:
            update(state='PRELIMINARY_THREE_PB_SAMPLING_AND_INPUTS_VERIFIED_NOT_FORMAL_NOISE',
                   final_audit=str(base.OUT / 'MCMC_COMPLETION_AUDIT.json'), finished_UTC=base.stamp())
            return
        update(state='BUDGET_ONLY_COPY_RESUME_ORIGINAL64K_PRESERVED')
        execute('extend_sto_latest_pb_budget_v1.py')
        update(state='AUDITING_ACTUAL_EXTENSION_FULL_HDF_AND_ORIGINAL_PREFIX')
        execute('audit_sto_latest_pb_extension_v1.py')
        result = base.read(extension.OUT / 'MCMC_COMPLETION_AUDIT.json')
        update(state=('PRELIMINARY_THREE_PB_SAMPLING_AND_INPUTS_VERIFIED_NOT_FORMAL_NOISE'
                      if result['all3_converged'] else 'ACTUAL_TRIALS_FINISHED_ORIGINAL_GATES_FAILED_REVIEW_REQUIRED'),
               all3_converged=result['all3_converged'], final_audit=str(extension.OUT / 'MCMC_COMPLETION_AUDIT.json'),
               finished_UTC=base.stamp())
    except BaseException:
        update(state='FAILED_OR_INTERRUPTED_PRESERVED_REQUIRES_ACTUAL_INSPECTION', error=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {12}:
        raise ValueError('single continuation on original login26 CPU12 only')
    if args.detach:
        log = base.OUT / 'audited_continuation_v1.log'
        launch = base.OUT / 'audited_continuation_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():
            raise ValueError('inspect prior actual one-shot continuation; no duplicate')
        with log.open('x') as stream:
            process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        base.save_json(launch, dict(pid=process.pid, host=socket.gethostname(), CPU_affinity=[12],
                                  created_UTC=base.stamp(), log=str(log)))
        print(json.dumps(base.read(launch)), flush=True)
    else:
        pipeline()


if __name__ == '__main__':
    main()
