"""模块大纲：四worker因进度int64退出后的同402输入/HDF续跑；原代码不改。

只修复进度JSON整数类型；原并行executor、模型、priors、seed/moves/门/预算
全部复用。失败状态和旧日志保持；本轮日志另存。原700测量不中断。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback
import unittest
import emcee
import run_sto_post_latest_cov_pb_v1 as base
from ensemble_progress_json_safe import advance_json_safe

ROOT=base.ROOT;OUT=base.OUT;REC=OUT/'progress_io_recovery_v1'


def worker(index,ensemble,steps):
    proof=base.read(REC/'RUNTIME_CONTRACT.json')
    for path,want in proof['runtime_sources'].items():
        if base.digest(path)!=want:raise ValueError('progress-only recovery source drift')
    previous=base.advance_ensemble;base.advance_ensemble=advance_json_safe
    try:base.worker(index,ensemble,steps)
    finally:base.advance_ensemble=previous


def pipeline():
    if REC.exists():raise ValueError('preserve existing progress recovery; inspect exact owner before retry')
    old=base.read(OUT/'STATUS.json')
    if (old['state']!='FAILED_OR_INTERRUPTED_PRESERVED_REQUIRES_EXACT_INSPECTION'
            or 'independent ensemble failed' not in old['error']
            or Path('/proc/'+str(old['pid'])).exists()):raise ValueError('old original402 controller must be terminal')
    for pid in old['worker_pids']:
        if Path('/proc/'+str(pid)).exists():raise ValueError('old writer must actually terminate before HDF inspection')
    base.guard()
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_ensemble_progress_json_safe','test_independent_ensemble_backend',
        'test_hdf_lock_retry','test_provisional_prior_safe_initialization']))
    if not tests.wasSuccessful() or tests.testsRun!=7:raise ValueError('seven actual IO/kernel/JSON tests required')
    checkpoints=[]
    case=base.case_directory(0)
    for j in range(4):
        log=case/f'worker_{j}_to6000.log'
        if 'Object of type int64 is not JSON serializable' not in log.read_text():
            raise ValueError('this recovery only addresses actual first-progress JSON failure')
        h=case/f'chain_ensemble_{j}.h5';backend=emcee.backends.HDFBackend(str(h),name=f'ensemble_{j}')
        if backend.iteration!=1:raise ValueError('verified original failure must retain exactly its first saved step')
        checkpoints.append(dict(ensemble=j,iteration=int(backend.iteration),
            first_saved_HDF_sha256=base.digest(h),failed_log_sha256=base.digest(log)))
    REC.mkdir();base.save_json(REC/'FAILED_ORIGINAL_STATUS.json',old)
    names=['resume_sto_latest_pb_progress_v1.py','ensemble_progress_json_safe.py','test_ensemble_progress_json_safe.py']
    frozen={str(base.PART/'code'/name):base.digest(base.PART/'code'/name) for name in names}
    base.save_json(REC/'RUNTIME_CONTRACT.json',dict(status='SAME402_HDF_RESUME_PROGRESS_JSON_INT_CONVERSION_ONLY',
        created_UTC=base.stamp(),runtime_sources=frozen,
        original_parent_sha256=base.digest(OUT/'PB_CONTRACT.json'),original_parent_pid=old['pid'],
        original_writers_confirmed_terminal=True,original_checkpoints=checkpoints,n_covariance=402,
        actual_native_regression_tests_passed=7,
        data_covariance_p_priors_moves_RNG_diagnostics_max_steps_changed=False,
        existing_chain_first_step_retained=True,original_frozen_sources_changed=False,
        original_failed_worker_logs_unchanged=True,new_worker_logs_directory='worker_logs'))
    state=dict(old);state.pop('error',None);state.update(state='LATEST_PB_PARALLEL_ENSEMBLES',
        pid=os.getpid(),resumed_UTC=base.stamp(),worker_pids=[],progress_JSON_recovery=True)
    def update(**values):
        state.update(values,updated_UTC=base.stamp());base.save_json(OUT/'STATUS.json',state)
    update()
    # 只重定向本轮worker入口和日志路径；原parallel_case/采样实现不复制。
    original_popen=base.subprocess.Popen;original_open=Path.open
    def popen(command,*args,**kwargs):
        if isinstance(command,list) and '--worker' in command:
            command=list(command)
            position=command.index(str(Path(base.__file__).resolve()))
            command[position]=str(Path(__file__).resolve())
        return original_popen(command,*args,**kwargs)
    def opened(path,mode='r',*args,**kwargs):
        if (mode=='x' and path.parent.parent==base.BOUT
                and path.name.startswith('worker_') and path.suffix=='.log'):
            replacement=REC/'worker_logs'/path.parent.name/path.name
            replacement.parent.mkdir(parents=True,exist_ok=True)
            return original_open(replacement,mode,*args,**kwargs)
        return original_open(path,mode,*args,**kwargs)
    base.subprocess.Popen=popen;Path.open=opened
    try:
        reports=[]
        for index in range(3):
            reports.append(base.parallel_case(index,update))
            state['completed_pb_cases'].append(base.COSMOS[index]);update()
        pc=base.read(base.POUT/'CONTRACT.json')
        base.save_json(OUT/'RESULTS.json',dict(status='THREE_LATEST_COV_PROVISIONAL_PB_TRIALS_FINISHED_REVIEW_GATES',
            reports=reports,p0_reports=[base.read(base.POUT/'cases'/name/'SUMMARY.json') for name in base.pbase.NAMES[1:]],
            physical_p_calibration=base.read(base.POUT/'cases'/base.pbase.CAL_NAME/'SUMMARY.json'),
            n_covariance=pc['n_covariance'],covariance_ids=pc['covariance_ids'],C_single=True,volume_Gpc_h3=1.,
            diagnostic_only=True,all_pb_cases_converged=all(s['converged'] for s in reports),
            full_STO_noise_closure_certified=False,progress_IO_recovery_contract_sha256=base.digest(REC/'RUNTIME_CONTRACT.json')))
        update(state='LATEST_COV_ALL3_PB_TRIALS_FINISHED_REVIEW_CONVERGENCE',finished_UTC=base.stamp())
    except BaseException:
        update(state='FAILED_OR_INTERRUPTED_PRESERVED_REQUIRES_EXACT_INSPECTION',error=traceback.format_exc());raise
    finally:base.subprocess.Popen=original_popen;Path.open=original_open


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--ensemble',type=int);ap.add_argument('--steps',type=int)
    args=ap.parse_args()
    if socket.gethostname()!='login26':raise ValueError('same original login26 CPU lease required')
    if args.worker is not None:worker(args.worker,args.ensemble,args.steps);return
    if os.sched_getaffinity(0)!={12}:raise ValueError('controller on original CPU12')
    if args.detach:
        log=OUT/'progress_io_recovery_v1.log';launch=OUT/'progress_io_recovery_v1.launch.json'
        if REC.exists() or log.exists() or launch.exists():raise ValueError('inspect prior actual progress recovery first')
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        base.save_json(launch,dict(pid=process.pid,host=socket.gethostname(),CPU_affinity=[12],created_UTC=base.stamp(),log=str(log)))
        print(json.dumps(base.read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
