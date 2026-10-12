"""模块大纲：实际原owner终止后，只克隆未收敛402链并增加预算到128k。

原64k目录全部只读保留；原parallel_case函数、writer、RNG/moves、诊断门
直接复用，不改变模型、目标、C_single、p、先验、初始点或已保存样本。
输出独立目录；所有writer终止才诊断，活动状态只读writer-owned JSON。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback
import unittest
import emcee
import run_sto_post_latest_cov_pb_v1 as base
import resume_sto_latest_pb_progress_v1 as recovery
from post_rsd_owner_quiescence import owner_quiescent
from post_rsd_budget_only import checkpoint_range

OUT = base.OUT / 'budget_extension_128k_v1'
MAXIMUM = 128000
ORIGINAL_CASE_DIRECTORY = base.case_directory


def read(path):
    return base.read(path)


def quiescent_original():
    state = read(base.OUT / 'STATUS.json')
    if not owner_quiescent(state):
        raise ValueError('original actual owner is still live or paused; no HDF access')
    for progress in base.BOUT.glob('*/PROGRESS_ensemble_*.json'):
        if not owner_quiescent(read(progress)):
            raise ValueError('original actual writer still live; no HDF access')
    if (state['state'] != 'LATEST_COV_ALL3_PB_TRIALS_FINISHED_REVIEW_CONVERGENCE'
            or state['completed_pb_cases'] != list(base.COSMOS)):
        raise ValueError('all original three actual trials must finish first')
    base.guard()
    return state


def extension_case_directory(index):
    return OUT / 'pb_cases' / ORIGINAL_CASE_DIRECTORY(index).name


def guard(full_original_hdf=False):
    base.guard()
    contract = read(OUT / 'CONTRACT.json')
    for path, expected in contract['immutable_inputs'].items():
        if base.digest(path) != expected:
            raise ValueError('budget-only immutable source/input/original product drift: ' + path)
    for path, record in contract['original_HDF_files'].items():
        actual = Path(path).stat()
        if actual.st_size != record['size_bytes'] or actual.st_mtime_ns != record['mtime_ns']:
            raise ValueError('original read-only HDF metadata drift: ' + path)
        if full_original_hdf and base.digest(path) != record['sha256']:
            raise ValueError('original read-only HDF content drift: ' + path)
    return contract


def prepare():
    quiescent_original()
    if OUT.exists():
        raise ValueError('preserve prior extension; inspect actual owner before any recovery')
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_post_rsd_budget_only', 'test_ensemble_progress_json_safe',
        'test_independent_ensemble_backend', 'test_hdf_lock_retry',
        'test_provisional_prior_safe_initialization', 'test_post_rsd_owner_quiescence']))
    if not tests.wasSuccessful() or tests.testsRun != 12:
        raise ValueError('twelve real HDF/RNG/guard/budget regression tests required')
    OUT.mkdir()
    immutable = dict(base.guard()['immutable_inputs'])
    source_names = ['extend_sto_latest_pb_budget_v1.py', 'post_rsd_budget_only.py',
                    'test_post_rsd_budget_only.py', 'post_rsd_owner_quiescence.py',
                    'test_post_rsd_owner_quiescence.py']
    for name in source_names:
        path = base.PART / 'code' / name
        immutable[str(path)] = base.digest(path)
    for path, want in read(recovery.REC / 'RUNTIME_CONTRACT.json')['runtime_sources'].items():
        if base.digest(path) != want:
            raise ValueError('original recovery source drift')
        immutable[path] = want
    for path in [base.OUT / 'STATUS.json', base.OUT / 'RESULTS.json',
                 base.OUT / 'PB_CONTRACT.json', base.OUT / 'INDEPENDENT_INPUT_AUDIT.json',
                 recovery.REC / 'RUNTIME_CONTRACT.json']:
        immutable[str(path)] = base.digest(path)
    records = []
    original_hdf = {}
    for index, truth in enumerate(base.COSMOS):
        original = ORIGINAL_CASE_DIRECTORY(index)
        summary = read(original / 'SUMMARY.json')
        steps = int(summary['steps'])
        if (steps > 64000 or steps % 2000 or (not summary['converged'] and steps != 64000)
                or summary['n_covariance'] != 402):
            raise ValueError('original real402/64k completion required')
        record = dict(index=index, cosmology=truth, initial_steps=steps,
                      already_converged=bool(summary['converged']), original_directory=str(original))
        for name in ['contract.json', 'likelihood.npz', 'MAP.json', 'SUMMARY.json', 'POSTERIOR_THIN.npz']:
            path = original / name
            immutable[str(path)] = base.digest(path)
        if not summary['converged']:
            target = extension_case_directory(index)
            target.mkdir(parents=True)
            for name in ['contract.json', 'likelihood.npz', 'MAP.json']:
                shutil.copy2(original / name, target / name)
                immutable[str(target / name)] = base.digest(target / name)
            for j in range(4):
                name = 'chain_ensemble_{}.h5'.format(j)
                path = original / name
                old = emcee.backends.HDFBackend(str(path), name='ensemble_{}'.format(j))
                if int(old.iteration) != steps:
                    raise ValueError('all original ensembles must be equal length before copying')
                actual = path.stat()
                original_hdf[str(path)] = dict(sha256=base.digest(path), size_bytes=actual.st_size,
                                                mtime_ns=actual.st_mtime_ns)
                shutil.copy2(path, target / name)
                if base.digest(target / name) != original_hdf[str(path)]['sha256']:
                    raise ValueError('initial HDF clone is not byte-identical')
            record['extension_directory'] = str(target)
        else:
            record['extension_directory'] = None
        records.append(record)
    base.save_json(OUT / 'CONTRACT.json', dict(
        status='BUDGET_ONLY_COPY_RESUME_402_PB_ORIGINAL64K_READ_ONLY', created_UTC=base.stamp(),
        immutable_inputs=immutable, original_HDF_files=original_hdf,
        records=records, max_steps_each_ensemble=MAXIMUM,
        original_budget=64000, actual_native_regression_tests_passed=12,
        original_parallel_case_function_reused=True, checkpoint_increment=2000,
        model_data_covariance_physical_p_priors_moves_RNG_diagnostics_changed=False,
        initial_HDF_clones_byte_identical=True, already_converged_cases_not_resampled=True,
        n_covariance=402, C_single=True, volume_Gpc_h3=1., target_count_per_truth=100,
        p_fixed_physical=base.guard()['p_fixed_physical'], p_uncertainty_propagated=False,
        ensemble_CPUs=list(base.CPUS), login_task_CPU_union=list(range(7, 15)),
        diagnostic_only=True, provisional=True, IR=False,
        full_STO_noise_closure_certified=False, formal_fNL_or_kmax_released=False))
    return guard()


def worker(index, ensemble, steps):
    guard()
    if steps > MAXIMUM:
        raise ValueError('declared extension cap exceeded')
    base.case_directory = extension_case_directory
    recovery.worker(index, ensemble, steps)


def pipeline():
    contract = prepare()
    state = dict(state='BUDGET_ONLY_PB_RUNNING', pid=os.getpid(), host=socket.gethostname(),
                 CPU_affinity=[12], started_UTC=base.stamp(), completed_pb_cases=[])
    def update(**values):
        state.update(values, updated_UTC=base.stamp())
        base.save_json(OUT / 'STATUS.json', state)
    update()
    original_popen = base.subprocess.Popen
    def popen(command, *args, **kwargs):
        if isinstance(command, list) and '--worker' in command:
            command = list(command)
            command[command.index(str(Path(base.__file__).resolve()))] = str(Path(__file__).resolve())
        return original_popen(command, *args, **kwargs)
    base.case_directory = extension_case_directory
    base.subprocess.Popen = popen
    try:
        reports = []
        for row in contract['records']:
            index = row['index']
            guard()
            if row['already_converged']:
                reports.append(read(Path(row['original_directory']) / 'SUMMARY.json'))
            else:
                base.range = checkpoint_range(row['initial_steps'], MAXIMUM)
                reports.append(base.parallel_case(index, update))
            state['completed_pb_cases'].append(row['cosmology'])
            update()
        base.save_json(OUT / 'RESULTS.json', dict(
            status='THREE_BUDGET_ONLY_PB_TRIALS_FINISHED_REVIEW_ORIGINAL_GATES', reports=reports,
            all_pb_cases_converged=all(x['converged'] for x in reports),
            original64k_products_unchanged=True, n_covariance=402, diagnostic_only=True,
            full_STO_noise_closure_certified=False, formal_fNL_or_kmax_released=False))
        guard(full_original_hdf=True)
        update(state='BUDGET_ONLY_ALL3_PB_TRIALS_FINISHED_REVIEW_CONVERGENCE', finished_UTC=base.stamp())
    except BaseException:
        update(state='FAILED_BUDGET_ONLY_PRESERVED_REQUIRES_ACTUAL_INSPECTION', error=traceback.format_exc())
        raise
    finally:
        base.subprocess.Popen = original_popen
        base.case_directory = ORIGINAL_CASE_DIRECTORY
        if 'range' in base.__dict__:
            del base.range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--ensemble', type=int)
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    if socket.gethostname() != 'login26':
        raise ValueError('original login26 CPU lease required')
    if args.worker is not None:
        worker(args.worker, args.ensemble, args.steps)
        return
    if os.sched_getaffinity(0) != {12}:
        raise ValueError('controller on CPU12 only')
    if args.detach:
        quiescent_original()
        log = base.OUT / 'budget_extension_128k_v1.log'
        launch = base.OUT / 'budget_extension_128k_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():
            raise ValueError('inspect actual prior extension; never duplicate')
        with log.open('x') as stream:
            process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        base.save_json(launch, dict(pid=process.pid, host=socket.gethostname(), CPU_affinity=[12],
                                  created_UTC=base.stamp(), log=str(log)))
        print(json.dumps(read(launch)), flush=True)
    else:
        pipeline()


if __name__ == '__main__':
    main()
