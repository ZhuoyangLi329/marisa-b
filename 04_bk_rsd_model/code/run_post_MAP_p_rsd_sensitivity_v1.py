"""模块大纲：固定真正post-P校准的联合MAP p，重跑九组RSD诊断链。

复用既有四独立ensemble采样、三参数P0和4+4 P+B及完整链诊断。
只另开输出目录、显式注入固定物理p并重折叠B的p轴；不改旧源码、
mean100数据、C_single500/V1、先验、窗口、噪声模型或no-IR口径。
最后独立积分三组P0，并与原post中位数p和实空间p逐项比较。
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
import traceback
import numpy as np
import run_realspace_p_rsd_sensitivity_v1 as sampler
from post_rsd_noir_polynomial import collapse_physical_p
from post_rsd_owner_quiescence import owner_quiescent

core = sampler.core
BASE = sampler.BASE
REALSPACE_BASE = sampler.OUT
BASE_NAMES = dict(sampler.BASE_NAMES)
OUT = core.PART / 'outputs/sto_marisa_post_MAP_p_rsd_20261011_v1'
CAL = BASE / 'cases/postP_physical_p_fNLfixed100'
PHYSICAL_P = 1.2494076995227128
ORIGINAL_SAMPLER_FILE = Path(sampler.__file__).resolve()


def directory(index):
    return OUT / 'cases' / BASE_NAMES[index].replace('_postPcal', '_postMAPp')


# 仅适配本进程中的输出路由和worker入口，磁盘上的共享实现保持原字节。
core.OUT = OUT
core.directory = directory
sampler.OUT = OUT
sampler.PHYSICAL_P = PHYSICAL_P
sampler.directory = directory
sampler.__file__ = str(Path(__file__).resolve())


def freeze():
    if (OUT / 'SETTINGS.json').exists():
        core.guard()
        return
    OUT.mkdir(exist_ok=True)
    choice = core.read(BASE / 'P_CHOICE.json')
    calmap = core.read(CAL / 'MAP.json')
    if choice['MAP_p'] != PHYSICAL_P or calmap['center'][0] != PHYSICAL_P:
        raise ValueError('固定值必须来自原post-P联合MAP完整精度')
    if not core.read(CAL / 'COMPLETION_AUDIT.json')['converged']:
        raise ValueError('原post-P校准没有完成')
    settings = core.read(BASE / 'SETTINGS.json')
    for task_file, want in settings['immutable_inputs'].items():
        if core.digest(task_file) != want:
            raise ValueError('原模型或输入改变: ' + task_file)
    source_files = [Path(__file__).resolve(), ORIGINAL_SAMPLER_FILE, BASE / 'SETTINGS.json',
                    BASE / 'P_CHOICE.json', CAL / 'MAP.json', CAL / 'contract.json',
                    CAL / 'SUMMARY.json', CAL / 'COMPLETION_AUDIT.json']
    source_files += [Path(task_file) for task_file in settings['immutable_inputs']]
    selection = dict(physical_p=PHYSICAL_P, estimator='joint posterior MAP',
        p_origin='same true post-RSD P0 calibration; fixed fNL=100',
        calibration_data_stage='P_post after Marisa displacement', calibration_count=100,
        original_marginal_p_median=choice['physical_p'], calibration_MAP_file=str(CAL / 'MAP.json'),
        calibration_MAP_sha256=core.digest(CAL / 'MAP.json'),
        new_calibration_performed=False, p_uncertainty_propagated=False,
        MAP_roundtrip_not_independent_mock_recovery=True)
    core.save_json(OUT / 'P_CHOICE.json', selection)
    source_files.append(OUT / 'P_CHOICE.json')
    tensor = core.arrays(core.TENSOR)
    for index in range(1, 10):
        src = BASE / 'cases' / BASE_NAMES[index]
        meta = core.read(src / 'contract.json')
        proof = core.read(src / 'COMPLETION_AUDIT.json')
        if (core.digest(src / 'likelihood.npz') != meta['likelihood_sha256'] or not proof['converged']
                or core.digest(src / 'SUMMARY.json') != proof['summary_sha256']):
            raise ValueError('原基线似然或结果改变')
        data = core.arrays(src / 'likelihood.npz')
        original = {key: value.copy() for key, value in data.items()}
        if index >= 4:
            loc = np.array([int(np.flatnonzero(tensor['indices'] == value)[0]) for value in data['indices']])
            np.testing.assert_allclose(collapse_physical_p(tensor['coefficients'],
                p_halo=meta['p_fixed_physical'])[..., loc], data['coefficients'], rtol=3e-13, atol=1e-8)
            data['coefficients'] = collapse_physical_p(tensor['coefficients'], p_halo=PHYSICAL_P)[..., loc]
        for key in data:
            if key != 'coefficients':
                np.testing.assert_array_equal(data[key], original[key])
        meta.update(name=directory(index).name, p_fixed_physical=PHYSICAL_P,
            previous_fixed_p=choice['physical_p'], source_case=str(src),
            p_estimator='joint posterior MAP', p_origin=selection['p_origin'],
            physical_p_calibration_MAP_sha256=selection['calibration_MAP_sha256'],
            p_uncertainty_propagated=False, new_calibration_performed=False,
            only_changed_fixed_physical_p=True, data_covariance_priors_windows_unchanged=True)
        if index >= 4:
            meta['source_P0_case'] = str(directory(1 + core.TRUTHS.index(core.SPECS[index][1])))
        dest = directory(index)
        dest.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(dest / 'likelihood.npz', **data)
        meta['likelihood_sha256'] = core.digest(dest / 'likelihood.npz')
        core.save_json(dest / 'contract.json', meta)
        case_sources = [src / name for name in ('contract.json', 'likelihood.npz', 'SUMMARY.json', 'COMPLETION_AUDIT.json')]
        case_sources += [OUT / 'P_CHOICE.json', CAL / 'MAP.json', core.TENSOR]
        core.save_json(dest / 'SOURCES.json', {str(task_file): core.digest(task_file) for task_file in case_sources})
        source_files += case_sources + [dest / name for name in ('contract.json', 'likelihood.npz', 'SOURCES.json')]
    core.save_json(OUT / 'SETTINGS.json', dict(created_UTC=core.stamp(), physical_p=PHYSICAL_P,
        case_names=[directory(index).name for index in range(1, 10)], case_indices=list(range(1, 10)),
        CPUs=list(core.CPUS), maximum_logical_CPUs=8, main_mean_target_count=100,
        C_single500=True, volume_Gpc_h3=1., P_cut=.08, B_cuts=[.08, .14], IR=False,
        ensembles=4, walkers=32, initial_steps=6000, step_increment=2000, max_steps=64000,
        p_estimator='joint posterior MAP', p_uncertainty_propagated=False,
        immutable_inputs={str(task_file): core.digest(task_file) for task_file in dict.fromkeys(source_files)}))
    for index in range(1, 10):
        core.input_audit(index)
    print('NINE_POST_MAP_P_INPUTS_VERIFIED', PHYSICAL_P, flush=True)


def comparison():
    # 所有writer已由共享sample等待退出并完整回放；controller本身不写HDF。
    from audit_postP_calibrated_grid_v1 import compute
    from report_postP_calibrated_rsd_v1 import record
    core.guard()
    matrix_data = core.arrays(core.MATRIX)
    pos = {tuple(json.loads(str(value))): index for index, value in enumerate(matrix_data['measurement_keys_json'])}
    matrix, covariance = matrix_data['matrix'], matrix_data['covariance_single_box']
    np.testing.assert_allclose(np.cov(matrix[[pos['fiducial', index] for index in range(500)]],
        rowvar=False, ddof=1), covariance, rtol=3e-13, atol=1e-7)
    tensor = core.arrays(core.TENSOR)
    xp = (PHYSICAL_P - 1.3) / .5
    collapsed = tensor['coefficients'][:, 0] + xp * tensor['coefficients'][:, 1] + (2*xp*xp-1)*tensor['coefficients'][:, 2]
    dest = OUT / 'final_comparison_v1'
    dest.mkdir(exist_ok=False)
    grid_dest = OUT / 'analytic_P0_grid_v1'
    grid_dest.mkdir(exist_ok=False)
    rows, grids, proofs = [], [], []
    for index in range(1, 10):
        core.verify_sources(index)
        case = core.case_at(index)
        meta = case.meta
        proof = core.read(case.directory / 'COMPLETION_AUDIT.json')
        if not proof['converged'] or core.digest(case.directory / 'SUMMARY.json') != proof['summary_sha256']:
            raise ValueError('实际采样证据不一致')
        for ensemble, want in enumerate(proof['chain_sha256']):
            if not owner_quiescent(core.read(case.directory / f'PROGRESS_ensemble_{ensemble}.json')):
                raise ValueError('不能读取活动writer')
            if core.digest(case.directory / f'chain_ensemble_{ensemble}.h5') != want:
                raise ValueError('链改变')
        original = core.arrays(Path(meta['source_case']) / 'likelihood.npz')
        for key in ('target', 'covariance', 'chol', 'templates', 'bounds'):
            np.testing.assert_array_equal(getattr(case, key), original[key])
        kind, truth, cut = core.SPECS[index]
        columns = np.arange(12, 24)
        if kind == 'PB':
            loc = np.array([int(np.flatnonzero(matrix_data['B_indices'] == value)[0]) for value in case.indices])
            columns = np.r_[columns, 24 + loc]
            np.testing.assert_allclose(case.coefficients, collapsed[..., loc], rtol=3e-13, atol=1e-7)
            for key in ('indices', 'primitives', 'weight_moments'):
                np.testing.assert_array_equal(getattr(case, key), original[key])
            if len(case.names) != 8 or meta['noise_parameter_count'] != 4 or not meta['P_B_cross_covariance']:
                raise ValueError('P+B参数或cross改变')
        else:
            np.testing.assert_array_equal(case.noise, original['noise'])
            grids.append(compute(case.directory, 'fNL', grid_dest))
        np.testing.assert_array_equal(case.target, matrix[[pos[truth, value] for value in range(100)]][:, columns].mean(axis=0))
        np.testing.assert_array_equal(case.covariance, covariance[np.ix_(columns, columns)])
        if (meta['p_fixed_physical'] != PHYSICAL_P or meta['n_target'] != 100 or meta['n_covariance'] != 500
                or meta['volume_Gpc_h3'] != 1 or meta['covariance_multiplier'] != 1
                or meta['covariance_is_of_mean'] or meta['IR'] or meta['kmax_P'] != .08):
            raise ValueError('统计口径改变')
        new = record(case.directory / 'SUMMARY.json')
        old = record(Path(meta['source_case']) / 'SUMMARY.json')
        realpath = REALSPACE_BASE / 'cases' / BASE_NAMES[index].replace('_postPcal', '_realspaceP') / 'SUMMARY.json'
        real = record(realpath)
        if core.digest(realpath) != core.read(realpath.parent / 'COMPLETION_AUDIT.json')['summary_sha256']:
            raise ValueError('实空间p敏感性对照改变')
        rows.append(dict(statistic='P0 only' if kind == 'P0' else f'P+B kB={cut:.2f}',
            kmax_B=cut, truth_fNL=meta['truth_fNL'], post_MAP_p=new, post_median_p=old,
            realspace_p=real, delta_median_MAPp_minus_medianp=new['fNL_median']-old['fNL_median'],
            sigma_ratio_MAPp_over_medianp=new['fNL_sigma']/old['fNL_sigma'],
            joint_MAP_fNL=core.read(case.directory / 'MAP.json')['center'][0]))
        proofs.append(proof)
    grid = dict(status='POST_MAP_P_P0_INDEPENDENT_INTEGRATION_COMPLETE', checks=grids,
        fixed_p=PHYSICAL_P, HDF_read=False, inputs_priors_covariance_MCMC_unmodified=True)
    core.save_json(grid_dest / 'AUDIT.json', grid)
    result = dict(status='NINE_POST_MAP_P_RSD_FITS_VERIFIED_CONDITIONAL_DIAGNOSTIC',
        created_UTC=core.stamp(), p_choice=core.read(OUT / 'P_CHOICE.json'), comparison_rows=rows,
        main_data_mean_count=100, n_covariance=500, C_single=True, volume_Gpc_h3=1.,
        kmax_P=.08, kmax_B=[.08, .14], PB_physics_plus_noise=[4, 4], P0_active_parameters=3,
        IR=False, p_uncertainty_propagated=False, full_P_B_cross_covariance=True,
        covariance_target_overlap_ids=list(range(100)), calibration_positive_target_overlap=True,
        full_STO_noise_closure_certified=False, original_models_priors_data_C_windows_unchanged=True,
        all_nine_original_sampling_gates_verified=True, chain_audits=proofs, independent_P0_grid=grid)
    core.save_json(dest / 'COMPARISON.json', result)
    print('POST_MAP_P_COMPARISON_READY', flush=True)
    for row in rows:
        new = row['post_MAP_p']
        print(row['statistic'], row['truth_fNL'], new['fNL_median'], new['fNL_sigma'], new['steps'], flush=True)


def pipeline():
    state = dict(pid=os.getpid(), host=socket.gethostname(), CPU_affinity=sorted(os.sched_getaffinity(0)),
        started_UTC=core.stamp(), state='PREPARING_FIXED_POST_MAP_P', completed_cases=[], physical_p=PHYSICAL_P)
    def update(**values):
        state.update(values, updated_UTC=core.stamp())
        core.save_json(OUT / 'STATUS.json', state)
    update()
    try:
        freeze()
        audits = []
        for index in range(1, 10):
            update(state='SAMPLING_P0_ONLY' if index < 4 else 'SAMPLING_4PLUS4_PB')
            audits.append(sampler.sample(index, update))
            state['completed_cases'].append(directory(index).name)
            update()
            if index == 3:
                core.save_json(OUT / 'P0_RESULTS.json', dict(physical_p=PHYSICAL_P,
                    reports=[core.read(directory(value) / 'SUMMARY.json') for value in range(1, 4)]))
        core.save_json(OUT / 'RESULTS.json', dict(physical_p=PHYSICAL_P,
            reports=[core.read(directory(value) / 'SUMMARY.json') for value in range(1, 10)],
            audits=audits, all_nine_original_sampling_gates_verified=True))
        update(state='NINE_CHAINS_VERIFIED_COMPARING', current_case=None, worker_pids=[])
        comparison()
        update(state='COMPLETE_NINE_CHAINS_AND_COMPARISON_VERIFIED', finished_UTC=core.stamp())
    except BaseException:
        update(state='FAILED_OR_INTERRUPTED_PRESERVED', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--detach', action='store_true')
    ap.add_argument('--worker', type=int)
    ap.add_argument('--ensemble', type=int)
    ap.add_argument('--steps', type=int)
    args = ap.parse_args()
    if socket.gethostname() != 'login26':
        raise ValueError('本轮沿用login26且最多8核')
    if args.worker is not None:
        core.worker(args.worker, args.ensemble, args.steps)
    elif args.prepare_only:
        freeze()
    elif args.detach:
        if os.sched_getaffinity(0) != {12}:
            raise ValueError('controller沿用单核12')
        OUT.mkdir(exist_ok=True)
        if (OUT / 'LAUNCH.json').exists() or (OUT / 'controller.log').exists():
            raise ValueError('已有启动记录，不重复提交')
        with (OUT / 'controller.log').open('x') as stream:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        core.save_json(OUT / 'LAUNCH.json', dict(pid=proc.pid, host=socket.gethostname(),
            CPUs=list(core.CPUS), physical_p=PHYSICAL_P, created_UTC=core.stamp()))
        print(json.dumps(core.read(OUT / 'LAUNCH.json')), flush=True)
    else:
        pipeline()
