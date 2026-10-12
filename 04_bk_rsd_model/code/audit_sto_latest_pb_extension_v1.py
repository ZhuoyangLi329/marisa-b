"""模块大纲：128k预算续跑终止后的全链/原前缀/原门独立验收。

不读取活动HDF；通过只表示402盒provisional三组P+B采样与输入验收，
不认证完整STO噪声、IR、正式fNL恢复、kmax或原C500/39组主流程。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import socket
import numpy as np
import emcee
import extend_sto_latest_pb_budget_v1 as extension
import audit_sto_latest_pb_inference_v1 as original
from post_rsd_owner_quiescence import owner_quiescent


def main():
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {12}:
        raise ValueError('original login26 CPU12 audit only')
    try:
        extension.quiescent_original()
    except ValueError as error:
        if 'still live' in str(error):
            print('NOT_READY_ACTUAL_ORIGINAL_OWNER_OR_WRITER_LIVE_NO_HDF_READ', flush=True)
            return
        raise
    if not (extension.OUT / 'STATUS.json').exists():
        print('NOT_READY_ACTUAL_EXTENSION_NOT_STARTED_NO_HDF_READ', flush=True)
        return
    state = original.read(extension.OUT / 'STATUS.json')
    if not owner_quiescent(state):
        print('NOT_READY_ACTUAL_EXTENSION_OWNER_LIVE_NO_HDF_READ', flush=True)
        return
    for path in (extension.OUT / 'pb_cases').glob('*/PROGRESS_ensemble_*.json'):
        if not owner_quiescent(original.read(path)):
            print('NOT_READY_ACTUAL_EXTENSION_WRITER_LIVE_NO_HDF_READ', flush=True)
            return
    if (state['state'] != 'BUDGET_ONLY_ALL3_PB_TRIALS_FINISHED_REVIEW_CONVERGENCE'
            or state['completed_pb_cases'] != list(original.run.COSMOS)):
        print('NOT_READY_ALL_THREE_ACTUAL_EXTENSION_CASES_NOT_FINISHED', flush=True)
        return
    contract = extension.guard(full_original_hdf=True)
    inputs = original.input_audit()
    controls = original.read(original.OUT / 'P_CONTROL_COMPLETION_AUDIT.json')
    if not controls['all_converged'] or len(controls['checks']) != 4:
        raise ValueError('actual physical-p and three P0 original-gate replay required')
    checks = []
    for row in contract['records']:
        index = row['index']
        old_directory = Path(row['original_directory'])
        directory = old_directory if row['already_converged'] else Path(row['extension_directory'])
        case = original.run.LatestPBCase(directory)
        case.initialize()
        stored = original.read(directory / 'SUMMARY.json')
        backends = [emcee.backends.HDFBackend(str(directory / f'chain_ensemble_{j}.h5'),
                                             name=f'ensemble_{j}') for j in range(4)]
        iterations = [int(backend.iteration) for backend in backends]
        if iterations != [stored['steps']] * 4 or stored['steps'] > extension.MAXIMUM:
            raise ValueError('actual equal four-ensemble declared checkpoint required')
        if not row['already_converged']:
            for j, backend in enumerate(backends):
                old = emcee.backends.HDFBackend(str(old_directory / f'chain_ensemble_{j}.h5'),
                                                name=f'ensemble_{j}')
                if int(old.iteration) != row['initial_steps']:
                    raise ValueError('original preserved HDF length drift')
                for start in range(0, row['initial_steps'], 4000):
                    stop = min(start + 4000, row['initial_steps'])
                    # 两个原生backend都已确认无活动writer；切片只用于降低比较内存。
                    with old.open() as source, backend.open() as copied:
                        for name in ('chain', 'log_prob'):
                            np.testing.assert_array_equal(source[old.name][name][start:stop],
                                                          copied[backend.name][name][start:stop])
        chains = np.stack([backend.get_chain() for backend in backends])
        accepted = np.stack([backend.accepted for backend in backends])
        if chains.shape != (4, stored['steps'], 32, 12):
            raise ValueError('complete actual four32walker12parameter chains required')
        physical = case.physical(chains)
        if (not np.isfinite(physical).all() or np.any(physical <= case.bounds[:, 0])
                or np.any(physical >= case.bounds[:, 1])):
            raise ValueError('actual full samples violate unchanged physical priors')
        replay, flat = original.sampler.diagnose(case, chains, accepted, stored['steps'])
        if replay['gates'] != stored['gates'] or replay['converged'] != stored['converged']:
            raise ValueError('actual original sampling-gate replay mismatch')
        np.testing.assert_allclose(replay['quantiles_2p5_16_50_84_97p5'],
                                   stored['quantiles_2p5_16_50_84_97p5'], rtol=2e-13, atol=1e-12)
        np.testing.assert_allclose(replay['fNL_sigma'], stored['fNL_sigma'], rtol=2e-13)
        with np.load(directory / 'POSTERIOR_THIN.npz', allow_pickle=False) as thin:
            np.testing.assert_array_equal(thin['parameter_names'], case.names)
            pick = np.random.default_rng(5500 + index).choice(len(flat), min(30000, len(flat)), replace=False)
            np.testing.assert_array_equal(thin['samples'], flat[pick])
        p0 = original.read(original.run.POUT / 'cases' / original.run.pbase.NAMES[index + 1] / 'SUMMARY.json')
        q = np.array(replay['quantiles_2p5_16_50_84_97p5'])[:, 0]
        checks.append(dict(cosmology=row['cosmology'], truth_fNL=case.meta['truth_fNL'],
            directory=str(directory), steps_each_ensemble=stored['steps'], actual_iterations=iterations,
            original64k_prefix_bitwise_identical=not row['already_converged'],
            already_converged_case_untouched=row['already_converged'],
            original_gates=replay['gates'], converged=bool(replay['converged']),
            rank_rhat_max=float(np.max(replay['rank_normalized_split_rhat'])),
            length_over_tau_min=float(np.min(replay['length_over_tau'])),
            fNL_quantiles_2p5_16_50_84_97p5=q.tolist(), fNL_sigma=float(replay['fNL_sigma']),
            P0_fNL_quantiles_2p5_16_50_84_97p5=np.array(p0['quantiles_2p5_16_50_84_97p5'])[:, 0].tolist(),
            P0_fNL_sigma=float(p0['fNL_sigma']),
            PB_to_P0_fNL_sigma_ratio=float(replay['fNL_sigma'] / p0['fNL_sigma']),
            boundary_mass_each_parameter=np.array(replay['boundary_mass_each_parameter']).tolist(),
            prior_sensitive=bool(replay['prior_sensitive']), minuit_valid=bool(case.initialize()['minuit_valid']),
            summary_sha256=original.digest(directory / 'SUMMARY.json')))
        del chains, physical, flat, backends
    all_converged = all(row['converged'] for row in checks)
    result = dict(status=('THREE_PRELIMINARY_PB_BUDGET_ONLY_CHAINS_FULLY_REPLAYED_NOT_FORMAL_NOISE'
                         if all_converged else 'BUDGET_EXTENSION_FINISHED_ORIGINAL_SAMPLING_GATES_FAILED'),
        created_UTC=original.stamp(), all3_converged=all_converged, checks=checks,
        original64k_products_and_HDF_files_unchanged=True, n_covariance=402, n_target_per_truth=100,
        C_single=True, covariance_ddof=1, covariance_is_of_mean=False, covariance_multiplier=1.,
        volume_Gpc_h3=1., P_B_cross_covariance=True, P_bins=12, B_bins=9, kmax_P=.08, kmax_B=.08,
        p_fixed_physical=inputs['p_fixed_physical'], p_uncertainty_propagated=False,
        input_audit_sha256=original.digest(original.OUT / 'INDEPENDENT_INPUT_AUDIT.json'),
        p_control_full_HDF_audit_sha256=original.digest(original.OUT / 'P_CONTROL_COMPLETION_AUDIT.json'),
        extension_contract_sha256=original.digest(extension.OUT / 'CONTRACT.json'),
        audit_source_sha256=original.digest(__file__), diagnostic_only=True, provisional=True, IR=False,
        full_STO_noise_closure_certified=False, formal_fNL_or_kmax_released=False,
        original700_and_C500_workflow_replaced=False)
    original.save_json(extension.OUT / 'MCMC_COMPLETION_AUDIT.json', original.sampler.clean(result))
    print(json.dumps(original.sampler.clean(result)), flush=True)


if __name__ == '__main__':
    main()
