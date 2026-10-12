"""模块大纲：目标300盒逐文件/来源/实际post权重验收，独立汇总mean100。

只读已完成的原canonical测量；复用原load_pilot/统计合同，不动生产队列。
重新计算POST_HALOS的N2/N3，保留mean(N2²)与mean(N2)²及两种B定义。
不创建主/部分协方差，不选择正式估计器，不进行校准或放行主MCMC。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
from audit_sto_post_fid20_pilots import load_pilot, sha
from assemble_sto_post_measurements import selection_from_record, CUTS, COUNTS
from post_rsd_target300_summary import TARGET_KEYS, TRUTHS, summarize_complete_targets

PART = Path(__file__).resolve().parents[1]
ROOT = PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT = ROOT/'target300_completion_v1'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def directory(key):
    c, i = key
    return ROOT/'pilots'/f'{c}_real{i:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'


def main():
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {13}:
        raise ValueError('short read-only target audit on known single login26 CPU13 required')
    if OUT.exists():
        raise ValueError('preserve existing target audit, inspect actual process, no overwrite')
    missing = [key for key in TARGET_KEYS if not (directory(key)/'DONE.json').exists()]
    if missing:
        raise ValueError('actual original target300 incomplete; no partial means: '+str(missing))
    selected_path = ROOT/'ENSEMBLE_SELECTION.json'
    selection_from_record(read(selected_path))
    source_path = ROOT/'VALIDATION_SOURCE_AUDIT.json'; source_audit = read(source_path)
    sources = {(v['cosmology'], v['realization']): v for v in source_audit['source_rows']}
    if (source_audit['status'] != 'ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE'
            or len(source_audit['source_rows']) != 300 or set(sources) != set(TARGET_KEYS)):
        raise ValueError('exact unchanged target300 source admission required')
    owner_contract = ROOT/'compute8_measurement_series_v3/CONTRACT.json'
    protected = read(owner_contract)['immutable_sources']
    reference_path = PART/'outputs/realspace_binning_reference/REALSPACE_BINNING.npz'
    window_path = ROOT/'calibration_holdout_p0/P0_WINDOW.npz'
    files = [Path(__file__), PART/'code/post_rsd_target300_summary.py',
        PART/'code/audit_sto_post_fid20_pilots.py', PART/'code/assemble_sto_post_measurements.py',
        PART/'code/post_rsd_ensemble.py', selected_path, source_path, owner_contract, reference_path, window_path]
    frozen = {**protected, **{str(path): sha(path) for path in files}}
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError('active production source drift: '+path)
    with np.load(reference_path, allow_pickle=False) as payload:
        reference = {k: np.array(payload[k]) for k in payload.files}
    indices = reference['certified_indices']
    if tuple(int(np.sum(np.max(reference['B000_k'][indices], axis=1) <= cut+1e-12)) for cut in CUTS) != COUNTS:
        raise ValueError('original six-cut B registry changed')
    with np.load(window_path, allow_pickle=False) as payload:
        p_edges = np.array(payload['edges']); p_modes = np.array(payload['nmodes']); p_k = np.array(payload['k'])
    OUT.mkdir()
    contract = dict(scope='actual target300 completion and mean100 only, not main covariance or inference',
        immutable_sources=frozen, target_keys=[list(v) for v in TARGET_KEYS],
        n_target_per_truth=100, required_main_covariance_count=500,
        covariance_ids=[0, 499], explicit_target_overlap=True,
        covariance_is_of_mean=False, covariance_multiplier=1., volume_Gpc_h3=1.,
        B_definition_chosen=False, physical_p_calibrated=False, raw_STO_noise_calibrated=False,
        main_MCMC_released=False, CPU_affinity=[13], numeric_threads=1)
    save(OUT/'CONTRACT.json', contract); contract_sha = sha(OUT/'CONTRACT.json')
    start = time.monotonic()
    state = dict(state='RUNNING', pid=os.getpid(), host=socket.gethostname(),
        starttime_ticks=(Path('/proc/self/stat')).read_text().rsplit(')', 1)[1].split()[19],
        started_UTC=datetime.now(timezone.utc).isoformat(), completed=0, required=300,
        contract_sha256=contract_sha, CPU_affinity=[13])

    def update(**values):
        state.update(values, updated_UTC=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        save(OUT/'STATUS.json', state); print(json.dumps(state), flush=True)

    def guard():
        if sha(OUT/'CONTRACT.json') != contract_sha:
            raise ValueError('target audit contract changed')
        for path, expected in frozen.items():
            if sha(path) != expected:
                raise ValueError('frozen source/input changed during target audit: '+path)

    matched = []; nonzero = []; moments = []; records = []; first = None
    try:
        update(); guard()
        pilot_sha = frozen[str(PART/'code/run_sto_post_rsd_pilot.py')]
        for number, key in enumerate(TARGET_KEYS):
            data, meta, done = load_pilot(directory(key)); source = sources[key]
            expected = dict(cosmology=key[0], realization=key[1], recon_mesh=256,
                measurement_mesh=256, nran=30, self_contraction='matched',
                zero_atom_diagnostics=True, nested_randoms=True, grid_origin_fraction=0.)
            if any(meta['arguments'][name] != value for name, value in expected.items()):
                raise ValueError('mixed original target measurement settings')
            if (done['entry_sha256'] != pilot_sha or len(done['CPU_affinity']) != 1
                    or meta['source_STO_sha256'] != source['source_sha256']
                    or meta['source_diagnostics_sha256'] != source['diagnostics_sha256']
                    or meta['raw_fof']['source_raw_sha256'] != source['raw_fof']['source_raw_sha256']
                    or not meta['data_has_rsd'] or meta['redshift'] != .5 or meta['LOS'] != 'z'):
                raise ValueError('actual target source/space/resource identity drift')
            if first is None:
                first = data, meta
                fixed = meta['fixed_reconstruction']
                required = dict(b_rec=2.2280546353635335, p_rec=1.2822, fNL_rec=100.,
                    algorithm_growth_rate=0., physical_growth_rate=.7625790037720044)
                if any(fixed[name] != value for name, value in required.items()):
                    raise ValueError('wrong frozen reconstruction or physical growth')
            if meta['fixed_reconstruction'] != first[1]['fixed_reconstruction']:
                raise ValueError('mixed fixed reconstruction')
            np.testing.assert_array_equal(data['B_edges'], reference['edges'])
            np.testing.assert_allclose(data['B_post_k'], reference['B000_k'], rtol=0., atol=5e-5)
            np.testing.assert_array_equal(data['B_post_nmodes'], first[0]['B_post_nmodes'])
            np.testing.assert_array_equal(data['P_edges'], p_edges)
            for label in ('P_weighted_input', 'P_post'):
                if data[label].shape != (12,):
                    raise ValueError('P bin dimension changed')
                np.testing.assert_array_equal(data[label+'_nmodes'], p_modes)
                np.testing.assert_allclose(data[label+'_k'], p_k, rtol=3e-6, atol=2e-8)
            if data['B_post'].shape != (120,):
                raise ValueError('original B registry dimension changed')
            with np.load(directory(key)/'POST_HALOS.npz', allow_pickle=False) as payload:
                weights = np.asarray(payload['weights'], dtype=float)
            if (len(weights) != source['nhalo'] or len(weights) != meta['raw_fof']['n_halo']
                    or not np.isfinite(weights).all() or weights.sum() <= 0):
                raise ValueError('actual post weights/halo count invalid')
            total = weights.sum(); n2 = float(1e9*np.sum(weights*weights)/total**2)
            n3 = float(1e18*np.sum(weights**3)/total**3)
            np.testing.assert_allclose([n2, n3], [source['N2_data_self'], source['N3_data_self']],
                                       rtol=8*np.finfo(float).eps, atol=0.)
            negative = int(np.count_nonzero(weights < 0))
            if negative != source['negative_weights']:
                raise ValueError('post reconstruction altered signed STO weights')
            common = np.r_[data['P_weighted_input'], data['P_post']]
            matched.append(np.r_[common, data['B_post'][indices]])
            nonzero.append(np.r_[common, data['B_post_nonzero_only_counterfactual'][indices]])
            moments.append([n2, n3, len(weights)])
            records.append(dict(cosmology=key[0], realization=key[1], DONE_sha256=sha(directory(key)/'DONE.json'),
                contract_sha256=done['contract_sha256'], measurement_sha256=done['measurement_sha256'],
                post_halos_sha256=done['post_halos_sha256'], source_STO_sha256=source['source_sha256'],
                raw_FoF_sha256=source['raw_fof']['source_raw_sha256'], N2_data_self=n2, N3_data_self=n3,
                nhalo=len(weights), negative_weights=negative))
            del weights; gc.collect()
            if number % 20 == 19:
                guard(); update(completed=number+1, last_key=list(key))
        arrays, self_moments = summarize_complete_targets(TARGET_KEYS, matched, nonzero, moments)
        zero_rows = []
        for truth, cosmology in enumerate(TRUTHS):
            original = arrays['matched_target_vectors'][truth, 24:]
            changed = arrays['nonzero_diagnostic_target_vectors'][truth, 24:]
            cuts = []
            for cut, count in zip(CUTS, COUNTS):
                mask = np.max(reference['B000_k'][indices], axis=1) <= cut+1e-12
                delta = changed[mask]-original[mask]
                norm = float(np.linalg.norm(original[mask]))
                if norm == 0:
                    raise ValueError('zero mean B norm; do not hide estimator diagnostic')
                cuts.append(dict(kmax_B=cut, bins=count, relative_L2_change=float(np.linalg.norm(delta)/norm),
                    maximum_abs_change_Mpc_h6=float(np.max(np.abs(delta))),
                    covariance_whitened=False, significance_assigned=False))
            zero_rows.append(dict(cosmology=cosmology, n_target=100, by_cut=cuts))
        guard()
        matrix_path = OUT/'TARGETS_ONLY.npz'
        np.savez_compressed(matrix_path, **arrays, matched_rows=np.array(matched),
            nonzero_diagnostic_rows=np.array(nonzero), target_keys_json=np.array([json.dumps(v) for v in TARGET_KEYS]),
            B_indices=indices, B_k=reference['B000_k'][indices], P_edges=p_edges, P_k=p_k, P_nmodes=p_modes)
        report = dict(status='ALL300_TARGET_MEASUREMENTS_MEAN100_AND_POST_WEIGHT_MOMENTS_VERIFIED_NOT_COVARIANCE_RELEASE',
            created_UTC=datetime.now(timezone.utc).isoformat(), contract_sha256=contract_sha,
            target_matrix_path=str(matrix_path), target_matrix_sha256=sha(matrix_path), rows=records,
            n_targets=300, n_per_truth=100, truths=[0, 100, -100], B_bins=44, P_post_bins=12,
            fixed_reconstruction=first[1]['fixed_reconstruction'], self_moment_statistics=self_moments,
            estimator_zero_atom_diagnostics=zero_rows, post_weights_recomputed_for_all300=True,
            main_covariance_created=False, partial_covariance_created=False, covariance_of_mean_created=False,
            physical_p_calibrated=False, raw_STO_noise_calibrated=False, B_definition_chosen=False,
            main_MCMC_released=False, volume_Gpc_h3=1., covariance_multiplier=1.,
            future_full700_target_means_must_agree=True, actual_CPU_affinity=[13],
            elapsed_seconds=time.monotonic()-start,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        save(OUT/'AUDIT.json', report); update(state='COMPLETE', completed=300)
    except Exception:
        update(state='FAILED_PRESERVED', error=traceback.format_exc()); raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--detach']:
        if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {13}:
            raise ValueError('launcher must inherit single original CPU13')
        log = ROOT/'target300_completion_v1.log'; launch = ROOT/'target300_completion_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():
            raise ValueError('existing target audit must be inspected, no duplicate launch')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(pid=proc.pid, host=socket.gethostname(), command=command,
            source_sha256=sha(Path(__file__)), CPU_affinity=[13], log=str(log),
            launched_UTC=datetime.now(timezone.utc).isoformat())
        with launch.open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
        print(json.dumps(value), flush=True)
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError('only original target audit or --detach launcher supported')
