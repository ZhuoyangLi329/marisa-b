"""模块大纲：原pre测量入口+post同一self adapter；保留旧pre并核验raw B回放。

仅fiducial0..19的具名pilot，默认real000。复用原目录读取、随机生成、周期边界
处理与B测量；不重构、不改权重为STO、不决定正式covariance或运行MCMC。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import measure_rsd_b000 as original
from rsd_b000_contract import PART, LEGACY_OUT, digest, selected

ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'
# 沿用rsd_b000_contract.freeze_contract中max-whitened数值尺度；k均值仅作坐标标签。
RAW_REPLAY_MAX_WHITENED_ERROR = 1e-5


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--realization', type=int, default=0); ap.add_argument('--detach', action='store_true')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('one logical CPU required')
    if not 0 <= args.realization < 20: raise ValueError('only already-audited fiducial pilot IDs 0..19')
    tag = f'fiducial_real{args.realization:03d}'
    out = ROOT / 'matched_pre_pilots' / (tag + '_meas256_ran30_v2')
    if args.detach:
        launch = ROOT / (tag + '_matched_pre_pilot_v2.launch.json')
        log = ROOT / (tag + '_matched_pre_pilot_v2.log')
        if launch.exists() or log.exists() or out.exists(): raise RuntimeError('inspect prior pre-pilot execution')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--realization', str(args.realization)]
        with log.open('xb') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        record = {'host': socket.gethostname(), 'pid': process.pid, 'new_session': True,
            'CPU_affinity': sorted(os.sched_getaffinity(0)), 'command': command, 'log': str(log),
            'created_UTC': datetime.now(timezone.utc).isoformat()}
        with launch.open('x') as stream: json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True); return
    if out.exists(): raise RuntimeError('existing pilot must not be overwritten')
    for name, status in (('MATCHED_SELF_AUDIT.json', 'MATCHED_SELF_CONTRACTION_TEST_PASS'),
                         ('ZERO_ATOM_AUDIT.json', 'ZERO_ATOM_DIAGNOSTIC_TEST_PASS')):
        if json.loads((ROOT / name).read_text())['status'] != status: raise ValueError('missing point-process gate')
    adapter = PART / 'code/post_rsd_estimator.py'
    if digest(adapter) != json.loads((ROOT / 'MATCHED_SELF_AUDIT.json').read_text())['adapter_sha256']:
        raise ValueError('self adapter changed since the independent point test')
    legacy_path = LEGACY_OUT / 'measurements_m256' / (tag + '.npz')
    with np.load(legacy_path, allow_pickle=False) as z:
        legacy = {k: np.array(z[k]) for k in z.files}
    legacy_meta = json.loads(str(legacy['metadata_json']))
    sys.path.insert(0, str(PART.parent / '01_reconstruction_methods/common'))
    sys.path.insert(0, str(PART.parent / '01_reconstruction_methods/marisa_gra'))
    import jax
    jax.config.update('jax_enable_x64', True)
    from mpi4py import MPI
    import halo_b000_pipeline as shared
    from post_rsd_estimator import matched_b000_self_contractions
    if MPI.COMM_WORLD.size != 1: raise RuntimeError('single MPI rank required')
    ensemble_path = LEGACY_OUT / 'B000_ENSEMBLE_MATRIX.npz'
    with np.load(ensemble_path, allow_pickle=False) as z:
        ensemble_meta = [json.loads(str(x)) for x in z['metadata_json']]
        cov_rows = [i for i, row in enumerate(ensemble_meta) if row['role'] == 'covariance']
        if len(cov_rows) != 500: raise ValueError('complete legacy500 required for numerical replay scale')
        reference_B = np.array(z['B'][cov_rows])
    fit_indices = selected(.20)
    reference_C = np.cov(reference_B[:, fit_indices], rowvar=False, ddof=1)
    reference_chol = np.linalg.cholesky(reference_C / ((500-len(fit_indices)-2)/499.))
    paths = [Path(__file__), Path(original.__file__), adapter, Path(shared.__file__), ensemble_path,
             PART / 'code/run_sto_matched_pre_pilot.sh',
             PART / 'code/run_rsd_bonly_workflow.py', PART / 'code/export_pre_rsd.py',
             PART / 'code/rsd_b000_contract.py', original.REFERENCE, legacy_path,
             ROOT / 'MATCHED_SELF_AUDIT.json', ROOT / 'ZERO_ATOM_AUDIT.json',
             PART / 'outputs/pre_rsd_b000_bonly_mean100_v1/RESULTS.json',
             PART / 'outputs/pre_rsd_pb_mean100_v1/RESULTS.json',
             PART / 'outputs/pre_rsd_p0only_mean100_v1/RESULTS.json']
    frozen = {str(p): digest(p) for p in paths}
    contract = {'status': 'MATCHED_PRE_ESTIMATOR_PILOT_ONLY',
        'samples': [{'cosmology': 'fiducial', 'realization': args.realization, 'role': 'matched_pre_control', 'pilot': True}],
        'tracer': 'unit-weight pre-RSD halos, not STO weighted input or reconstructed catalog',
        'meshsize': 256, 'nran': 30, 'z': .5, 'LOS': 'z', 'RSD': True, 'reconstruction': False,
        'matched_self_contractions': True, 'zero_atom_counterfactual_saved_separately': True,
        'legacy_pre_path': str(legacy_path), 'immutable_sources': frozen,
        'numerical_replay_covariance': 'legacy pre Fid500 Csingle with its dimension-specific Hartlap; not post covariance',
        'raw_replay_max_whitened_error_gate': RAW_REPLAY_MAX_WHITENED_ERROR,
        'kmean_label_absolute_tolerance': 5e-5,
        'previous_failed_pre_pilot_preserved': str(ROOT / 'matched_pre_pilots' / (tag + '_meas256_ran30')),
        'old_pre_results_modified': False, 'main_sample_choice_changed': False,
        'main_covariance_or_MCMC_released': False}
    out.mkdir(parents=True); save(out / 'contract.json', contract)
    state = {'state': 'RUNNING', 'stage': 'matched_pre_measurement', 'host': socket.gethostname(),
        'pid': os.getpid(), 'CPU_affinity': sorted(os.sched_getaffinity(0)),
        'started_UTC': datetime.now(timezone.utc).isoformat()}
    save(out / 'STATUS.json', state); start = time.time(); records = []; norms = []
    original_measure = shared.measure_b000; original_out = original.OUT; old_argv = sys.argv
    def measure(*values, **kwargs):
        with matched_b000_self_contractions(records, matched_paint=True):
            spectrum = original_measure(*values, **kwargs)
            jax.block_until_ready(spectrum)
            norms.append(np.asarray(spectrum.get((0, 0, 0)).values('norm'), dtype=float))
            return spectrum
    try:
        shared.measure_b000 = measure; original.OUT = out
        sys.argv = [str(original.__file__), '--meshsize', '256']
        original.main()
        if len(records) != 1 or len(norms) != 1: raise ValueError('unexpected measurement count')
        target = out / 'measurements_m256' / (tag + '.npz')
        with np.load(target, allow_pickle=False) as z: current = {k: np.array(z[k]) for k in z.files}
        meta = json.loads(str(current['metadata_json']))
        for field in ('source_raw_sha256', 'random_seed', 'nran', 'meshsize', 'periodic_guard'):
            if meta[field] != legacy_meta[field]: raise ValueError('pre sample/recipe changed: ' + field)
        np.testing.assert_array_equal(current['B000_nmodes'], legacy['B000_nmodes'])
        np.testing.assert_array_equal(current['B000_k_edges'], legacy['B000_k_edges'])
        np.testing.assert_array_equal(current['phase_fingerprint'], legacy['phase_fingerprint'])
        np.testing.assert_allclose(current['B000_k'], legacy['B000_k'], rtol=0., atol=5e-5)
        norm = norms[0]
        if np.any(norm <= 0) or not np.isfinite(norm).all(): raise ValueError('invalid estimator normalization')
        zero = records[0]['zero_leg_num_shotnoise'] / norm
        s113 = records[0]['difference'].real / norm
        difference = current['B_raw'] - legacy['B_raw']
        raw_exact = bool(np.array_equal(current['B_raw'], legacy['B_raw']))
        raw_white = np.linalg.solve(reference_chol, difference[fit_indices])
        raw_error = float(np.max(np.abs(raw_white)))
        raw_pass = raw_error <= RAW_REPLAY_MAX_WHITENED_ERROR
        diagnostics = out / 'SELF_DIAGNOSTICS.npz'
        np.savez_compressed(diagnostics, B_pre_matched=current['B'], B_pre_raw=current['B_raw'],
            B_pre_matched_shot=current['B_shot'], B_pre_nonzero_only=current['B']+zero,
            zero_leg_shot_contribution=zero, zero_leg_pair_fraction=records[0]['zero_leg_pair_fraction'],
            S113_discrete_minus_bincenter=s113, B_legacy=legacy['B'], B_legacy_raw=legacy['B_raw'],
            B_k=current['B000_k'], B_nmodes=current['B000_nmodes'], normalization=norm)
        for path, expected in frozen.items():
            if digest(path) != expected: raise ValueError('frozen input changed: ' + path)
        report = {'status': 'MATCHED_PRE_PILOT_RAW_REPLAY_PASS' if raw_pass else 'MATCHED_PRE_PILOT_RAW_PARITY_REQUIRES_INSPECTION',
            'created_UTC': datetime.now(timezone.utc).isoformat(), 'cosmology': 'fiducial',
            'realization': args.realization, 'raw_B_bitwise_identical_to_legacy': raw_exact,
            'raw_B_max_whitened_error': raw_error, 'raw_B_whitened_L2': float(np.linalg.norm(raw_white)),
            'raw_B_max_whitened_error_gate': RAW_REPLAY_MAX_WHITENED_ERROR,
            'replay_error_scale_is_legacy_pre_Csingle': True,
            'kmean_label_max_absolute_difference': float(np.max(np.abs(current['B000_k']-legacy['B000_k']))),
            'raw_B_max_absolute_difference': float(np.max(np.abs(difference))),
            'raw_B_L2_relative_difference': float(np.linalg.norm(difference)/np.linalg.norm(legacy['B_raw'])),
            'measurement_sha256': digest(target), 'diagnostics_sha256': digest(diagnostics),
            'contract_sha256': digest(out / 'contract.json'), 'legacy_measurement_sha256': digest(legacy_path),
            'raw_FoF_sha256': meta['source_raw_sha256'], 'random_seed': meta['random_seed'],
            'same_B_registry_and_window': True, 'old_pre_results_unchanged': True,
            'P_remeasured': False, 'pre_or_post_covariance_created': False, 'MCMC_released': False,
            'host': socket.gethostname(), 'CPU_affinity': sorted(os.sched_getaffinity(0)),
            'elapsed_seconds': time.time()-start}
        save(out / 'AUDIT.json', report)
        state.update(state='COMPLETE' if raw_pass else 'NEEDS_RAW_PARITY_INSPECTION', stage='complete',
            updated_UTC=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time()-start)
        save(out / 'STATUS.json', state); print(json.dumps(report, indent=2), flush=True)
    except Exception:
        state.update(state='FAILED', error=traceback.format_exc(), updated_UTC=datetime.now(timezone.utc).isoformat())
        save(out / 'STATUS.json', state); raise
    finally:
        shared.measure_b000 = original_measure; original.OUT = original_out; sys.argv = old_argv


if __name__ == '__main__': main()
