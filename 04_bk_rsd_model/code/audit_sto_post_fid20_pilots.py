"""模块大纲：验收20盒同版本post pilot；保存均值/单盒散布；不创建主协方差。

只在队列及全部测量完成后生成产物。单盒网格/随机点对照用pilot单盒std标度，
不宣称均值偏差、全协方差白化、fNL恢复或网格/随机点收敛已经通过。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import numpy as np


PART = Path(__file__).resolve().parents[1]
ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'
CUTS = (.08, .10, .12, .14, .18, .20)
COUNTS = (9, 14, 20, 27, 35, 44)
FIELDS = ('P_weighted_input', 'P_weighted_input_raw', 'P_weighted_input_shot',
          'P_post', 'P_post_raw', 'P_post_shot', 'B_post', 'B_post_raw', 'B_post_shot',
          'B_post_nonzero_only_counterfactual', 'zero_leg_shot_contribution',
          'S113_discrete_minus_bincenter')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pilot(directory):
    done = json.loads((directory / 'DONE.json').read_text())
    contract = json.loads((directory / 'CONTRACT.json').read_text())
    if (done['status'] != 'PILOT_COMPLETE_NOT_SCIENTIFIC_VALIDATION'
            or done['measurement_sha256'] != sha(directory / 'MEASUREMENTS.npz')
            or done['post_halos_sha256'] != sha(directory / 'POST_HALOS.npz')
            or done['contract_sha256'] != sha(directory / 'CONTRACT.json')):
        raise ValueError('pilot completion provenance mismatch: ' + str(directory))
    with np.load(directory / 'MEASUREMENTS.npz', allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    for prefix in ('P_weighted_input', 'P_post', 'B_post'):
        np.testing.assert_allclose(arrays[prefix + '_raw'] - arrays[prefix + '_shot'],
                                   arrays[prefix], rtol=3e-13, atol=1e-6)
    np.testing.assert_array_equal(arrays['B_post_nonzero_only_counterfactual'],
                                  arrays['B_post'] + arrays['zero_leg_shot_contribution'])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('nonfinite pilot measurement')
    return arrays, contract, done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queue', default='pilot_fid20_recovery_20261009')
    ap.add_argument('--check-ready', action='store_true')
    args = ap.parse_args()
    if not args.queue or any(not (c.isalnum() or c in '_-') for c in args.queue):
        raise ValueError('safe queue path component required')
    queue = ROOT / args.queue
    status = json.loads((queue / 'STATUS.json').read_text())
    ready = status['state'] == 'COMPLETE' and status['completed'] == list(range(20)) and not status['workers']
    if args.check_ready:
        print(json.dumps({'ready': ready, 'state': status['state'], 'completed': len(status['completed']),
                          'required': 20, 'artifact_created': False}), flush=True)
        return
    if not ready: raise RuntimeError('complete original twenty-box pilot set required; no partial fallback')
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('single CPU audit only')
    out = ROOT / 'pilot_fid20_diagnostics'
    if out.exists(): raise RuntimeError('preserve existing population diagnostics')
    contract = json.loads((queue / 'CONTRACT.json').read_text())
    for field in ('immutable_sources', 'original_immutable_sources'):
        for path, expected in contract.get(field, {}).items():
            if sha(path) != expected: raise ValueError('frozen source changed: ' + path)
    reference = PART / 'outputs/realspace_binning_reference/REALSPACE_BINNING.npz'
    pre_contract = json.loads((PART / 'outputs/pre_rsd_b000_bonly_mean100_v1/contract.json').read_text())
    if sha(reference) != pre_contract['reference_sha256']: raise ValueError('realspace binning changed')
    with np.load(reference, allow_pickle=False) as z:
        ref = {k: np.array(z[k]) for k in z.files}
    ix = ref['certified_indices']
    if tuple(int(np.sum(np.max(ref['B000_k'][ix], axis=1) <= cut + 1e-12)) for cut in CUTS) != COUNTS:
        raise ValueError('B cutoff registry changed')
    with np.load(ROOT / 'calibration_holdout_p0/P0_WINDOW.npz', allow_pickle=False) as z:
        p_edges = z['edges'].copy(); p_nmodes = z['nmodes'].copy(); p_k = z['k'].copy()
    source_audit = json.loads((ROOT / 'VALIDATION_SOURCE_AUDIT.json').read_text())
    sources = {r['realization']: r for r in source_audit['source_rows'] if r['cosmology'] == 'fiducial'}
    pilot_sha = sha(PART / 'code/run_sto_post_rsd_pilot.py')
    matrices = {field: [] for field in FIELDS}; rows = []; first = None
    for index in range(20):
        directory = ROOT / 'pilots' / f'fiducial_real{index:03d}_rec256_meas256_ran30_selfmatched_zeroaudit_nested'
        data, meta, done = load_pilot(directory); source = sources[index]
        expected = {'cosmology': 'fiducial', 'realization': index, 'recon_mesh': 256, 'measurement_mesh': 256,
                    'nran': 30, 'self_contraction': 'matched', 'zero_atom_diagnostics': True,
                    'nested_randoms': True, 'grid_origin_fraction': 0.}
        if any(meta['arguments'][key] != value for key, value in expected.items()):
            raise ValueError('mixed pilot settings')
        if (done['entry_sha256'] != pilot_sha or meta['source_STO_sha256'] != source['source_sha256']
                or meta['raw_fof']['source_raw_sha256'] != source['raw_fof']['source_raw_sha256']
                or len(done['CPU_affinity']) != 1):
            raise ValueError('pilot source or resource contract mismatch')
        if first is None: first = (data, meta, done)
        if meta['fixed_reconstruction'] != first[1]['fixed_reconstruction']:
            raise ValueError('frozen reconstruction differs between boxes')
        np.testing.assert_allclose(data['B_post_k'], ref['B000_k'], rtol=0., atol=5e-5)
        np.testing.assert_array_equal(data['B_edges'], ref['edges'])
        np.testing.assert_array_equal(data['P_edges'], p_edges)
        for prefix in ('P_weighted_input', 'P_post'):
            np.testing.assert_array_equal(data[prefix + '_nmodes'], p_nmodes)
            np.testing.assert_allclose(data[prefix + '_k'], p_k, rtol=3e-6, atol=2e-8)
            if data[prefix].shape != (12,): raise ValueError('P vector dimension changed')
        if data['B_post'].shape != (120,): raise ValueError('B vector dimension changed')
        for field in FIELDS: matrices[field].append(data[field])
        rows.append({'realization': index, 'DONE_sha256': sha(directory / 'DONE.json'),
                     'measurement_sha256': done['measurement_sha256'], 'post_halos_sha256': done['post_halos_sha256'],
                     'source_STO_sha256': source['source_sha256'], 'raw_sha256': source['raw_fof']['source_raw_sha256'],
                     'N2_data_self': source['N2_data_self'], 'N3_data_self': source['N3_data_self']})
    matrices = {k: np.array(v) for k, v in matrices.items()}
    derived = {}
    for field in ('P_weighted_input', 'P_post', 'B_post', 'B_post_nonzero_only_counterfactual'):
        derived[field + '_pilot_mean'] = matrices[field].mean(axis=0)
        derived[field + '_single_box_std'] = matrices[field].std(axis=0, ddof=1)
    if (np.any(derived['P_post_single_box_std'] <= 0)
            or np.any(derived['B_post_single_box_std'][ix] <= 0)):
        raise ValueError('invalid pilot dispersion')
    controls = []
    for label, name in (
        ('half_cell_origin', 'fiducial_real000_rec256_meas256_ran30_selfmatched_zeroaudit_originhalf'),
        ('nested_random30_to60', 'fiducial_real000_rec256_meas256_ran60_selfmatched_zeroaudit_nested')):
        directory = ROOT / 'pilots' / name
        control, cm, cd = load_pilot(directory)
        for field in ('source_STO_sha256', 'random_seed', 'fixed_reconstruction', 'shared_measure_source_sha256'):
            if cm[field] != first[1][field]: raise ValueError('unpaired numerical control: ' + field)
        if label == 'nested_random30_to60' and cd['post_halos_sha256'] != first[2]['post_halos_sha256']:
            raise ValueError('random-density control changed the post halo positions')
        rows_by_cut = []
        for cut, count in zip(CUTS, COUNTS):
            selected = ix[np.max(ref['B000_k'][ix], axis=1) <= cut + 1e-12]
            scaled = (control['B_post'][selected] - first[0]['B_post'][selected]) / derived['B_post_single_box_std'][selected]
            rows_by_cut.append({'kmax_B': cut, 'bins': count, 'max_abs_delta_over_single_box_std': float(np.max(np.abs(scaled)))})
        controls.append({'control': label, 'paired_realization_count': 1, 'by_cut': rows_by_cut,
                         'coherent_mean_bias_or_convergence_certified': False,
                         'control_DONE_sha256': sha(directory / 'DONE.json')})
    out.mkdir()
    matrix_path = out / 'PILOT_MEASUREMENTS.npz'
    np.savez_compressed(matrix_path, **matrices, **derived, realizations=np.arange(20),
                        B_indices=ix, B_k=ref['B000_k'], P_k=p_k, P_edges=p_edges,
                        N2_data_self=[row['N2_data_self'] for row in rows], N3_data_self=[row['N3_data_self'] for row in rows])
    report = {'status': 'FID20_PILOT_MEASUREMENTS_VERIFIED_NOT_MAIN_COVARIANCE',
              'created_UTC': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
              'CPU_affinity': sorted(os.sched_getaffinity(0)), 'cosmology': 'fiducial', 'pilot_count': 20,
              'realizations': list(range(20)), 'rows': rows, 'P_bins': 12, 'B_registry_bins': 120,
              'B_fit_bins_max': 44, 'kmax_B': CUTS, 'cut_counts': COUNTS, 'kmax_P': .08,
              'fixed_reconstruction': first[1]['fixed_reconstruction'], 'controls': controls,
              'control_scale': 'pointwise single-box std from these20 pilots; not full covariance whitening',
              'single_box_dispersion_ddof': 1, 'main_covariance_created': False, 'covariance_of_mean_created': False,
              'main_covariance_required_count': 500, 'main_volume_Gpc_h3': 1.,
              'physical_p_calibrated': False, 'main_MCMC_run': False, 'scientific_kmax_certified': False,
              'matrix_sha256': sha(matrix_path), 'queue_contract_sha256': sha(queue / 'CONTRACT.json'),
              'source_sha256': sha(Path(__file__)), 'reference_sha256': sha(reference)}
    with (out / 'AUDIT.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__': main()
