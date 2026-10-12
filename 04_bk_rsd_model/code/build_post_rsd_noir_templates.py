"""模块大纲：原44-bin FFT源的no-IR物理p张量模板；独立参数点/双网格核验。

逐节点checkpoint，保留旧源和拟合结果；p校准后才收缩p轴。这里只生成确定性
G/L/Q，不将未闭合随机项默认为零，也不放行似然或MCMC。支持独立session运行。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
from rsd_b000_contract import PART, TABLE, REFERENCE, F_GROWTH, digest, reference, selected, CUTS
from post_rsd_ir_inputs import ROOT, ALPHA, ALPHA_SHA, frozen_reconstruction
from audit_post_rsd_ir_registry import meshes_for_pair
from post_rsd_fft_projection import PostFFTProjection, integrate_post_design
from post_rsd_noir_polynomial import (B_NODES, P_NODES, B_CENTER, B_RADIUS, P_CENTER, P_RADIUS,
                                     fit_noir_design, evaluate_noir_design)

OUT = ROOT / 'noir_paware_templates'


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detach', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--run-label', default='full44')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('one logical CPU required')
    if not args.run_label or any(not (c.isalnum() or c in '_-') for c in args.run_label):
        raise ValueError('safe run label required')
    if args.detach:
        log = ROOT / ('noir_paware_templates_' + args.run_label + '.log')
        launch = ROOT / ('noir_paware_templates_' + args.run_label + '.launch.json')
        if log.exists() or launch.exists(): raise RuntimeError('prior launch must be inspected, not overwritten')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())] + [v for v in sys.argv[1:] if v != '--detach']
        with log.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        record = {'host': socket.gethostname(), 'pid': proc.pid, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
                  'new_session': True, 'log': str(log), 'command': command,
                  'launched_UTC': datetime.now(timezone.utc).isoformat()}
        with launch.open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True); return
    if (OUT / 'COMPLETION_AUDIT.json').exists(): raise RuntimeError('completed templates exist; inspect without refreezing')
    previous_path = OUT / 'STATUS.json'
    if previous_path.exists():
        old = json.loads(previous_path.read_text())
        if not args.resume: raise RuntimeError('prior execution exists; explicit recovery and process check required')
        if old['host'] != socket.gethostname(): raise RuntimeError('recover on the prior host to check its PID')
        try:
            os.kill(old['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('prior template process still exists; do not restart')
    for name, expected in (('SOURCE_AUDIT.json', 'SOURCE_TESTS_PASS_PRODUCTION_NOT_RELEASED'),
        ('estimator_B_window/AUDIT.json', 'PRODUCTION_B000_WINDOW_MATCHES_EXISTING_FFT_ALL44_BINS')):
        if json.loads((ROOT / name).read_text())['status'] != expected: raise ValueError('prerequisite failed: ' + name)
    tests = unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_noir_polynomial')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful() or result.testsRun != 3: raise RuntimeError('polynomial regressions failed')
    recon = frozen_reconstruction(256); ref = reference(); indices = selected(.20).tolist()
    window_path = ROOT / 'estimator_B_window/B000_WINDOW.npz'
    window_audit = json.loads((ROOT / 'estimator_B_window/AUDIT.json').read_text())
    if digest(window_path) != window_audit['window_sha256']: raise ValueError('actual B window changed')
    with np.load(window_path, allow_pickle=False) as z:
        actual_counts = z['measured_pair_counts'].copy()
        np.testing.assert_array_equal(z['fit_indices'], indices)
    paths = [Path(__file__), TABLE, REFERENCE, ALPHA, window_path, ROOT / 'estimator_B_window/AUDIT.json',
             ROOT / 'SOURCE_AUDIT.json']
    paths += [PART / 'code' / name for name in ('post_rsd_noir_polynomial.py', 'test_post_rsd_noir_polynomial.py',
        'post_rsd_fft_projection.py', 'post_rsd_fixedrec.py', 'post_rsd_ir_inputs.py', 'rsd_b000_fft_projection.py',
        'rsd_b000_contract.py', 'rsd_bonly_common.py', 'audit_post_rsd_ir_registry.py',
        'vendor/quijote_strict_triangle_fft_estimator.py')]
    contract = {'scope': 'deterministic fixed-catalog no-IR G/L/Q; no stochastic closure or likelihood release',
        'indices': indices, 'cuts': {str(c): selected(c).tolist() for c in CUTS},
        'axes': ['b1_chebyshev5', 'physical_p_chebyshev3', 'one_b2_bK2', 'G_L_Q', 'observable_bin'],
        'b1_nodes': B_NODES.tolist(), 'physical_p_nodes': P_NODES.tolist(),
        'coordinate_centers_and_scales': {'b1': [B_CENTER, B_RADIUS], 'p': [P_CENTER, P_RADIUS]},
        'physical_p_is_not_fitted_or_calibrated_here': True, 'numerical_nodes_are_not_priors': True,
        'new_MCMC_parameter_added': False, 'f_growth': F_GROWTH, 'IR': False,
        'b_rec': recon.bias_recon, 'p_rec': recon.p_recon, 'fNL_rec': recon.fnl_recon,
        'recon_mesh': 256, 'R': recon.smoothing_radius, 'CIC_window_power': recon.cic_window_power,
        'alpha_sha256': ALPHA_SHA, 'zero_leg_source': 0, 'closing_leg_cut': False,
        'denominator': 'full actual production two-shell mode counts',
        'meshes': {str(i): meshes_for_pair(ref['edges'], ref['shell_pairs'][i]) for i in indices},
        'off_node_probes': [[1.83641, 1.287], [3.613, 1.513]],
        'fNL_range_coefficient_scaling': [1., 500., 250000.],
        'combined_nonnode_fine_grid_scaled_tolerance': 2e-10,
        'covariance_whitening_required_after_post_covariance': True,
        'immutable_sources': {str(p): digest(p) for p in paths}}
    contract = json.loads(json.dumps(contract))
    OUT.mkdir(exist_ok=True); (OUT / 'nodes').mkdir(exist_ok=True); (OUT / 'bins').mkdir(exist_ok=True)
    cp = OUT / 'CONTRACT.json'
    if cp.exists():
        if json.loads(cp.read_text()) != contract: raise ValueError('template contract changed; preserve previous work')
    else:
        save(cp, contract)
    csha = digest(cp); start = time.time()
    state = {'state': 'RUNNING', 'host': socket.gethostname(), 'pid': os.getpid(),
             'CPU_affinity': sorted(os.sched_getaffinity(0)), 'contract_sha256': csha,
             'started_UTC': datetime.now(timezone.utc).isoformat(), 'completed_bins': [], 'required_bins': indices}
    def stage(name, **values):
        state.update(stage=name, **values, updated_UTC=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.time()-start)
        save(OUT / 'STATUS.json', state); print(json.dumps(state), flush=True)
    def guard():
        if digest(cp) != csha: raise ValueError('contract drift during build')
        for path, expected in contract['immutable_sources'].items():
            if digest(path) != expected: raise ValueError('source drift during build: ' + path)
    def project(index, mesh, b, p):
        field = PostFFTProjection(mesh, ref['edges'], ref['shell_pairs'][index], b, F_GROWTH,
                                  p_halo=p, reconstruction=recon)
        if int(field.denominator) != int(actual_counts[index]): raise ValueError('theory/data denominator mismatch')
        design = integrate_post_design(field)
        if design.shape != (3, 3) or not np.isfinite(design).all(): raise ValueError('invalid source design')
        peak = int(field.cache_peak_bytes); del field; gc.collect()
        return design, peak
    summaries = []
    try:
        stage('begin'); guard()
        for index in indices:
            guard(); target = OUT / 'bins' / f'bin{index:03d}.npz'; jp = target.with_suffix('.json')
            if target.exists() or jp.exists():
                if not (target.exists() and jp.exists()): raise ValueError('partial bin artifact requires inspection')
                summary = json.loads(jp.read_text())
                if summary['contract_sha256'] != csha or summary['coefficients_sha256'] != digest(target):
                    raise ValueError('bin checkpoint drift')
            else:
                coarse, fine = contract['meshes'][str(index)]
                nodes = np.empty((5, 3, 3, 3)); peak = 0
                for ib, b in enumerate(B_NODES):
                    for ip, p in enumerate(P_NODES):
                        guard(); path = OUT / 'nodes' / f'bin{index:03d}_b{ib}_p{ip}.json'
                        if path.exists():
                            node = json.loads(path.read_text())
                            if (node['contract_sha256'] != csha or node['b1'] != b or node['p_halo'] != p):
                                raise ValueError('node checkpoint drift')
                            design = np.array(node['design']); used = node['cache_peak_bytes']
                        else:
                            stage('project_node', bin_index=index, b1_node=ib, p_node=ip)
                            tick = time.time(); design, used = project(index, coarse, float(b), float(p))
                            node = {'index': index, 'b1': float(b), 'p_halo': float(p), 'mesh': coarse,
                                    'design': design.tolist(), 'cache_peak_bytes': used,
                                    'elapsed_seconds': time.time()-tick, 'contract_sha256': csha}
                            save(path, node)
                        nodes[ib, ip] = design; peak = max(peak, used)
                coefficients = fit_noir_design(nodes); probes = []
                for b, p in contract['off_node_probes']:
                    guard(); stage('nonnode_fine_grid_probe', bin_index=index, probe_b1=b, probe_p=p)
                    direct, used = project(index, fine, b, p); peak = max(peak, used)
                    predicted = evaluate_noir_design(coefficients, b, p_halo=p)
                    powers = np.array(contract['fNL_range_coefficient_scaling'])[None, :]
                    scale = np.maximum(1., np.max(np.abs(direct) * powers, axis=1, keepdims=True))
                    error = float(np.max(np.abs(predicted-direct) * powers / scale))
                    probes.append({'b1': b, 'p_halo': p, 'mesh': fine, 'direct': direct.tolist(),
                                   'predicted': predicted.tolist(), 'scaled_error': error})
                    if error > contract['combined_nonnode_fine_grid_scaled_tolerance']:
                        raise ValueError('nonnode/fine-grid gate failed; do not relax or drop the bin')
                np.savez_compressed(target, coefficients=coefficients, node_designs=nodes,
                    shell_pair=ref['shell_pairs'][index], bin_index=index, denominator=actual_counts[index],
                    b1_nodes=B_NODES, physical_p_nodes=P_NODES, contract_sha256=np.array(csha))
                summary = {'index': index, 'contract_sha256': csha, 'coefficients_sha256': digest(target),
                           'cache_peak_bytes': peak, 'probes': probes,
                           'max_probe_scaled_error': max(r['scaled_error'] for r in probes)}
                save(jp, summary)
            summaries.append(summary); state['completed_bins'].append(index); stage('bin_complete', bin_index=index)
        guard(); combined = []
        for index in indices:
            with np.load(OUT / 'bins' / f'bin{index:03d}.npz', allow_pickle=False) as z:
                combined.append(np.array(z['coefficients']))
        destination = OUT / 'DETERMINISTIC_COEFFICIENTS.npz'
        if destination.exists(): raise RuntimeError('aggregate exists; inspect interrupted finalization')
        np.savez_compressed(destination, coefficients=np.stack(combined, axis=-1), indices=indices,
                            b1_nodes=B_NODES, physical_p_nodes=P_NODES, contract_sha256=np.array(csha))
        report = {'status': 'ALL44_DETERMINISTIC_NOIR_P_AWARE_TEMPLATES_COMPLETE_NOT_LIKELIHOOD',
                  'created_UTC': datetime.now(timezone.utc).isoformat(), 'bins': 44,
                  'contract_sha256': csha, 'coefficients_sha256': digest(destination),
                  'rows': summaries, 'max_probe_scaled_error': max(r['max_probe_scaled_error'] for r in summaries),
                  'physical_p_calibrated': False, 'stochastic_model_closed': False,
                  'post_covariance_whitened_error_verified': False, 'MCMC_released': False,
                  'numeric_affinity': sorted(os.sched_getaffinity(0)), 'elapsed_seconds': time.time()-start}
        save(OUT / 'COMPLETION_AUDIT.json', report); state['state'] = 'COMPLETE'; stage('complete')
    except Exception:
        state.update(state='FAILED', error=traceback.format_exc()); stage('failed'); raise


if __name__ == '__main__': main()
