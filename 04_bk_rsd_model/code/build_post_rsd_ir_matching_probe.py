"""模块大纲：全44原B分箱的Sigma=0信号对照与hard-tree matching候选。

只复用原IR/FFT、冻结重构与已独立验收的no-IR系数。两网格逐bin检查，
不选择production matching/零腿/noise策略，不修改原MCMC或数据。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import resource
import signal
import socket
import time
import numpy as np
from rsd_b000_contract import PART, REFERENCE, F_GROWTH, digest, reference, selected
from post_rsd_ir_inputs import ROOT, ALPHA, SPLIT, SPLIT_META, FrozenIRSpectra, frozen_reconstruction
from post_rsd_ir_tree_matching import identity_damping_context, tree_matched_signal
from post_rsd_ir_projection import IRFFTProjection
from post_rsd_fft_projection import integrate_post_design
from post_rsd_noir_polynomial import evaluate_noir_design

OUT = ROOT/'ir_hard_matching_all44_v1'
BT = ROOT/'noir_paware_templates'
B1 = 2.2280546353635335
PH = 1.31


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def scaled_error(a, b):
    powers = np.array([1., 500., 250000.])[None, :]
    scale = np.maximum(1., np.max(np.abs(b)*powers, axis=1, keepdims=True))
    return float(np.max(np.abs(a-b)*powers/scale))


def prepare():
    spectra = FrozenIRSpectra(); ref = reference(); indices = selected(.20).tolist()
    template_contract = read(BT/'CONTRACT.json'); csha = digest(BT/'CONTRACT.json')
    audit = read(BT/'INDEPENDENT_COEFFICIENT_AUDIT.json')
    coefficient_path = BT/'DETERMINISTIC_COEFFICIENTS.npz'
    if (indices != template_contract['indices'] or len(indices) != 44
            or audit['status'] != 'ALL44_NOIR_COEFFICIENT_NODE_AND_FINE_GRID_REPLAY_PASS'
            or audit['contract_sha256'] != csha or audit['coefficients_sha256'] != digest(coefficient_path)):
        raise ValueError('actual independently verified all44 no-IR hard-tree inputs required')
    with np.load(coefficient_path, allow_pickle=False) as payload:
        coefficients = np.array(payload['coefficients'])
        np.testing.assert_array_equal(payload['indices'], indices)
    hard = evaluate_noir_design(coefficients, B1, p_halo=PH)
    if hard.shape != (3, 3, 44) or not np.isfinite(hard).all():
        raise ValueError('original native-bias/G-L-Q hard tree registry required')
    window_path = ROOT/'estimator_B_window/B000_WINDOW.npz'
    window_audit_path = ROOT/'estimator_B_window/AUDIT.json'
    if read(window_audit_path)['window_sha256'] != digest(window_path):
        raise ValueError('actual original B window drift')
    with np.load(window_path, allow_pickle=False) as payload:
        pair_counts = np.array(payload['measured_pair_counts'])
    protected = read(ROOT/'compute8_measurement_series_v3/CONTRACT.json')['immutable_sources']
    names = ('post_rsd_ir_tree_matching.py','post_rsd_ir_projection.py','post_rsd_ir_harmonics.py',
        'post_rsd_scipy_harmonic_compat.py','post_rsd_ir_inputs.py','post_rsd_fft_projection.py',
        'post_rsd_fixedrec.py','post_rsd_noir_polynomial.py','rsd_b000_fft_projection.py',
        'rsd_b000_contract.py','rsd_bonly_common.py')
    files = [Path(__file__), REFERENCE, ALPHA, SPLIT, SPLIT_META, coefficient_path,
        BT/'CONTRACT.json', BT/'INDEPENDENT_COEFFICIENT_AUDIT.json', window_path, window_audit_path,
        Path(spectra.provenance['physical_table'])]+[PART/'code'/name for name in names]
    frozen = {**protected, **{str(path): digest(path) for path in files}}
    for path, expected in protected.items():
        if digest(path) != expected: raise ValueError('active measurement source drift: '+path)
    contract = dict(scope='all44 original B bins Sigma-zero matching at one numerical physical-bias probe',
        indices=indices, meshes=template_contract['meshes'], probe_b1=B1, probe_p_halo=PH,
        f_growth=F_GROWTH, Sigma_forced_zero_not_measured=True,
        reconstruction_still_original_and_not_switched_off=True,
        full_hard_tree_retained=True, higher_order_wiggle_remainder_resummed=False,
        physical_p_calibrated=False, STO_noise_validated=False, production_matching_selected=False,
        MCMC_released=False, scaled_tolerance=1e-9, fNL_scaling=[1.,500.,250000.],
        denominator='original full two-shell mode count', closing_leg_cut=False,
        immutable_sources=frozen, spectrum=spectra.provenance)
    OUT.mkdir(exist_ok=True); (OUT/'bins').mkdir(exist_ok=True); (OUT/'workers').mkdir(exist_ok=True)
    path = OUT/'CONTRACT.json'
    if path.exists():
        if read(path) != contract: raise ValueError('matching contract drift; preserve old checkpoints')
    else: save(path, contract)
    return spectra, ref, indices, hard, pair_counts, contract, digest(path)


def completed(index, csha):
    path = OUT/'bins'/f'bin{index:03d}.json'
    if not path.exists(): return False
    row = read(path)
    if row['contract_sha256'] != csha or row['double_grid_scaled_error'] >= 1e-9:
        raise ValueError('matching checkpoint contract or numerical gate failed')
    return True


def run_bin(index):
    # Slurm预警不杀正在计算的bin；父调度器自然排空，TIMEOUT仍由scheduler管理。
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    if len(os.sched_getaffinity(0)) != 1: raise ValueError('one logical CPU per projection required')
    spectra, ref, indices, hard, pair_counts, contract, csha = prepare()
    if index not in indices: raise ValueError('bin outside original all44 registry')
    if completed(index, csha): print('REUSE verified matching bin', index, flush=True); return
    start = time.monotonic(); slot = indices.index(index); exact = hard[..., slot]
    rec = frozen_reconstruction(256)
    harmonic = identity_damping_context(b1=B1, p_halo=PH, f_growth=F_GROWTH, reconstruction=rec)
    status = dict(state='RUNNING', pid=os.getpid(), host=socket.gethostname(),
        starttime_ticks=(Path('/proc/self/stat')).read_text().rsplit(')',1)[1].split()[19],
        CPU_affinity=sorted(os.sched_getaffinity(0)), bin_index=index, contract_sha256=csha,
        started_UTC=datetime.now(timezone.utc).isoformat())
    status_path = OUT/'workers'/f'bin{index:03d}_STATUS.json'; save(status_path, status)
    grids = []
    for mesh in contract['meshes'][str(index)]:
        for path, expected in contract['immutable_sources'].items():
            if digest(path) != expected: raise ValueError('matching source changed during projection')
        tick = time.monotonic()
        projection = IRFFTProjection(mesh, ref['edges'], ref['shell_pairs'][index], B1, F_GROWTH,
            p_halo=PH, reconstruction=rec, power_nw=spectra.nowiggle, harmonics=harmonic)
        if int(projection.denominator) != int(pair_counts[index]):
            raise ValueError('original full-shell denominator changed')
        strict = integrate_post_design(projection)
        matched_zero = tree_matched_signal(exact, strict, strict)
        np.testing.assert_array_equal(matched_zero, exact)
        grids.append(dict(mesh=mesh, strict_undamped=strict.tolist(), hard_remainder=(exact-strict).tolist(),
            original_hard_tree=exact.tolist(), counts=projection.counts, denominator=int(projection.denominator),
            zero_damping_matched_error=scaled_error(matched_zero, exact),
            strict_source_matching_remainder_scaled=scaled_error(strict, exact),
            cache_peak_bytes=projection.cache_peak_bytes, elapsed_seconds=time.monotonic()-tick))
        del projection; gc.collect()
    if grids[0]['counts'] != grids[1]['counts']: raise ValueError('two-grid mode counts differ')
    difference = scaled_error(np.array(grids[0]['strict_undamped']),np.array(grids[1]['strict_undamped']))
    if difference >= 1e-9: raise ValueError('original double-grid tolerance failed, no relaxation')
    row = dict(index=index, contract_sha256=csha, grids=grids, double_grid_scaled_error=difference,
        actual_CPU_affinity=sorted(os.sched_getaffinity(0)), elapsed_seconds=time.monotonic()-start,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        production_matching_selected=False, physical_p_calibrated=False, STO_noise_closed=False, MCMC_released=False)
    save(OUT/'bins'/f'bin{index:03d}.json', row)
    status.update(state='COMPLETE', elapsed_seconds=row['elapsed_seconds'], peak_RSS_MiB=row['peak_RSS_MiB'])
    save(status_path, status); print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true'); parser.add_argument('--bin-index', type=int)
    args = parser.parse_args()
    if args.plan_only:
        _, _, indices, _, _, _, csha = prepare()
        print(json.dumps(dict(status='ALL44_MATCHING_PLAN_VALIDATED_NOT_EXECUTED', indices=indices, contract_sha256=csha)))
    elif args.bin_index is not None: run_bin(args.bin_index)
    else: raise ValueError('explicit --plan-only or --bin-index required')
