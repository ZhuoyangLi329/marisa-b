"""模块大纲：原12-bin P窗口/真实谱的IR信号验证与9/17/33节点检查。

复用B的原1025/48/96/lmax12位移与tagged源；独立显式Z1/t导数与逐mode求和。
记录零阻尼strict-O(Pw)相对精确tree的matching余项，不改原no-IR模型。
数值p=1.31仅为probe；不校准STO噪声或physical p，不放行P+B/IR MCMC。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import socket
import time
import traceback
import numpy as np
from rsd_b000_contract import PART, F_GROWTH, digest
from post_rsd_ir_p0_projection import ir_p0_modes, project_p0_ir_window
from post_rsd_ir_inputs import FrozenIRSpectra, frozen_reconstruction, ROOT, SPLIT_META, ALPHA
from post_rsd_ir_covariance_basis import HarmonicCovarianceBasis
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_fixedrec import pbs_bias
from post_rsd_ir_signal_interpolation import IRSignalInterpolation, bias_nodes

OUT = ROOT/'ir_p0_signal_actual_v1'
PROBES = (.5, .71, 1.31, 2.2280546353635335, 3.1, 3.73, 4.)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def error(a, b):
    powers = np.array([1., 500., 250000.])[:, None]
    scale = np.maximum(1., np.max(np.abs(b)*powers, axis=0, keepdims=True))
    return float(np.max(np.abs(a-b)*powers/scale))


def independent_modes(v, p, pn, m, b, ph, damping):
    r = b+F_GROWTH*v[:, 2]**2/np.sum(v*v, axis=1); phi = 2*1.686*(b-ph)
    mn = m*np.sqrt(pn/p); base = np.array([pn*r*r, 2*pn*r*phi/mn, pn*(phi/mn)**2])
    eta = (p-pn)/(2*pn); w0 = 2*eta*base[0]; w1 = eta*base[1]; d0, d1, d2 = damping.T
    return np.array([base[0]+d0*w0, base[1]+d0*(w1+d1*w0), base[2]+d0*(d1*w1+d2*w0)])


def main():
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {13}:
        raise ValueError('short single-CPU13 actual-window audit required')
    if OUT.exists():
        raise ValueError('preserve existing actual P IR audit, no overwrite')
    window_path = ROOT/'calibration_holdout_p0/P0_WINDOW.npz'
    window_audit_path = ROOT/'calibration_holdout_p0/POST_P0_WINDOW_AUDIT.json'
    window_audit = read(window_audit_path)
    if window_audit['status'] != 'PASS' or window_audit['window_sha256'] != digest(window_path):
        raise ValueError('actual production P window audit required')
    with np.load(window_path, allow_pickle=False) as payload:
        window = {key: np.array(payload[key]) for key in payload.files}
    v = window['vectors_integer']*(2*np.pi/1000.); k = np.linalg.norm(v, axis=1)
    labels = window['labels']; weights = window['weights']; counts = window['nmodes']
    spectra = FrozenIRSpectra(); p, m = spectra.physical(k); pn = spectra.nowiggle(k)
    rec = frozen_reconstruction(256); ph = 1.31
    old = read(ROOT/'compute8_measurement_series_v3/CONTRACT.json')['immutable_sources']
    names = ('post_rsd_ir_p0_projection.py', 'post_rsd_ir_projection.py', 'post_rsd_ir_harmonics.py',
        'post_rsd_displacement.py', 'post_rsd_ir_covariance_basis.py', 'post_rsd_scipy_harmonic_compat.py',
        'post_rsd_ir_inputs.py', 'post_rsd_ir_signal_interpolation.py', 'post_rsd_fft_projection.py',
        'post_rsd_fixedrec.py', 'rsd_b000_fft_projection.py', 'rsd_b000_contract.py', 'rsd_bonly_common.py')
    files = [Path(__file__), window_path, window_audit_path, SPLIT_META, ALPHA,
        Path(spectra.provenance['physical_table']), Path(spectra.provenance['no_wiggle_split'])]
    files += [PART/'code'/name for name in names]
    frozen = {**old, **{str(path): digest(path) for path in files}}
    for path, expected in old.items():
        if digest(path) != expected:
            raise ValueError('active production source drift')
    OUT.mkdir(); start = time.monotonic()
    contract = dict(scope='actual all12 P IR signal at a numerical p probe; strict O(Pw), fnl<=2',
        immutable_sources=frozen, spectrum=spectra.provenance, fixed_probe_p=ph, physical_p_calibrated=False,
        b1_domain=[.5, 4.], fNL_domain=[-500., 500.], probe_b1=list(PROBES), node_counts=[9, 17, 33],
        harmonic_rule=dict(nq=1025, n_mu=48, n_phi=96, lmax=12, r_bao=110.),
        original_P_bins=12, stored_nonzero_modes=len(v), weighted_mode_count=float(counts.sum()),
        original_window_denominator_retained=True, scaled_tolerance=1e-9,
        raw_STO_noise_calibrated=False, B_noise_closed=False, MCMC_released=False)
    save(OUT/'CONTRACT.json', contract); csha = digest(OUT/'CONTRACT.json')
    state = dict(state='RUNNING', host=socket.gethostname(), pid=os.getpid(),
        starttime_ticks=(Path('/proc/self/stat')).read_text().rsplit(')', 1)[1].split()[19],
        CPU_affinity=[13], started_UTC=datetime.now(timezone.utc).isoformat(), contract_sha256=csha)

    def update(**values):
        state.update(values, elapsed_seconds=time.monotonic()-start,
            updated_UTC=datetime.now(timezone.utc).isoformat()); save(OUT/'STATUS.json', state)

    def guard():
        for path, expected in frozen.items():
            if digest(path) != expected:
                raise ValueError('original source changed during P IR audit: '+path)
        if digest(OUT/'CONTRACT.json') != csha:
            raise ValueError('P IR audit contract changed')

    update(stage='covariance_basis'); guard()
    long = spectra.long_modes(1025, n_mu=48, n_phi=96)
    basis = HarmonicCovarianceBasis.from_long_modes(long, f_growth=F_GROWTH, los=(0, 0, 1),
        reconstruction=rec, lmax=12, r_bao=110.)
    direct = {}; rows = []
    def project(b):
        b = float(b)
        if b not in direct:
            guard(); update(stage='original_window_IR_projection', b1=b)
            h = basis.harmonics_at(b1=b, bphi=pbs_bias(b, p_halo=ph).bphi)
            direct[b] = project_p0_ir_window(window, p, pn, m, b, F_GROWTH,
                p_halo=ph, reconstruction=rec, harmonics=h)
        return direct[b]
    for b in PROBES:
        h = basis.harmonics_at(b1=b, bphi=pbs_bias(b, p_halo=ph).bphi)
        modes = independent_modes(v, p, pn, m, b, ph, h.damping(v))
        expected = np.array([[sum(weights[j]*value[j] for j in range(len(v)) if labels[j] == bin_index)/counts[bin_index]
                               for bin_index in range(12)] for value in modes])
        numerical_error = error(project(b), expected)
        if numerical_error >= 1e-9:
            raise ValueError('direct original-window P IR oracle failed; no relaxed tolerance')
        no_damp = np.zeros((len(v), 3)); no_damp[:, 0] = 1.
        strict = ir_p0_modes(v, p, pn, m, b, F_GROWTH, p_halo=ph, reconstruction=rec, damping=no_damp)
        r = b+F_GROWTH*v[:, 2]**2/np.sum(v*v, axis=1); phi = pbs_bias(b, p_halo=ph).bphi
        exact = np.array([p*r*r, 2*p*r*phi/m, p*(phi/m)**2])
        no_wiggle = ir_p0_modes(v, p, p, m, b, F_GROWTH, p_halo=ph, reconstruction=rec, damping=h.damping(v))
        if error(no_wiggle, exact) >= 1e-9:
            raise ValueError('no-wiggle exact original linear P source regression failed')
        rows.append(dict(b1=b, independent_original_window_error=numerical_error,
            no_wiggle_exact_tree_error=error(no_wiggle, exact),
            zero_damping_strict_wiggle_matching_remainder_scaled=error(strict, exact)))
    cache_rows = []
    for b in (.5, 2.2280546353635335, 4.):
        h = HarmonicDisplacement.from_long_modes(long, b1=b, bphi=pbs_bias(b, p_halo=ph).bphi,
            f_growth=F_GROWTH, los=(0, 0, 1), reconstruction=rec, lmax=12, r_bao=110.)
        uncached = project_p0_ir_window(window, p, pn, m, b, F_GROWTH,
            p_halo=ph, reconstruction=rec, harmonics=h)
        difference = error(project(b), uncached); cache_rows.append(dict(b1=b, scaled_error=difference))
        if difference >= 2e-10:
            raise ValueError('P IR full projection covariance-cache replay failed')
    candidates = []
    for n in (9, 17, 33):
        designs = np.zeros((n, 3, 3, 12))
        for index, b in enumerate(bias_nodes(n)):
            designs[index, 0] = project(b)
        model = IRSignalInterpolation.from_nodes(designs, p_halo=ph)
        errors = [error(model.designs_at(b)[0, 0], project(b)) for b in PROBES]
        path = OUT/f'candidate{n:02d}.npz'
        np.savez_compressed(path, coefficients=model.coefficients, direct_node_designs=designs,
            node_b1=bias_nodes(n), P_edges=window['edges'], P_k=window['k'], P_nmodes=counts)
        candidates.append(dict(nodes=n, max_scaled_error=max(errors), passed=max(errors) < 1e-9,
            payload_sha256=digest(path)))
    if not any(row['passed'] for row in candidates):
        raise ValueError('all original P IR interpolation candidates failed; preserve candidates')
    guard()
    result = dict(status='ACTUAL_ALL12_P0_IR_SIGNAL_WINDOW_AND_BIAS_PROBES_VERIFIED_NOT_MCMC_RELEASE',
        created_UTC=datetime.now(timezone.utc).isoformat(), contract_sha256=csha,
        original_P_bins=12, stored_nonzero_modes=len(v), weighted_mode_count=float(counts.sum()),
        direct_window_rows=rows, independent_uncached_harmonic_projection_rows=cache_rows,
        node_candidates=candidates, direct_bias_parameter_evaluations=len(direct),
        strict_wiggle_matching_remainder_quantified_not_removed=True,
        physical_p_calibrated=False, raw_STO_noise_calibrated=False, full_PB_IR_model_validated=False,
        post_C_single500_whitened_error_checked=False, MCMC_released=False,
        elapsed_seconds=time.monotonic()-start,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
    save(OUT/'AUDIT.json', result); update(state='COMPLETE', stage='complete')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status_path = OUT/'STATUS.json'
        if status_path.exists():
            status = read(status_path)
            status.update(state='FAILED_PRESERVED', error=traceback.format_exc(),
                updated_UTC=datetime.now(timezone.utc).isoformat())
            save(status_path, status)
        raise
