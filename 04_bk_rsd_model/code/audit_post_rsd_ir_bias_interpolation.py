"""模块大纲：真实最低bin、完整b1域的IR插值/独立节点/双网格实测。

复用原谱、冻结重构核和唯一IR/FFT源，不改旧五节点no-IR代码。
保留9/17/33节点各候选失败，原1e-9门不变；此单bin数值探针
不能认证全44-bin、physical p、STO噪声、whitened误差或MCMC。
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
from rsd_b000_contract import PART, REFERENCE, reference, selected, F_GROWTH, digest
from post_rsd_scipy_harmonic_compat import install_legacy_sph_harm
install_legacy_sph_harm()
from post_rsd_ir_covariance_basis import HarmonicCovarianceBasis
from post_rsd_ir_harmonics import HarmonicDisplacement
from post_rsd_ir_inputs import FrozenIRSpectra, frozen_reconstruction, ROOT
from post_rsd_ir_grid import FastIRFFTProjection
from post_rsd_fft_projection import integrate_post_design
from post_rsd_fixedrec import pbs_bias
from post_rsd_ir_signal_interpolation import bias_nodes, IRSignalInterpolation
from audit_post_rsd_ir_registry import meshes_for_pair

OUT = ROOT/'ir_bias_interpolation_probe_v1'
NODE_COUNTS = (9, 17, 33)
PROBES = (.5, .71, 1.09, 1.47, 2.02, 2.2280546353635335, 2.83, 3.29, 3.77, 4.)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def scaled_error(actual, expected):
    powers = np.array([1., 500., 250000.])[None, :, None]
    denominator = np.maximum(1., np.max(np.abs(expected)*powers, axis=1, keepdims=True))
    return float(np.max(np.abs(actual-expected)*powers/denominator))


def main():
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {13}:
        raise ValueError('one known free login26 CPU13 required for this short low-bin probe')
    if OUT.exists():
        raise ValueError('preserve existing probe and inspect its real process, never overwrite')
    # 旧控制源按现有合同逐项只读验证，不改冻结source/统计或原队列状态。
    owner_contract = ROOT/'compute8_measurement_series_v3/CONTRACT.json'
    old_sources = read(owner_contract)['immutable_sources']
    for path, expected in old_sources.items():
        if digest(path) != expected:
            raise ValueError('active measurement source drift: '+path)
    specs = FrozenIRSpectra(); reconstruction = frozen_reconstruction(256)
    ref = reference(); index = int(selected(.20)[0]); pair = ref['shell_pairs'][index]
    meshes = meshes_for_pair(ref['edges'], pair)
    if index != 1 or meshes != (32, 48):
        raise ValueError('exact original low-bin geometry required, not an easier invented bin')
    physical_p = 1.31  # 纯数值探针；绝不是p_rec或尚未完成的physical-p校准。
    paths = [Path(__file__), REFERENCE, PART/'work/build/libmarisa_tree_rsd_fixedrec.so']
    paths += [PART/'code'/name for name in (
        'post_rsd_ir_signal_interpolation.py', 'post_rsd_ir_covariance_basis.py',
        'post_rsd_scipy_harmonic_compat.py', 'post_rsd_ir_harmonics.py', 'post_rsd_displacement.py',
        'post_rsd_ir_inputs.py', 'post_rsd_ir_grid.py', 'post_rsd_ir_projection.py',
        'post_rsd_fft_projection.py', 'post_rsd_fixedrec.py', 'rsd_b000_fft_projection.py',
        'rsd_b000_contract.py', 'rsd_bonly_common.py', 'audit_post_rsd_ir_registry.py')]
    contract = dict(scope='actual low-bin IR signal interpolation, unchanged whole b1 prior',
        bin_index=index, shell_pair=pair.tolist(), meshes=list(meshes), b1_domain=[.5, 4.],
        node_counts=list(NODE_COUNTS), probes=list(PROBES), p_fixed_numerical_probe=physical_p,
        physical_p_calibrated=False, fNL_domain=[-500., 500.], fNL_degree=2,
        harmonics=dict(nq=1025, n_mu=48, n_phi=96, lmax=12, r_bao=110.),
        interpolation_scaled_tolerance=1e-9, double_grid_scaled_tolerance=1e-9,
        covariance_cache_scaled_tolerance=2e-10, full_shell_denominator=True,
        any_source_zero_leg_excluded=True, closing_third_leg_cut=False,
        immutable_sources={str(path): digest(path) for path in paths},
        active_measurement_sources=old_sources, spectrum=specs.provenance,
        actual_cpu_limit=1, other_production_jobs_not_modified=True,
        stochastic_closure_validated=False, MCMC_released=False, goal_completed=False)
    OUT.mkdir(); (OUT/'designs').mkdir(); save(OUT/'CONTRACT.json', contract)
    contract_sha = digest(OUT/'CONTRACT.json')
    started = time.monotonic()
    state = dict(state='RUNNING', pid=os.getpid(), host=socket.gethostname(),
        starttime_ticks=(Path('/proc/self/stat')).read_text().rsplit(')', 1)[1].split()[19],
        affinity=sorted(os.sched_getaffinity(0)), started_UTC=datetime.now(timezone.utc).isoformat(),
        contract_sha256=contract_sha, completed_projections=0)

    def update(**values):
        state.update(values, updated_UTC=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        save(OUT/'STATUS.json', state)
        print(json.dumps(state), flush=True)

    def guard():
        if digest(OUT/'CONTRACT.json') != contract_sha:
            raise ValueError('probe contract changed')
        for path, expected in {**old_sources, **contract['immutable_sources']}.items():
            if digest(path) != expected:
                raise ValueError('immutable source changed: '+path)

    try:
        update(stage='exact_covariance_basis'); guard()
        long_modes = specs.long_modes(1025, n_mu=48, n_phi=96)
        basis = HarmonicCovarianceBasis.from_long_modes(long_modes, f_growth=F_GROWTH,
            los=(0, 0, 1), reconstruction=reconstruction, r_bao=110., lmax=12)
        covariance_rows = []
        for b in (.5, 2.2280546353635335, 4.):
            phi = pbs_bias(b, p_halo=physical_p).bphi
            exact = HarmonicDisplacement.from_long_modes(long_modes, b1=b, bphi=phi,
                f_growth=F_GROWTH, los=(0, 0, 1), reconstruction=reconstruction, r_bao=110., lmax=12)
            cached = basis.coefficients_at(b1=b, bphi=phi)
            scale = np.maximum(1., np.max(np.abs(exact.coefficients)))
            error = float(np.max(np.abs(cached-exact.coefficients))/scale)
            covariance_rows.append(dict(b1=b, scaled_error=error))
            if error >= 2e-10:
                raise ValueError('new original1025/lmax12 covariance-cache comparison failed')
        designs = {}
        def direct(b, mesh):
            key = (float(b), int(mesh))
            if key in designs:
                return designs[key]
            guard(); update(stage='direct_IR_projection', b1=float(b), theory_mesh=int(mesh))
            harmonic = basis.harmonics_at(b1=b, bphi=pbs_bias(b, p_halo=physical_p).bphi)
            projection = FastIRFFTProjection(mesh, ref['edges'], pair, b, F_GROWTH,
                p_halo=physical_p, reconstruction=reconstruction,
                power_nw=specs.nowiggle, harmonics=harmonic)
            design = integrate_post_design(projection)[..., None]
            if design.shape != (3, 3, 1) or not np.isfinite(design).all():
                raise ValueError('actual IR source produced invalid design')
            if int(projection.denominator) != 153174 or list(projection.counts) != [147, 1042]:
                raise ValueError('original full-shell denominator or integer mode counts changed')
            number = len(designs)
            save(OUT/'designs'/f'projection{number:03d}.json', dict(b1=float(b), mesh=int(mesh),
                design=design.tolist(), counts=list(projection.counts), denominator=int(projection.denominator),
                contract_sha256=contract_sha))
            designs[key] = design; del projection; gc.collect()
            update(completed_projections=len(designs))
            return design

        probe_rows = []
        for b in PROBES:
            coarse = direct(b, meshes[0]); fine = direct(b, meshes[1])
            error = scaled_error(coarse, fine)
            probe_rows.append(dict(b1=b, double_grid_scaled_error=error, fine_design=fine.tolist()))
            if error >= 1e-9:
                raise ValueError('original exact double-grid gate failed, no tolerance relaxation')
        candidates = []
        for n in NODE_COUNTS:
            samples = np.array([direct(b, meshes[1]) for b in bias_nodes(n)])
            model = IRSignalInterpolation.from_nodes(samples, p_halo=physical_p)
            probe_errors = [scaled_error(model.designs_at(b)[0], direct(b, meshes[1])) for b in PROBES]
            predictions = model.designs_at(np.asarray(PROBES))
            np.savez_compressed(OUT/f'candidate{n:02d}.npz', coefficients=model.coefficients,
                node_b1=bias_nodes(n), node_designs=samples, probes=np.array(PROBES), predictions=predictions)
            candidates.append(dict(nodes=n, degree=n-1, probe_errors=probe_errors,
                max_scaled_error=max(probe_errors), passed=max(probe_errors) < 1e-9,
                coefficients_sha256=digest(OUT/f'candidate{n:02d}.npz')))
            update(stage='candidate_verified', candidate_nodes=n, candidate_error=max(probe_errors))
        passed = [row['nodes'] for row in candidates if row['passed']]
        if not passed:
            raise ValueError('all predefined candidates failed; preserve actual failures, do not shrink domain')
        guard()
        report = dict(status='ACTUAL_LOW_BIN_WHOLE_B1_DOMAIN_IR_INTERPOLATION_PROBE_PASS',
            created_UTC=datetime.now(timezone.utc).isoformat(), contract_sha256=contract_sha,
            bin_index=index, b1_domain=[.5, 4.], p_fixed_numerical_probe=physical_p,
            node_candidates=candidates, lowest_candidate_passing_this_probe=min(passed),
            direct_double_grid_rows=probe_rows, covariance_cache_rows=covariance_rows,
            actual_full_IR_projections=len(designs), fNL_scaled_tolerance=1e-9,
            full44_parameter_interpolation_certified=False, post_covariance_whitening_done=False,
            physical_p_calibrated=False, stochastic_closure_validated=False, MCMC_released=False,
            elapsed_seconds=time.monotonic()-started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        save(OUT/'AUDIT.json', report); update(state='COMPLETE', stage='complete')
    except Exception:
        update(state='FAILED_PRESERVED', error=traceback.format_exc()); raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--detach']:
        if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {13}:
            raise ValueError('the detached probe must inherit the same single CPU13')
        launch_path = ROOT/'ir_bias_interpolation_probe_v1.launch.json'
        log_path = ROOT/'ir_bias_interpolation_probe_v1.log'
        if OUT.exists() or launch_path.exists() or log_path.exists():
            raise ValueError('existing probe/launch/log must be inspected, not replaced')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log_path.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=proc.pid, host=socket.gethostname(), command=command,
            source_sha256=digest(Path(__file__)), CPU_affinity=[13],
            launched_UTC=datetime.now(timezone.utc).isoformat(), log=str(log_path))
        with launch_path.open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True)
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError('only the original probe or its --detach launcher is supported')
