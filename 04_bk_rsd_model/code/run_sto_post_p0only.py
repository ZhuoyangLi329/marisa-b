"""模块大纲：C500和原physical-p校准通过后先做原39中的三组post-P0。

独立输出namespace保护旧结果，不伪造B噪声门；原P-only MAP及MCMC
唯一实现不变。其余18B+18PB和IR继续必需，不能用三组P结束完整goal。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
from rsd_b000_contract import PART, digest, save_json, F_GROWTH
from rsd_bonly_common import TABLE
import rsd_pb_common as spectral
import run_rsd_b000_mcmc as shared
from run_sto_post_noir_mcmc import PostPOnlyCase as SharedPostPOnlyCase
from post_rsd_noir_case_inputs import case_specs
from post_rsd_p0only_inputs import make_p0_case, verify_ensemble_arrays

ROOT = PART/'outputs/sto_marisa_post_rsd_mean100_v1'
OUT = ROOT/'mcmc_noir_p0only'
CAL = ROOT/'physical_p_calibration_mcmc'
CAL_NAME = 'physical_p_GSR2_RSD_fNLfixed100'
HOLDOUT = ROOT/'calibration_holdout_p0'


def read(path):
    return json.loads(Path(path).read_text())


def parent():
    value = read(OUT/'CONTRACT.json')
    if (value['status'] != 'POST_P0_ONLY_THREE_MAIN_INPUTS_VALIDATED_NOT_B_RELEASE'
            or value['main_P0_case_count'] != 3 or value['remaining_main_B_PB_case_count'] != 36
            or value['B_noise_model_released'] or value['full_goal_reduced']):
        raise ValueError('not the separately scoped main-P0 contract')
    for path, want in value['immutable_inputs'].items():
        if digest(path) != want:
            raise ValueError('post-P0 frozen source/input drift: '+path)
    for name, want in value['case_contract_sha256'].items():
        if digest(OUT/'cases'/name/'contract.json') != want:
            raise ValueError('post-P0 case contract drift: '+name)
    return value


def freeze():
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one logical core required')
    from continue_sto_post_p0only import validate_calibration_exit
    validate_calibration_exit()
    if OUT.exists():
        parent(); print('REUSE same frozen three main post-P0 inputs', flush=True); return
    admission = ROOT/'FID500_ADMISSION_COMPLETION_AUDIT.json'
    if read(admission)['status'] != 'ALL500_FIDUCIAL_SOURCES_AND_MEASUREMENT_ADMISSION_INDEPENDENTLY_VERIFIED':
        raise ValueError('all500 actual source admission required')
    report_path = ROOT/'ensemble_measurements/AUDIT.json'; report = read(report_path)
    if report['selection_sha256'] != digest(ROOT/'ENSEMBLE_SELECTION.json'):
        raise ValueError('assembled ensemble selection drift')
    datasets = {}; matrix_paths = []
    for definition in ('matched', 'nonzero_diagnostic'):
        info = report['ensemble_files'][definition]; path = Path(info['path'])
        if digest(path) != info['sha256']:
            raise ValueError('actual ensemble payload drift: '+definition)
        with np.load(path, allow_pickle=False) as z:
            datasets[definition] = {k: np.array(z[k]) for k in z.files}
        matrix_paths.append(path)
    ensemble, keys = verify_ensemble_arrays(report, datasets['matched'], datasets['nonzero_diagnostic'])
    summary_path = CAL/'cases'/CAL_NAME/'SUMMARY.json'; summary = read(summary_path)
    cal_parent = read(CAL/'contract.json'); cal_case = CAL/'cases'/CAL_NAME/'contract.json'
    if digest(cal_case) != cal_parent['case_contract_sha256']:
        raise ValueError('original physical-p calibration case drift')
    for path, want in cal_parent['immutable_inputs'].items():
        if digest(path) != want:
            raise ValueError('original physical-p calibration source/input drift: '+path)
    if (summary['converged'] is not True or summary['parameters'][0] != 'physical_p'
            or summary['calibration_fNL_fixed'] != 100. or summary['fNL_fitted'] is not False):
        raise ValueError('same-tracer original physical-p calibration must converge')
    quantiles = np.asarray(summary['quantiles_2p5_16_50_84_97p5'], dtype=float)
    if quantiles.shape != (5, 3) or not np.isfinite(quantiles).all():
        raise ValueError('actual physical-p calibration quantiles required')
    p_halo = float(quantiles[2, 0])
    window_path = HOLDOUT/'P0_WINDOW.npz'; audit_path = HOLDOUT/'POST_P0_WINDOW_AUDIT.json'
    audit = read(audit_path)
    if (audit['status'] != 'PASS' or not audit['holdout_measurement_nmodes_match']
            or digest(window_path) != audit['window_sha256']):
        raise ValueError('actual production post-P window audit required')
    with np.load(window_path, allow_pickle=False) as z:
        edges = np.array(z['edges']); nmodes = np.array(z['nmodes'])
    np.testing.assert_array_equal(edges, datasets['matched']['P_edges'])
    previous = spectral.OUT
    try:
        spectral.OUT = HOLDOUT; templates, modes, k = spectral.p_lattice(edges)
    finally:
        spectral.OUT = previous
    np.testing.assert_array_equal(modes, nmodes)
    moments = datasets['matched']['data_N2_N3_nhalo']
    if moments.shape != (700, 3) or not np.isfinite(moments).all() or np.any(moments[:, 0] <= 0):
        raise ValueError('actual positive N2 parameter units required')
    positions = {key: i for i, key in enumerate(keys)}; ready = []
    for spec in (s for s in case_specs() if s.observable == 'P0_only'):
        unit = float(moments[[positions[spec.cosmology, i] for i in range(100)], 0].mean())
        arrays, meta = make_p0_case(ensemble, spec, templates, unit, p_halo=p_halo)
        meta.update(f_growth_fixed=F_GROWTH, physical_p_calibration_summary_sha256=digest(summary_path),
                    fixed_reconstruction=report['fixed_reconstruction'],
                    likelihood_scope='P0-only; neither B estimator selection nor B noise audit is bypassed')
        ready.append((spec, arrays, meta))
    sources = [Path(__file__), PART/'code/post_rsd_p0only_inputs.py',
        PART/'code/post_rsd_ensemble.py', PART/'code/post_rsd_noir_case_inputs.py',
        PART/'code/post_rsd_noir_polynomial.py', PART/'code/post_rsd_noir_likelihood.py',
        PART/'code/run_sto_post_noir_mcmc.py', PART/'code/run_rsd_p0only.py',
        PART/'code/run_rsd_b000_mcmc.py', PART/'code/rsd_pb_common.py',
        PART/'code/rsd_b000_contract.py', PART/'code/post_rsd_p0_model.py', TABLE,
        PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py',
        ROOT/'ENSEMBLE_SELECTION.json', admission, report_path, *matrix_paths,
        CAL/'contract.json', cal_case, summary_path, CAL/'cases'/CAL_NAME/'RUN_PROVENANCE.json',
        window_path, audit_path, PART/'code/continue_sto_post_p0only.py',
        ROOT/'p0only_continuation_v1/CONTRACT.json', ROOT/'p0only_continuation_v1.launch.json',
        ROOT/'physical_p_continuation_login_v3/STATUS.json']
    sources += [Path(p) for p in cal_parent['immutable_inputs']]
    frozen = {str(p): digest(p) for p in dict.fromkeys(sources)}
    # 全部真实门通过后才创建目录；不存在伪B噪声报告或部分盒数回退。
    OUT.mkdir(); contracts = {}
    for spec, arrays, meta in ready:
        directory = OUT/'cases'/spec.name; directory.mkdir(parents=True)
        np.savez_compressed(directory/'likelihood.npz', **arrays, P_k=k, P_nmodes=modes, P_edges=edges)
        meta['likelihood_sha256'] = digest(directory/'likelihood.npz')
        save_json(directory/'contract.json', meta); contracts[spec.name] = digest(directory/'contract.json')
    save_json(OUT/'CASES.json', [s.name for s, _, _ in ready])
    frozen[str(OUT/'CASES.json')] = digest(OUT/'CASES.json')
    save_json(OUT/'CONTRACT.json', dict(status='POST_P0_ONLY_THREE_MAIN_INPUTS_VALIDATED_NOT_B_RELEASE',
        immutable_inputs=frozen, case_contract_sha256=contracts, main_P0_case_count=3,
        remaining_main_B_PB_case_count=36, original_main_case_count=39, full_goal_reduced=False,
        B_noise_model_released=False, B_estimator_selection_released=False, IR_completed=False,
        p_fixed_physical=p_halo, covariance_is_of_mean=False, covariance_multiplier=1.,
        volume_Gpc_h3=1., strict_independent_mock_recovery=False,
        B_definition_independence='actual P rows and P-P covariance identical in both B ensembles'))
    save_json(OUT/'FREEZE_AUDIT.json', dict(status='THREE_MAIN_POST_P0_INPUTS_FROZEN_NOT_SAMPLED',
        main_cases=list(contracts), same_P_input_and_covariance_both_B_definitions=True,
        covariance_recomputed_from_actual500_rows=True, target_means_recomputed_from_actual100_rows=True,
        MCMC_started=False, B_noise_audit_created=False, full_goal_reduced=False))
    print('FROZEN three main post-P0 cases; 36B/PB and IR still required', flush=True)


class POnlyCase(SharedPostPOnlyCase):
    def __init__(self, directory):
        self.directory = Path(directory); self.meta = read(self.directory/'contract.json')
        if self.meta['B_likelihood'] or not self.meta['P_likelihood'] or self.meta['IR']:
            raise ValueError('only no-IR main post-P0 allowed')
        if digest(self.directory/'likelihood.npz') != self.meta['likelihood_sha256']:
            raise ValueError('post-P0 likelihood drift')
        with np.load(self.directory/'likelihood.npz', allow_pickle=False) as z:
            for name in z.files:
                setattr(self, name, np.array(z[name]))
        self.names = self.meta['parameters']; self.ndim = len(self.names)
        if self.names != ['fNL', 'b1', 'Pshot_residual_nbar'] or self.target.shape != (12,):
            raise ValueError('original main P0 parameter/window dimensions required')
        if (self.bounds.shape != (3, 2) or self.templates.shape != (6, 12) or self.noise.shape != (12,)
                or not np.isfinite(self.meta['p_fixed_physical'])):
            raise ValueError('original P-only arrays and explicit physical p required')
        from post_rsd_noir_case_inputs import B_BOUNDS
        np.testing.assert_array_equal(self.bounds[:2], B_BOUNDS[:2])
        np.testing.assert_array_equal(self.noise, np.full(12, self.meta['Pshot_unit_Mpc_h3']))
        np.testing.assert_allclose(self.bounds[2]*self.meta['Pshot_unit_Mpc_h3'], [-1e6, 1e6], rtol=2e-16)
        alpha = 486/499.
        if (self.meta['Hartlap_factor'] != alpha or self.meta['n_covariance'] != 500
                or self.meta['n_target'] != 100 or self.meta['covariance_is_of_mean']
                or self.meta['covariance_multiplier'] != 1. or self.meta['volume_Gpc_h3'] != 1.):
            raise ValueError('original mean100/C_single500/V1 required')
        np.testing.assert_array_equal(self.bounds, self.meta['bounds'])
        np.testing.assert_allclose(self.chol@self.chol.T, self.covariance/alpha, rtol=3e-13, atol=1e-8)
        self.white_templates = np.linalg.solve(self.chol, self.templates.T).T
        self.nwhite = np.linalg.solve(self.chol, self.noise)
        self.twhite = np.linalg.solve(self.chol, self.target)


def sample(index, max_steps=64000, extend_budget=False):
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one logical CPU per MCMC required')
    parent()
    if index not in range(3):
        raise ValueError('only three main post-P0 cases')
    previous = shared.OUT, shared.Case, shared.budget_provenance, shared.save_json
    def provenance(directory, value, extend=False):
        value.update(post_P0_adapter_sha256=digest(Path(__file__)),
                     parent_contract_sha256=digest(OUT/'CONTRACT.json'))
        return previous[2](directory, value, extend)
    def save(path, value):
        if Path(path).name == 'SUMMARY.json':
            meta = read(Path(path).parent/'contract.json')
            value.update(data_stage='STO+Marisa post-recon RSD', P_likelihood=True, B_likelihood=False,
                P_bins=12, B_bins=0, kmax_P=.08, kmax_B=None, P_B_cross_covariance=False,
                covariance_multiplier=1., covariance_is_of_mean=False, p_fixed_physical=meta['p_fixed_physical'],
                physical_p_is_not_p_rec=True, IR=False, B_noise_model_released=False,
                gof_pvalue=None, gof_note='mean100 target with C_single500; no calibrated GOF p-value')
        return previous[3](path, value)
    shared.OUT, shared.Case, shared.budget_provenance, shared.save_json = OUT, POnlyCase, provenance, save
    try:
        return shared.run(index, max_steps, extend_budget)
    finally:
        shared.OUT, shared.Case, shared.budget_provenance, shared.save_json = previous


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('command', choices=('freeze', 'sample'))
    ap.add_argument('--case-index', type=int, default=0); ap.add_argument('--max-steps', type=int, default=64000)
    ap.add_argument('--extend-budget', action='store_true'); args = ap.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        sample(args.case_index, args.max_steps, args.extend_budget)
