"""模块大纲：导出实际生产B000的格点标签，逐mode验证现有FFT理论窗口。

调用原粒子准备和BinMesh3SpectrumPoles，不修改共享包或拟合分箱。
旧GPU参考的mode counts不是当前CPU窗口；只有实际标签与测量对照可判定匹配。
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from mpi4py import MPI
import jaxpower
import jaxpower.mesh3 as binner_source
import clustering_statistics.spectrum3_tools as production


PART = Path(__file__).resolve().parents[1]
ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'
sys.path.insert(0, str(PART.parent / '01_reconstruction_methods/common'))
sys.path.insert(0, str(PART.parent / '01_reconstruction_methods/marisa_gra'))
import halo_b000_pipeline as shared
from rsd_bonly_common import fft


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if len(os.sched_getaffinity(0)) != 1 or MPI.COMM_WORLD.size != 1:
        raise RuntimeError('one CPU and one MPI rank required')
    out = ROOT / 'estimator_B_window'
    if out.exists(): raise RuntimeError('preserve previous B window audit')
    directory = ROOT / 'pilots/fiducial_real000_rec256_meas256_ran30_selfmatched_zeroaudit_nested'
    done = json.loads((directory / 'DONE.json').read_text())
    metadata = json.loads((directory / 'CONTRACT.json').read_text())
    if (done['post_halos_sha256'] != sha(directory / 'POST_HALOS.npz')
            or done['measurement_sha256'] != sha(directory / 'MEASUREMENTS.npz')
            or metadata['shared_measure_source_sha256'] != sha(shared.__file__)):
        raise ValueError('production pilot provenance changed')
    reference = PART / 'outputs/realspace_binning_reference/REALSPACE_BINNING.npz'
    original = json.loads((PART / 'outputs/pre_rsd_b000_bonly_mean100_v1/contract.json').read_text())
    if sha(reference) != original['reference_sha256']: raise ValueError('reference geometry changed')
    with np.load(reference, allow_pickle=False) as z:
        ref = {k: np.array(z[k]) for k in z.files}
    with np.load(directory / 'MEASUREMENTS.npz', allow_pickle=False) as z:
        measured_counts = z['B_post_nmodes'].copy()
    with np.load(directory / 'POST_HALOS.npz', allow_pickle=False) as z:
        positions = z['positions'].copy(); weights = z['weights'].copy()
    cat = shared.make_halo_catalog(positions, 'z', MPI.COMM_WORLD, weights=weights, redshift=.5)
    mattrs = {'boxsize': 1000. * np.ones(3, dtype='f4'), 'boxcenter': 500. * np.ones(3, dtype='f4'), 'meshsize': 256}
    # box/mesh显式固定；原prepare入口从真实post halo取得与生产相同的dtype/attrs。
    with shared.jaxpower_runtime_compat(MPI.COMM_WORLD):
        with jaxpower.create_sharding_mesh(meshsize=256):
            particles = production.prepare_jaxpower_particles(lambda: {'data': cat}, mattrs=mattrs)
            attrs = particles[0]['data'].attrs
            binning = jaxpower.BinMesh3SpectrumPoles(attrs, edges=ref['edges'], basis='sugiyama',
                ells=[(0, 0, 0)], buffer_size=0, mask_edges=shared.resolve_mask_edges('default'))
            coords = binning.mattrs.kcoords(kind='index', sparse=True)
            shape = np.broadcast_shapes(*[q.shape for q in coords])
            labels = [np.asarray(q).reshape(shape) for q in binning.ibin1d]
            if shape != (256, 256, 256): raise ValueError('unexpected Fourier geometry')
            np.testing.assert_array_equal(labels[0], labels[1])
            pairs = np.asarray(binning._iedges)
            counts = np.asarray(binning.nmodes1d[0])
            np.testing.assert_array_equal(pairs, ref['shell_pairs'])
            np.testing.assert_array_equal(counts[pairs[:, 0]] * counts[pairs[:, 1]], measured_counts)
            for shell in range(len(counts)):
                if np.count_nonzero(labels[0] == shell) != counts[shell]:
                    raise ValueError('actual labels disagree with production counts')
            # 与原FFTProjection完全相同的整数格点和float64物理波数，不手调边界。
            integers = [np.rint(np.asarray(q)).astype('i2') for q in coords]
            k2 = sum((q.astype('f8') * (2 * np.pi / 1000.)) ** 2 for q in integers)
            theoretical = fft.assign_shells(np.sqrt(k2), ref['edges'])
            fit = ref['certified_indices']; used = np.unique(pairs[fit])
            union = np.isin(theoretical, used) | np.isin(labels[0], used)
            mismatch = int(np.count_nonzero(theoretical[union] != labels[0][union]))
            if mismatch: raise ValueError(f'{mismatch} actual/theory lattice labels differ in fitted shells')
            model_counts = np.bincount(theoretical[theoretical >= 0], minlength=len(counts))
            model_pairs = model_counts[pairs[:, 0]] * model_counts[pairs[:, 1]]
            np.testing.assert_array_equal(model_pairs[fit], measured_counts[fit])
            select = np.isin(labels[0], used)
            vectors = np.column_stack([np.broadcast_to(q, shape)[select] for q in integers])
            compressed_labels = labels[0][select].astype('i2')
            for shell in used:
                if np.count_nonzero(compressed_labels == shell) != counts[shell]:
                    raise ValueError('compressed window lost a mode')
    paths = [Path(__file__), Path(shared.__file__), Path(production.__file__), Path(binner_source.__file__),
             Path(fft.__file__), PART / 'code/rsd_b000_fft_projection.py', reference,
             directory / 'DONE.json', directory / 'MEASUREMENTS.npz']
    report = {'status': 'PRODUCTION_B000_WINDOW_MATCHES_EXISTING_FFT_ALL44_BINS',
              'created_UTC': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
              'CPU_affinity': sorted(os.sched_getaffinity(0)), 'backend': jax.default_backend(),
              'production_mesh': 256, 'basis': 'sugiyama', 'multipole': [0, 0, 0],
              'used_shells': used.tolist(), 'mode_count_exported': len(vectors), 'fit_bins': len(fit),
              'actual_theory_label_mismatches_in_fitted_shells': mismatch,
              'all120_actual_pair_counts_match_measured_pilot': True,
              'all44_model_pair_counts_match_measured_pilot': True,
              'legacy_reference_count_difference_fit_bins': fit[ref['B000_nmodes'][fit] != measured_counts[fit]].tolist(),
              'legacy_reference_max_relative_count_difference': float(np.max(np.abs(measured_counts[fit] / ref['B000_nmodes'][fit] - 1))),
              'theory_or_measurement_modified': False, 'old_reference_preserved': True,
              'third_leg_kmax_cut_added': False, 'full_two_shell_denominator_preserved': True,
              'zero_leg_source_policy_not_changed': True, 'main_MCMC_released': False,
              'source_sha256': {str(p): sha(p) for p in paths}}
    out.mkdir()
    target = out / 'B000_WINDOW.npz'
    np.savez_compressed(target, vectors_integer=vectors, labels=compressed_labels, used_shells=used,
                        shell_counts=counts, pairs=pairs, measured_pair_counts=measured_counts,
                        model_pair_counts=model_pairs, fit_indices=fit, edges=ref['edges'])
    report['window_sha256'] = sha(target)
    with (out / 'AUDIT.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
