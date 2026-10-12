"""模块大纲：真实300份marked探针的独立显式三/四点回放。

不调用producer的partition、moment/cumulant或标签响应函数；手写本探针的
有限uniform random条件矩、三个Wick配对与connected remainder。只认证3个
有向三角形/有限q，绝不生成formal noise PASS、协方差或MCMC放行文件。
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

TOL = 2e-11
EXPECTED_MATRIX_SHA = '18da8eb56250cab3c6052cb2d0197970e5386621b4a45aa65cb9552507347f58'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    a = np.asarray(value)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def decode(value):
    a = np.asarray(value)
    return a[..., 0] + 1j*a[..., 1]


def explicit_replay(fields, counts, modes, triangle, shift_modes, response):
    """原非零闭合外腿；直接条件积分，不复用生产矩引擎。

    返回各阶[corrected,raw,self]及逐盒完整响应。random在三F1项中只能
    有一对非零反向mode；三random的闭合和=-q，因此为零。D始终data-only。
    """
    lookup = {tuple(k): i for i, k in enumerate(modes)}
    size = len(fields)
    zero = np.zeros(size, dtype=complex)

    def d(power, k):
        return fields[:, power-1, lookup[tuple(k)]]

    def covariance(k, a, f_a, b, f_b):
        # 非零总波矢由齐次平移精确为零；F1(0)由归一化精确为零。
        if np.any(a+b) or (f_a and not np.any(a)) or (f_b and not np.any(b)):
            return 0j
        row = d(1, a)*d(1, b)
        if f_a and f_b and np.any(a):
            row = row + 1/counts
        return row.mean()

    raw = np.zeros(size, dtype=complex)
    self_term = np.zeros(size, dtype=complex)
    parts = np.zeros((3, 3), dtype=complex)  # pair/third/fourth × corrected/raw/self
    for leg, k in enumerate(triangle):
        others = [j for j in range(3) if j != leg]
        b, c = triangle[others]
        for q, r in zip(shift_modes, response):
            g = np.dot(k, q)/np.dot(q, q)*r
            a = k-q
            # E4全部F1外腿非零；内部a=0时整项消失，不能用D1(0)=1替代。
            e4 = zero.copy() if not np.any(a) else d(1, q)*d(1, a)*d(1, b)*d(1, c)
            if np.any(a):
                for u, v, w in ((a,b,c), (b,a,c), (c,a,b)):
                    if not np.any(v+w):
                        e4 += d(1, q)*d(1, u)/counts
            wick = (covariance(q, q, False, a, True)*covariance(q, b, True, c, True)
                    + covariance(q, q, False, b, True)*covariance(q, a, True, c, True)
                    + covariance(q, q, False, c, True)*covariance(q, a, True, b, True))
            t = -k-q
            e3a = d(1, q)*(d(2, t) + (1/counts if not np.any(t) else 0))*d(1, k)
            e3b = zero.copy() if not np.any(a) else d(2, -k)*d(1, q)*d(1, a)
            mean2 = (d(2, t)+1/counts).mean() if not np.any(t) else 0j
            mean_pair = mean2*covariance(q, q, False, k, True)
            raw -= g*e4
            self_term += g*(e3a-e3b)
            pair = np.array([-g*(wick+mean_pair), -g*wick, g*mean_pair])
            third = np.array([-g*(e3a.mean()-mean_pair-e3b.mean()), 0j,
                              g*(e3a.mean()-mean_pair-e3b.mean())])
            fourth = np.array([-g*(e4.mean()-wick), -g*(e4.mean()-wick), 0j])
            parts += np.array([pair, third, fourth])
    full_rows = np.array([raw-self_term, raw, self_term])
    intrinsic = np.prod(np.array([d(1,k) for k in triangle]), axis=0)
    intrinsic -= sum(d(2,-k)*d(1,k) for k in triangle)
    intrinsic += 2*d(3, np.zeros(3,dtype=int))
    return parts, full_rows, intrinsic


def audit(directory, source_audit=None):
    directory = Path(directory)
    report_path = directory/'REPORT.json'
    report = json.loads(report_path.read_text())
    matrix = directory/'ACTUAL300_PARTICLE_FOURIER_FIELDS.npz'
    if sha(matrix) != EXPECTED_MATRIX_SHA or report['field_matrix_sha256'] != EXPECTED_MATRIX_SHA:
        raise ValueError('actual300 matrix identity drift')
    if report['status'] != 'ACTUAL300_FINITE_RANDOM_MARKED_CONNECTED_PROBES_COMPLETE_NOT_STO_CLOSURE':
        raise ValueError('producer is not complete')
    with np.load(matrix, allow_pickle=False) as z:
        fields=z['fields']; modes=z['integer_modes']; counts=z['nrandom']
        keys=[json.loads(s) for s in z['target_keys_json']]
    expected=[[cosmo,r] for cosmo in ('fiducial','LC_p','LC_m') for r in range(100)]
    if keys != expected or fields.shape != (300,3,169) or counts.shape != (300,):
        raise ValueError('actual300 source selection or field shape differs')
    if not np.isfinite(fields).all() or np.any(counts <= 0):
        raise ValueError('nonfinite or invalid actual field matrix')
    if [[r['cosmology'],r['realization']] for r in report['records']] != expected:
        raise ValueError('source record identity differs')
    lookup={tuple(k):j for j,k in enumerate(modes)}
    zero=lookup[(0,0,0)]
    np.testing.assert_allclose(fields[:,0,zero],1.,rtol=0,atol=3e-15)
    if not np.array_equal(counts,30*np.array([r['nhalo'] for r in report['records']])):
        raise ValueError('finite random counts are not actual 30*nhalo')
    conjugate=max(np.max(np.abs(fields[:,:,j]-fields[:,:,lookup[tuple(-k)]].conj()))
                  for j,k in enumerate(modes))
    if conjugate > 3e-15:
        raise ValueError('exact particle DFT conjugacy fails')
    source_verified=False
    if source_audit is not None:
        admission=json.loads(Path(source_audit).read_text())
        rows={(r['cosmology'],r['realization']):r for r in admission['source_rows']}
        for i,r in enumerate(report['records']):
            parent=rows[tuple(expected[i])]
            for key in ('source_sha256','diagnostics_sha256'):
                if parent[key] != r[key]:
                    raise ValueError('producer source hash differs from source admission')
            if parent['raw_fof']['source_raw_sha256'] != r['raw_fof_sha256']:
                raise ValueError('FoF hash differs from source admission')
            np.testing.assert_allclose([1e9*fields[i,1,zero].real,1e18*fields[i,2,zero].real],
                [parent['N2_data_self'],parent['N3_data_self']],rtol=8*np.finfo(float).eps)
        source_verified=True
    result=[];max_error=0.;max_random_error=0.
    for index, group in enumerate(report['analysis']):
        sl=slice(index*100,(index+1)*100)
        for probe in group['probes']:
            triangle=np.array(probe['triangle_integer_modes'])
            q=np.array(probe['shift_support_integer_modes'])
            r=np.array(probe['shift_response'])
            parts, full, intrinsic=explicit_replay(fields[sl],counts[sl],modes,triangle,q,r)
            pair,third,fourth=parts*1e18
            obtained=dict(pair_and_means=pair,third_cumulant=third,fourth_cumulant=fourth,
                          full=parts.sum(axis=0)*1e18)
            expected_parts={k:decode(v) for k,v in probe['cumulant_parts_physical'].items()}
            scale=max(1.,max(np.max(np.abs(v)) for v in expected_parts.values()))
            error=max(np.max(np.abs(obtained[k]-expected_parts[k])) for k in obtained)/scale
            error=max(error,np.max(np.abs(full.mean(axis=1)*1e18-obtained['full']))/scale)
            error=max(error,abs(intrinsic.mean()*1e18-decode(probe['intrinsic_factorial_B_physical']))/scale)
            scatter=np.std(full,axis=1,ddof=1)*1e18
            error=max(error,np.max(np.abs(scatter-decode(probe['exact_first_order_response_single_box_scatter'])))/scale)
            if error > TOL:
                raise ValueError('independent actual probe replay fails')
            # 对同一真实data条件random平均：Nr∞对照，只要求corrected相消，raw/self不相消。
            _, continuous, _=explicit_replay(fields[sl],np.full(100,np.inf),modes,triangle,q,r)
            delta=full-continuous
            random_scale=max(np.max(np.abs(full)),np.max(np.abs(continuous)),1e-30)
            random_error=np.max(np.abs(delta[0]))/random_scale
            if random_error > TOL:
                raise ValueError('conditional finite-random corrected cancellation fails')
            max_error=max(max_error,float(error));max_random_error=max(max_random_error,float(random_error))
            result.append(dict(cosmology=group['cosmology'],triangle=triangle.tolist(),
                independent_scaled_error=float(error),conditional_corrected_random_error=float(random_error),
                raw_and_self_finite_random_terms_retained=bool(np.max(np.abs(delta[1:]))>0),
                corrected_pair_third_fourth_physical=encode(np.array([pair[0],third[0],fourth[0]])),
                full_corrected_single_box_scatter=float(scatter[0])))
    proof=dict(status='INDEPENDENT_ACTUAL300_MARKED_FINITE_RANDOM_CONNECTED_PROBE_REPLAY_PASS_NOT_STO_CLOSURE',
        created_UTC=datetime.now(timezone.utc).isoformat(),actual_catalogs=300,n_per_truth=100,
        directed_triangles_per_truth=3,required_modes=169,q_per_probe=12,
        matrix_sha256=sha(matrix),producer_report_sha256=sha(report_path),audit_source_sha256=sha(__file__),
        source_admission_metadata_verified=source_verified,max_scaled_error=max_error,tolerance=TOL,
        max_conditional_corrected_random_cancellation_error=max_random_error,
        max_particle_DFT_conjugate_error=float(conjugate),probes=result,
        covariance_created=False,data_vectors_changed=False,pure_noise_separated=False,
        all44_full_shell_closure=False,finite_production_shift_closure=False,IR_noise_closure=False,
        formal_noise_audit_released=False,MCMC_started=False)
    output=directory/'INDEPENDENT_REPLAY_AUDIT.json'
    if output.exists():
        raise ValueError('preserve previous independent audit; inspect before rerun')
    output.write_text(json.dumps(proof,indent=2,ensure_ascii=False)+'\n')
    return proof


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',required=True);ap.add_argument('--source-audit')
    args=ap.parse_args();proof=audit(args.directory,args.source_audit)
    print(json.dumps({k:proof[k] for k in ('status','max_scaled_error',
        'max_conditional_corrected_random_cancellation_error','source_admission_metadata_verified')},indent=2))


if __name__=='__main__':main()
