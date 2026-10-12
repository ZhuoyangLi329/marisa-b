"""模块大纲：直接IR节点文件→独立节点积插值/手写Chebyshev递推回放。

不调用生产DCT、Clenshaw或IRSignalInterpolation；逐节点对原direct设计，
回放9/17/33实际系数和全部off-node探针。仅认证已完成低bin的表示。
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path('/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model')
OUT = ROOT/'outputs/sto_marisa_post_rsd_mean100_v1/ir_bias_interpolation_probe_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def scaled_error(a, b):
    power = np.array([1., 500., 250000.])[None, :, None]
    scale = np.maximum(1., np.max(np.abs(b)*power, axis=1, keepdims=True))
    return float(np.max(np.abs(a-b)*power/scale))


def main():
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one logical CPU required')
    result_path = OUT/'INDEPENDENT_INTERPOLATION_AUDIT.json'
    if result_path.exists():
        raise ValueError('preserve completed independent replay')
    report = read(OUT/'AUDIT.json'); contract = read(OUT/'CONTRACT.json')
    if (report['status'] != 'ACTUAL_LOW_BIN_WHOLE_B1_DOMAIN_IR_INTERPOLATION_PROBE_PASS'
            or sha(OUT/'CONTRACT.json') != report['contract_sha256']):
        raise ValueError('actual completed original-spectrum low-bin report required')
    for path, expected in {**contract['immutable_sources'], **contract['active_measurement_sources']}.items():
        if sha(path) != expected:
            raise ValueError('original source drift: '+path)
    direct = {}; paths = sorted((OUT/'designs').glob('projection*.json'))
    if len(paths) != report['actual_full_IR_projections']:
        raise ValueError('actual direct projection count mismatch')
    for path in paths:
        value = read(path)
        if (value['contract_sha256'] != report['contract_sha256']
                or value['denominator'] != 153174 or value['counts'] != [147, 1042]):
            raise ValueError('original source or full-shell geometry differs')
        key = float(value['b1']), value['mesh']
        if key in direct:
            raise ValueError('duplicate direct parameter/grid record')
        direct[key] = np.array(value['design'])
    rows = []; frozen = {str(path): sha(path) for path in paths}
    for candidate in report['node_candidates']:
        n = candidate['nodes']; path = OUT/f'candidate{n:02d}.npz'
        if sha(path) != candidate['coefficients_sha256']:
            raise ValueError('actual node payload SHA drift')
        frozen[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as payload:
            c = payload['coefficients']; nodes = payload['node_b1']; samples = payload['node_designs']
            probes = payload['probes']; produced = payload['predictions']
        if c.shape != (n, 3, 3, 1) or samples.shape != c.shape:
            raise ValueError('IR node/bias/fNL dimensions changed')
        expected_nodes = 2.25+1.75*np.cos((np.arange(n)+.5)*np.pi/n)
        np.testing.assert_array_equal(nodes, expected_nodes)
        np.testing.assert_array_equal(probes, contract['probes'])
        for b, sample in zip(nodes, samples):
            np.testing.assert_array_equal(sample, direct[float(b), 48])
        # 独立节点差乘积权重，不借用Chebyshev库或生产转换。
        weights = np.array([1/np.prod(node-np.delete(nodes, i)) for i, node in enumerate(nodes)])
        errors = []
        for b, production in zip(probes, produced):
            x = (float(b)-2.25)/1.75
            t = [1., x]
            for _ in range(2, n):
                t.append(2*x*t[-1]-t[-2])
            manual = np.tensordot(np.array(t), c, axes=(0, 0))
            ratio = weights/(b-nodes)
            independent = np.tensordot(ratio, samples, axes=(0, 0))/ratio.sum()
            error = max(scaled_error(manual, independent), scaled_error(production, independent))
            if error >= 1e-9:
                raise ValueError('independent direct-node interpolation replay failed')
            actual_error = scaled_error(manual, direct[float(b), 48])
            errors.append(dict(b1=float(b), independent_representation_scaled_error=error,
                               actual_IR_projection_scaled_error=actual_error))
        rows.append(dict(nodes=n, node_designs_independently_matched=n, probes=errors,
            maximum_independent_representation_error=max(v['independent_representation_scaled_error'] for v in errors),
            maximum_direct_IR_projection_error=max(v['actual_IR_projection_scaled_error'] for v in errors)))
    result = dict(status='ACTUAL_LOW_BIN_IR_COEFFICIENTS_INDEPENDENT_NODE_PRODUCT_REPLAY_PASS',
        created_UTC=datetime.now(timezone.utc).isoformat(), producer_report_sha256=sha(OUT/'AUDIT.json'),
        producer_contract_sha256=sha(OUT/'CONTRACT.json'), source_sha256=sha(Path(__file__)),
        immutable_projection_payloads=frozen, actual_full_IR_projection_files=len(paths), rows=rows,
        scaled_tolerance=1e-9, production_DCT_Clenshaw_or_model_called=False,
        whole44_parameter_domain_verified=False, calibrated_physical_p=False,
        post_covariance_whitening_done=False, actual_STO_noise_validated=False, MCMC_released=False)
    with result_path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: result[key] for key in ('status','actual_full_IR_projection_files','rows','MCMC_released')}))


if __name__ == '__main__':
    main()
