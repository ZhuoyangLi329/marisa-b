"""模块大纲：已完成44-bin算子的独立标量/整数闭合回放，原双网格与误差门。

复用既有oracle，仅读取已验收的PN_same/NN系数；不重算vector模板。
只认证两个算子的全44几何/数值，不校准STO、不选择零腿定义、不放行MCMC。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
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
import numpy as np
from rsd_b000_contract import PART, reference, digest, save_json
from post_rsd_ir_inputs import ROOT, frozen_reconstruction
from post_rsd_poisson_projection import PoissonShiftFFTProjection
from audit_post_rsd_noise_transport_coefficients import (
    actual_job, evaluate_independent, scalar_oracle, integer_count_oracle)
from run_sto_post_login_queue_v2 import memory_budget

OUT = ROOT/'noise_scalar_all44_v1'
SOURCE = ROOT/'noise_transport_templates_v1'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detach', action='store_true')
    ap.add_argument('--job-id', type=int, default=59597420)
    args = ap.parse_args()
    if socket.gethostname() != 'login26' or os.sched_getaffinity(0) != {12}:
        raise ValueError('single login26 CPU12, shared with lightweight controllers, required')
    if OUT.exists():
        raise RuntimeError('preserve existing all44 execution; inspect its actual PID, never duplicate')
    if args.detach:
        log = ROOT/'noise_scalar_all44_v1.log'
        launch = ROOT/'noise_scalar_all44_v1.launch.json'
        if log.exists() or launch.exists():
            raise RuntimeError('prior launch must be inspected, not overwritten')
        with log.open('xb') as stream:
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--job-id', str(args.job_id)]
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=proc.pid, host=socket.gethostname(), command=command,
            CPU_affinity=[12], launched_UTC=datetime.now(timezone.utc).isoformat(), log=str(log))
        with launch.open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True)
        return
    observed = subprocess.run(['sacct', '-X', '-n', '-P', '-j', str(args.job_id),
        '-o', 'JobIDRaw,State,ExitCode'], check=True, text=True, stdout=subprocess.PIPE, timeout=20)
    job = actual_job(observed.stdout, args.job_id)
    if job['state'] != 'COMPLETED' or job['exit_code'] != '0:0':
        raise ValueError('actual successful operator job termination required')
    cp = SOURCE/'CONTRACT.json'; contract = read(cp)
    replay_path = SOURCE/'INDEPENDENT_COEFFICIENT_AUDIT.json'; replay = read(replay_path)
    aggregate = SOURCE/'OPERATOR_COEFFICIENTS.npz'; status = read(SOURCE/'COMPUTE_STATUS.json')
    if (replay['status'] != 'ALL44_NOISE_OPERATOR_NODE_AND_FINE_GRID_REPLAY_PASS_NOT_STO_CLOSURE'
        or replay['actual_job'] != job or replay['bins'] != 44 or replay['nodes_replayed'] != 396
        or replay['fine_probes_replayed'] != 88 or status['state'] != 'COMPLETE'
        or status['active_workers'] or status['job_id'] != str(args.job_id)
        or replay['contract_sha256'] != digest(cp)
        or replay['coefficients_sha256'] != digest(aggregate)):
        raise ValueError('complete same-job independently verified operator coefficients required')
    frozen = dict(contract['immutable_sources'])
    for p in (Path(__file__), PART/'code/audit_post_rsd_noise_transport_coefficients.py',
              PART/'code/run_sto_post_login_queue_v2.py', cp, replay_path, aggregate,
              SOURCE/'COMPLETION_AUDIT.json', SOURCE/'COMPUTE_STATUS.json'):
        frozen[str(p)] = digest(p)
    def guard():
        for p, want in frozen.items():
            if digest(p) != want:
                raise ValueError('all44 read-only audit input/source drift: '+p)
    guard()
    with np.load(aggregate, allow_pickle=False) as z:
        coefficients = np.array(z['coefficients'])
        np.testing.assert_array_equal(z['indices'], contract['indices'])
    if coefficients.shape != (3, 3, 3, 3, 44) or len(set(contract['indices'])) != 44:
        raise ValueError('original full44 coefficient shape and unique registry required')
    with np.load(ROOT/'estimator_B_window/B000_WINDOW.npz', allow_pickle=False) as z:
        counts = np.array(z['measured_pair_counts'])
    budget = memory_budget(); ref = reference(); rec = frozen_reconstruction(256)
    b, ph = contract['probes'][0]; powers = np.asarray(contract['fNL_coefficient_scaling'])[None, :]
    OUT.mkdir()
    save_json(OUT/'CONTRACT.json', dict(immutable_sources=frozen, actual_job=job,
        scope='PN_same_leg and NN only; all44 original bins and both original grids',
        probe_b1=b, probe_p_halo=ph, original_scaled_tolerance=contract['scaled_tolerance'],
        memory_budget=budget, required_free_memory_bytes=4*1024**3,
        original_full_shell_denominator=True, physical_p_is_not_calibrated=True,
        actual_STO_closure=False, PN_cross_independent_all44_grid_oracle=False, MCMC_released=False))
    state = dict(state='RUNNING', pid=os.getpid(), host=socket.gethostname(), CPU_affinity=[12],
        starttime_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        started_UTC=datetime.now(timezone.utc).isoformat(), completed_grid_probes=0)
    def update(**values):
        state.update(values, updated_UTC=datetime.now(timezone.utc).isoformat())
        save_json(OUT/'STATUS.json', state)
    update(); rows = []; start = time.monotonic()
    try:
        for position, index in enumerate(contract['indices']):
            expected = evaluate_independent(coefficients[..., position], b, ph,
                contract['coordinate_centers_and_scales'])[1:]
            for mesh in contract['meshes'][str(index)]:
                guard()
                while True:
                    used = int(Path(budget['current_file']).read_text())
                    if budget['limit_bytes']-used >= 4*1024**3:
                        break
                    update(state='WAITING_FOR_ACTUAL_LOGIN_MEMORY_HEADROOM', used_memory_bytes=used)
                    time.sleep(5); guard()
                update(state='RUNNING', bin_index=index, mesh=mesh)
                field = PoissonShiftFFTProjection(mesh, ref['edges'], ref['shell_pairs'][index], b,
                    contract['f_growth'], p_halo=ph, reconstruction=rec)
                if field.denominator != counts[index]:
                    raise ValueError('actual original B estimator denominator changed')
                scalar = scalar_oracle(field); integer, info = integer_count_oracle(field)
                scale = np.maximum(1., np.max(np.abs(integer)*powers, axis=1, keepdims=True))
                ce = float(np.max(np.abs(expected-integer)*powers/scale))
                se = float(np.max(np.abs(scalar-integer)*powers/scale))
                row = dict(bin_index=index, mesh=mesh, coefficient_design=expected.tolist(),
                    scalar_integral_design=scalar.tolist(), independent_integer_design=integer.tolist(),
                    coefficient_scaled_error=ce, scalar_scaled_error=se, counts=info,
                    cache_peak_bytes=field.cache_peak_bytes)
                save_json(OUT/('bin%03d_mesh%d.json'%(index, mesh)), row)
                rows.append(row); update(completed_grid_probes=len(rows))
                print(json.dumps(dict(bin_index=index, mesh=mesh, coefficient_scaled_error=ce,
                    scalar_scaled_error=se, completed_grid_probes=len(rows))), flush=True)
                del field; gc.collect()
                if max(ce, se) > contract['scaled_tolerance']:
                    raise ValueError('original fNL-scaled numerical gate failed; no clipping or tolerance change')
        guard()
        if len(rows) != 88 or len({r['bin_index'] for r in rows}) != 44:
            raise ValueError('all original44 and both grids required')
        result = dict(status='ALL44_DUAL_GRID_PN_SAME_NN_INTEGER_SCALAR_PASS_NOT_STO_CLOSURE',
            created_UTC=datetime.now(timezone.utc).isoformat(), actual_job=job,
            rows=rows, bins=44, grid_probes=88, operators=['PN_same_leg', 'NN'],
            maximum_scaled_error=max(max(r['coefficient_scaled_error'], r['scalar_scaled_error']) for r in rows),
            original_scaled_tolerance=contract['scaled_tolerance'], immutable_sources=frozen,
            CPU_affinity=[12], elapsed_seconds=time.monotonic()-start,
            full44_PN_same_NN_scalar_grid_oracle_executed=True,
            full44_PN_cross_independent_grid_oracle_executed=False,
            physical_p_calibrated=False, actual_STO_noise_validated=False,
            formal_zero_atom_choice_validated=False, production_model_modified=False, MCMC_released=False, IR=False)
        save_json(OUT/'AUDIT.json', result)
        update(state='COMPLETE')
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows', 'immutable_sources')}, indent=2), flush=True)
    except Exception:
        update(state='FAILED', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
