"""模块大纲：完整500份原输入核验；471同payload配方证据路由；准入只追加。

复用完整FoF质量/速度、STO zip、原权重及行身份审计。仅471 diagnostics
使用原producer/既有DM重放且16数组/整文件SHA相同的副本，旧v2记录不改。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import signal
import time
import migrate_sto_post_login_queue as lifecycle
import audit_sto_post_fid500_payload_complete as base

original = base.original
ROOT = base.ROOT
OUT = ROOT / 'fid0_499_raw_source_audit_routed_v3'
REPLAY = ROOT / 'input_recipe_replay_fid471'
ARRAYS = {'positions_mpc_h', 'weights', 'env_labels', 'mass_bin_labels', 'density_bin_labels',
          'component_labels', 'base_weights', 'solution_weight_by_env', 'solution_weight_by_mass',
          'solution_weight_by_density', 'solution_weight_by_component', 'component_names',
          'method', 'env_source', 'weight_source', 'split_half_labels'}


def validate_replay_report(report):
    equal = report.get('all_arrays_exactly_equal', {})
    if (report.get('status') != 'EXISTING_STO_PAYLOAD_REPRODUCED_FROM_SAME_RECIPE_AND_EXISTING_DM'
        or report.get('cosmology') != 'fiducial' or report.get('realization') != 471
        or set(equal) != ARRAYS or not all(v is True for v in equal.values())
        or report.get('original_payload_sha256') != report.get('regenerated_payload_sha256')
        or report.get('original_inputs_modified') is not False
        or report.get('new_DM_downloaded_or_Nbody_generated') is not False):
        raise ValueError('real471 full16-array original-payload replay not verified')


def route_read(previous, old, fresh, path, **kwargs):
    return previous(fresh if Path(path) == old else path, **kwargs)


def verify_replay():
    report = json.loads(original.read_blob(REPLAY / 'AUDIT.json'))
    validate_replay_report(report)
    contract_path = REPLAY / 'CONTRACT.json'
    contract = json.loads(original.read_blob(contract_path))
    if (original.sha(original.read_blob(contract_path)) != report['contract_sha256']
        or contract['realization'] != 471 or contract['cosmology'] != 'fiducial'):
        raise ValueError('real471 replay contract drift')
    for path, expected in contract['immutable_sources'].items():
        if original.sha(original.read_blob(path)) != expected:
            raise ValueError('real471 original producer/source drift: ' + path)
    source = original.SOURCE / 'fiducial/real471/env_lam1_gsr2_thr15'
    fresh = Path(report['fresh_diagnostics_path'])
    expected = report['original_payload_sha256']
    if (original.sha(original.read_blob(source / 'post_sto_halo.npz')) != expected
        or original.sha(original.read_blob(fresh.parent / 'post_sto_halo.npz')) != expected
        or original.sha(original.read_blob(fresh)) != report['fresh_diagnostics_sha256']):
        raise ValueError('actual original/fresh STO bytes or replay metadata drift')
    diag = json.loads(original.read_blob(fresh))
    if (any(diag['config'].get(k) != v for k, v in original.RECIPE.items())
        or diag['component_config']['method'] != 'nenv'
        or diag['component_config']['base_weight'] != 'number'):
        raise ValueError('replay metadata is not the selected GSR2 tracer recipe')
    return source / 'diagnostics.json', fresh, report


def stop_prior_reader():
    old, fresh, report = verify_replay()
    pid = 536207
    before = lifecycle.process(pid)
    expected_entry = str(lifecycle.PART / 'code/audit_sto_post_fid500_payload_complete.py')
    if before is None or expected_entry not in before['command']:
        raise ValueError('exact original v2 process not verified; do not signal a different PID')
    descriptors = [os.readlink(p) for p in Path('/proc', str(pid), 'fd').iterdir()]
    if str(old) not in descriptors:
        raise ValueError('v2 no longer waiting at verified real471 metadata; inspect new actual state')
    evidence = ROOT / 'fid0_499_v2_reader_termination_for_metadata471.json'
    if evidence.exists():
        raise ValueError('preserve existing strategy switch evidence; inspect process, not repeat stop')
    record = dict(state='PREPARED_FOR_VERIFIED_METADATA_STRATEGY_SWITCH', host=socket.gethostname(),
        before_process=before, before_fd_targets=descriptors,
        old_status=json.loads((ROOT/'fid0_499_raw_source_audit_routed_v2/STATUS.json').read_text()),
        replay_audit_sha256=original.sha(original.read_blob(REPLAY/'AUDIT.json')),
        original_STO_sha256=report['original_payload_sha256'], original_weights_not_replaced=True,
        old_diagnostics_path=str(old), new_same_payload_diagnostics_path=str(fresh),
        entry_sha256=original.sha(original.read_blob(Path(__file__))),
        reason='verified original-producer full-payload metadata route; not observation timeout',
        created_UTC=datetime.now(timezone.utc).isoformat())
    lifecycle.save(evidence, record)
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if lifecycle.process(pid) is None:
            record.update(state='PRIOR_V2_PID536207_CONFIRMED_ABSENT',
                          finished_UTC=datetime.now(timezone.utc).isoformat())
            lifecycle.save(evidence, record)
            print(json.dumps(dict(state=record['state'],evidence=str(evidence))),flush=True)
            return
        time.sleep(.1)
    record['state']='TERMINATION_REQUESTED_ORIGINAL_PROCESS_STILL_EXISTS'
    lifecycle.save(evidence,record)
    raise ValueError('prior process still exists; cannot start a second shared admission writer')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detach', action='store_true')
    ap.add_argument('--stop-prior-reader', action='store_true'); args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1 or socket.gethostname() != 'login26':
        raise ValueError('single-core original audit host required')
    if args.stop_prior_reader:
        if args.detach:raise ValueError('inspect stop result directly, do not detach it')
        stop_prior_reader();return
    # 旧v2不能在未来返回后写同一准入目录；实际PID须已退出，状态文件不够。
    try:
        os.kill(536207, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('original v2 reader still exists; no concurrent admission writer')
    old, fresh, replay = verify_replay()
    if args.detach:
        launch = ROOT / (OUT.name + '.launch.json'); log = ROOT / (OUT.name + '.log')
        if OUT.exists() or launch.exists() or log.exists():
            raise ValueError('previous v3 execution must be inspected, not replaced')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(host=socket.gethostname(), pid=proc.pid, command=command, log=str(log),
                      CPU_affinity=sorted(os.sched_getaffinity(0)), new_session=True,
                      created_UTC=datetime.now(timezone.utc).isoformat(),
                      prior_v2_PID536207_confirmed_absent=True)
        with launch.open('x') as f:
            json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(record), flush=True); return
    previous_read, previous_save, previous_out = original.read_blob, original.save, base.OUT
    paths = (Path(__file__), REPLAY / 'AUDIT.json', REPLAY / 'CONTRACT.json', fresh)
    frozen = {str(p): original.sha(previous_read(p)) for p in paths}

    def read(path, **kwargs):
        return route_read(previous_read, old, fresh, path, **kwargs)

    def save(path, value):
        if path.name == 'CONTRACT.json':
            value['immutable_sources'].update(frozen)
        if path.name in ('CONTRACT.json', 'AUDIT.json'):
            value.update(metadata_replay_realizations=[127, 293, 471],
                metadata_replay_audit_for_real471=str(REPLAY / 'AUDIT.json'),
                real471_original_STO_payload_sha256=replay['original_payload_sha256'],
                original_STO_weights_unchanged=True, new_DM_or_Nbody_generated=False,
                prior_v2_PID536207_confirmed_absent=True, prior_v2_read_wait_artifacts_preserved=True)
        if path.parent.name == 'rows' and value['realization'] == 471:
            value.update(diagnostics_original_path=str(old), diagnostics_original_sha256=None,
                diagnostics_consumed_path=str(fresh),
                metadata_replay_audit_sha256=frozen[str(REPLAY / 'AUDIT.json')],
                diagnostics_provenance='original producer/existing DM; all16 arrays and full STO SHA identical')
        return previous_save(path, value)

    original.read_blob, original.save, base.OUT = read, save, OUT
    sys.argv = [base.__file__]
    try:
        base.main()
    finally:
        original.read_blob, original.save, base.OUT = previous_read, previous_save, previous_out


if __name__ == '__main__':
    main()
