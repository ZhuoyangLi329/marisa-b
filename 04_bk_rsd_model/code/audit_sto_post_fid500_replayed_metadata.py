"""模块大纲：同一500份原STO/FoF；仅real127消费已证明同payload的配方重放记录。

复用完整payload/质量/行身份/RSD审计；原real127 diagnostics读取等待保留为历史。
重放STO副本仅用于metadata证据，所有500份weights仍读取原文件。来源逐行明确保存。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import audit_sto_post_fid500_raw_sources as raw_audit

original = raw_audit.original
ROOT = original.ROOT
OUT = ROOT / 'fid0_499_raw_source_audit_v3'
REPLAY = ROOT / 'input_recipe_replay_fid127_v2'


def verify_route():
    record = json.loads(original.read_blob(REPLAY / 'AUDIT.json'))
    if (record['status'] != 'EXISTING_REAL127_STO_PAYLOAD_REPRODUCED_FROM_SAME_RECIPE_AND_EXISTING_DM'
            or not record['all_arrays_exactly_equal'] or not all(record['all_arrays_exactly_equal'].values())):
        raise ValueError('same-payload producer replay has not passed')
    contract = json.loads(original.read_blob(REPLAY / 'CONTRACT.json'))
    if (original.sha(original.read_blob(REPLAY / 'CONTRACT.json')) != record['contract_sha256']
            or contract['realization'] != 127 or contract['cosmology'] != 'fiducial'):
        raise ValueError('replay contract mismatch')
    old = original.SOURCE / 'fiducial/real127/env_lam1_gsr2_thr15'
    fresh = Path(record['fresh_diagnostics_path'])
    expected = record['original_payload_sha256']
    if (expected != record['regenerated_payload_sha256']
            or original.sha(original.read_blob(old / 'post_sto_halo.npz')) != expected
            or original.sha(original.read_blob(fresh.parent / 'post_sto_halo.npz')) != expected
            or original.sha(original.read_blob(fresh)) != record['fresh_diagnostics_sha256']):
        raise ValueError('verified replay/current payload or metadata drift')
    diag = json.loads(original.read_blob(fresh))
    if (any(diag['config'].get(k) != v for k,v in original.RECIPE.items())
            or diag['component_config']['method'] != 'nenv' or diag['component_config']['base_weight'] != 'number'):
        raise ValueError('replayed metadata recipe mismatch')
    return old / 'diagnostics.json', fresh, record


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--detach', action='store_true')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('single-core input audit required')
    old_diag, fresh_diag, replay = verify_route()
    if args.detach:
        launch = ROOT / 'fid0_499_raw_source_audit_v3.launch.json'
        log = ROOT / 'fid0_499_raw_source_audit_v3.log'
        if launch.exists() or log.exists() or OUT.exists(): raise RuntimeError('prior v3 audit requires inspection')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        record = {'host': socket.gethostname(), 'pid': process.pid, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
            'new_session': True, 'command': command, 'log': str(log),
            'created_UTC': datetime.now(timezone.utc).isoformat(),
            'metadata_evidence_for_real127': str(fresh_diag), 'original_weights_sha256': replay['original_payload_sha256']}
        with launch.open('x') as stream: json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True); return
    paths = [Path(__file__), REPLAY / 'AUDIT.json', REPLAY / 'CONTRACT.json',
             fresh_diag, fresh_diag.parent / 'post_sto_halo.npz']
    frozen = {str(p): original.sha(original.read_blob(p)) for p in paths}
    prior_out = raw_audit.OUT
    raw_audit.OUT = OUT
    try:
        with raw_audit.raw_source_audit_context():
            prior_read, prior_save = original.read_blob, original.save
            def read(path, **kwargs):
                return prior_read(fresh_diag if Path(path) == old_diag else path, **kwargs)
            def save(path, value):
                if path.name == 'CONTRACT.json': value['immutable_sources'].update(frozen)
                if path.name in ('CONTRACT.json', 'AUDIT.json'):
                    value.update(metadata_replay_role='real127 only; original cached STO payload used',
                        old_diagnostics_path_for_real127=str(old_diag),
                        consumed_diagnostics_path_for_real127=str(fresh_diag),
                        same_payload_replay_audit=str(REPLAY / 'AUDIT.json'),
                        original_payload_sha256_for_real127=replay['original_payload_sha256'],
                        all_production_source_weights_read_from_original_cache=True,
                        new_DM_downloaded=False, new_Nbody_realization_generated=False,
                        same_recipe_STO_audit_copy_generated=True,
                        original_production_input_payloads_modified=False)
                    if path.name == 'AUDIT.json':
                        value['new_DM_or_STO_generated_or_downloaded'] = True
                if path.parent.name == 'rows' and path.name == 'fiducial_real127.json':
                    value.update(diagnostics_original_path=str(old_diag), diagnostics_original_sha256=None,
                        diagnostics_consumed_path=str(fresh_diag),
                        diagnostics_provenance='unchanged producer/recipe replay with all arrays and entire STO SHA identical',
                        metadata_replay_audit_sha256=frozen[str(REPLAY / 'AUDIT.json')])
                return prior_save(path, value)
            original.read_blob, original.save = read, save
            try: original.main()
            finally: original.read_blob, original.save = prior_read, prior_save
    finally:
        raw_audit.OUT = prior_out


if __name__ == '__main__': main()
