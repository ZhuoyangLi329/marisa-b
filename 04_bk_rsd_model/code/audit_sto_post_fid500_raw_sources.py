"""模块大纲：直接原FoF+STO输入的独立验收；不依赖冗余位置/质量导出缓存。

两个旧payload审计因已证实的周期端点误拒绝而终止；PID身份和退出均已核对，日志与源码快照保留。
本入口只改变是否额外交叉检查旧导出cache，完整复用原FoF质量选择、RSD转换、
STO配方、全部500份行身份和前100份既有哈希回归；不改主样本或任何源文件。
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
import audit_sto_post_fid500_payloads as original

ORIGINAL_SHA = '127444e44b9cef859364118d69cab524e6ad36c226c826971ff6e2530c46b30c'
OUT = original.ROOT / 'fid0_499_raw_source_audit_v2'


def omit_optional_export(path, *, max_bytes):
    """仅白名单内的冗余导出不参与；原FoF/STO/diagnostics仍必须全部读到。"""
    relative = Path(path).relative_to(original.adapter.original.OLD_POS)
    parts = relative.parts
    if (len(parts) != 3 or parts[0] != 'fiducial' or len(parts[1]) != 7
            or not parts[1].startswith('real') or not parts[1][4:].isdigit()
            or not 0 <= int(parts[1][4:]) < 500
            or parts[2] not in ('positions_mpc_h.npy', 'masses_msun_h.npy')
            or max_bytes != 128*1024*1024):
        raise ValueError('not an optional historical catalog export')
    return None


@contextmanager
def raw_source_audit_context():
    """单进程临时IO配置；退出后恢复原入口，源码和数值reader均不修改。"""
    previous = original.OUT, original.optional_blob, original.save
    paths = (Path(__file__), Path(__file__).with_name('test_post_rsd_raw_source_audit.py'))
    frozen = {str(path): original.sha(original.read_blob(path)) for path in paths}
    def save(path, value):
        if path.name == 'CONTRACT.json':
            value['immutable_sources'].update(frozen)
        if path.name in ('CONTRACT.json', 'AUDIT.json'):
            value.update(optional_historical_position_or_mass_export_checked=False,
                original_FoF_mass_selection_and_RSD_conversion_preserved=True,
                all_required_FoF_STO_recipe_and_row_identity_checks_preserved=True,
                old_partial_audit_artifacts_preserved=True,
                old_readers_terminated_for_confirmed_endpoint_validation_bug=[1838796, 1900874],
                audit_entry='audit_sto_post_fid500_raw_sources.py',
                input_strategy='original FoF bytes plus original STO payloads; no redundant exported position/mass caches')
        return previous[2](path, value)
    try:
        original.OUT = OUT; original.optional_blob = omit_optional_export; original.save = save
        yield
    finally:
        original.OUT, original.optional_blob, original.save = previous


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--detach', action='store_true')
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1: raise RuntimeError('single-core raw-input audit required')
    if original.sha(original.read_blob(original.__file__)) != ORIGINAL_SHA:
        raise ValueError('original full-payload audit changed')
    if args.detach:
        launch = original.ROOT / 'fid0_499_raw_source_audit_v2.launch.json'
        log = original.ROOT / 'fid0_499_raw_source_audit_v2.log'
        if launch.exists() or log.exists() or OUT.exists(): raise RuntimeError('prior raw-source audit requires inspection')
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
        with log.open('xb') as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        record = {'host': socket.gethostname(), 'pid': process.pid, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
            'new_session': True, 'log': str(log), 'command': command,
            'created_UTC': datetime.now(timezone.utc).isoformat(), 'original_optional_export_audit_restarted': False}
        with launch.open('x') as stream: json.dump(record, stream, indent=2); stream.write('\n')
        print(json.dumps(record), flush=True); return
    tests = unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_raw_source_audit')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful() or result.testsRun != 3: raise RuntimeError('raw-input IO policy and periodic endpoint tests failed')
    with raw_source_audit_context(): original.main()


if __name__ == '__main__': main()
