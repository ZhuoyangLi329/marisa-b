"""模块大纲：同字节暂存已验收target；调用原post数值入口；保存原始来源。

只绕开冗余旧导出cache的Lustre读取；质量选择、速度转换、重构和P/B实现
全部复用现有函数。完整target测量不等于MCMC科学验收或协方差放行。
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


PART = Path(__file__).resolve().parents[1]
ROOT = PART / 'outputs/sto_marisa_post_rsd_mean100_v1'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@contextmanager
def stage_source(row, read_blob):
    """对照既有完整身份审计；同字节暂存，不改原文件或选择。
    read_blob参数允许用轻量fixture独立测试，无需加载JAX。
    """
    original = Path(row['source'])
    payload = read_blob(original / 'post_sto_halo.npz')
    diagnostics = read_blob(original / 'diagnostics.json')
    raw = read_blob(Path(row['raw_fof']['source_raw']))
    for blob, expected in ((payload, row['source_sha256']),
                           (diagnostics, row['diagnostics_sha256']),
                           (raw, row['raw_fof']['source_raw_sha256'])):
        if sha(blob) != expected:
            raise ValueError('audited input payload changed; stop without substitution')
    with TemporaryDirectory(prefix='sto_post_target_', dir='/tmp') as temporary:
        root = Path(temporary)
        source = root / row['cosmology'] / f"real{row['realization']:03d}" / 'env_lam1_gsr2_thr15'
        source.mkdir(parents=True)
        (source / 'post_sto_halo.npz').write_bytes(payload)
        (source / 'diagnostics.json').write_bytes(diagnostics)
        yield root, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cosmology', choices=('fiducial', 'LC_p', 'LC_m'), required=True)
    ap.add_argument('--realization', type=int, required=True)
    args = ap.parse_args()
    if len(os.sched_getaffinity(0)) != 1 or not 0 <= args.realization < 100:
        raise ValueError('single-core audited target0..99 only')
    from post_rsd_blob_catalog import read_blob, catalog_from_blobs
    import run_sto_post_rsd_pilot as pilot
    audit_path = ROOT / 'VALIDATION_SOURCE_AUDIT.json'
    audit_blob = read_blob(audit_path)
    audit = json.loads(audit_blob)
    if audit['status'] != 'ALL_300_VALIDATION_SOURCE_IDENTITIES_PASS_NOT_PRODUCTION_RELEASE':
        raise ValueError('full target source audit missing')
    rows = [r for r in audit['source_rows'] if
            (r['cosmology'], r['realization']) == (args.cosmology, args.realization)]
    if len(rows) != 1:
        raise ValueError('target identity ambiguous')
    row = rows[0]
    previous_source, previous_catalog, previous_save = pilot.SOURCE, pilot.catalog, pilot.save_json
    with stage_source(row, read_blob) as (temporary_root, raw):
        def catalog(cosmology, realization):
            if (cosmology, realization) != (args.cosmology, args.realization):
                raise ValueError('unexpected catalog request')
            return catalog_from_blobs(cosmology, realization, raw,
                                      raw_source=row['raw_fof']['source_raw'])

        def save(path, value):
            if path.name == 'CONTRACT.json':
                value.update(source_STO=row['source'],
                    input_staging='identical audited bytes in local temporary storage; original numerical reader',
                    target_source_audit_sha256=sha(audit_blob),
                    IO_adapter_sha256=sha(Path(__file__).read_bytes()),
                    redundant_export_cache_used=False, original_inputs_modified=False)
            return previous_save(path, value)

        pilot.SOURCE, pilot.catalog, pilot.save_json = temporary_root, catalog, save
        sys.argv = [pilot.__file__, '--cosmology', args.cosmology, '--realization', str(args.realization),
                    '--recon-mesh', '256', '--measurement-mesh', '256', '--nran', '30',
                    '--self-contraction', 'matched', '--zero-atom-diagnostics', '--nested-randoms']
        try:
            pilot.main()
        except Exception:
            import traceback
            if pilot.PILOT_STATUS_PATH is not None:
                state = json.loads(pilot.PILOT_STATUS_PATH.read_text())
                state.update(state='FAILED', error=traceback.format_exc())
                previous_save(pilot.PILOT_STATUS_PATH, state)
            raise
        finally:
            pilot.SOURCE, pilot.catalog, pilot.save_json = previous_source, previous_catalog, previous_save


if __name__ == '__main__':
    main()
