#!/usr/bin/env python3
"""生成 IR 实现、缓存身份与科学输出的可复查清单。

模块大纲：核对模板内容哈希；保存内容寻址的 IR 配置键；记录源码与输出；
不改写任何理论、数据、拟合或原冻结结果。
"""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'halo/outputs/local_png_tree_ir_20260930'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return {'path':str(path),'sha256':sha(path),'bytes':Path(path).stat().st_size}

def main():
    source_names=[
        'marisa_b_core/Makefile','marisa_b_core/src/marisa_b/tree_png_ir.h',
        'marisa_b_core/src/marisa_b/tree_png_ir.cpp','marisa_b_core/src/marisa_b/tree_ir_lattice_nodes.cpp',
        'marisa_b_core/scripts/build_tree_png_ir.sh','marisa_b_core/python/marisa_b/tree_png_ir_native.py',
        'marisa_b_core/python/marisa_b/tree_png_ir_resummation.py',
        'marisa_b_core/tests/eft_v2/test_tree_png_ir_native.cpp',
        'marisa_b_core/tests/python/test_tree_png_ir_resummation.py',
        'marisa_b_core/tests/python/test_tree_png_ir_binding.py',
        'marisa_b_core/docs/TREE_PNG_IR_NATIVE.md','marisa_b_core/docs/TREE_LEVEL_PNG_IR_RESUMMATION_PLAN.md',
        'halo/code/inspect_tree_ir_inputs.py','halo/code/prepare_tree_png_ir_templates.py',
        'halo/code/run_tree_png_ir_kmax_scan.py','halo/code/audit_tree_png_ir_scan.py',
        'halo/code/write_tree_png_ir_manifest.py']
    caches={}
    for metadata_path in sorted(OUT.glob('tree_ir_templates_*.json')):
        metadata=json.loads(metadata_path.read_text())
        assert sha(metadata['output']['path'])==metadata['output']['sha256'], 'template content hash drift'
        identity={k:metadata[k] for k in ('redshift','units','sigma2_by_kir','r_bao','smoothing_lambda_ln_k',
                                         'split','variants','terms','projection','lattice_nodes','inputs')}
        caches[metadata_path.name]={'ir_cache_key_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
                                    'identity':identity,'output':metadata['output']}
    audit=json.loads((OUT/'IR_SCIENTIFIC_AUDIT.json').read_text())
    assert audit['status']=='pass', 'scientific/numerical audit incomplete'
    low=json.loads((OUT/'ir_kmax_scan_primary.json').read_text())
    high=json.loads((OUT/'ir_kmax_scan_convergence.json').read_text())
    summary={'schema':'marisa-b-local-png-tree-ir-kmax-summary-v1',
             'scope':'pre-reconstruction B000, finite local-PNG through fNL^2, tree-only',
             'task_repository':str(ROOT.parent),
             'primary_decisions':{k:v['largest_contiguous_valid_kmax'] for k,v in low['summary'].items()},
             'converged_summary':high['summary'],'numerical_audit_gates':audit['gates'],
             'post_boundary':'fixed-brec/CIC-off algebra tested; existing adaptive/CIC inference not promoted'}
    (OUT/'RESULTS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    destination=OUT/'TREE_PNG_IR_CHANGE_MANIFEST.json'
    payload={'schema':'marisa-b-tree-png-ir-change-manifest-v1',
             'created_utc':datetime.now(timezone.utc).isoformat(),
             'source_files':[record(ROOT/name) for name in source_names],
             'unchanged_native_reference':[record(ROOT/'marisa_b_core/src/marisa_b'/name) for name in ('marisa_b_native.cpp','marisa_b_native.h')],
             'original_makefile':record(ROOT/'halo/work/tree_ir_20260930_originals/Makefile'),
             'ir_cache_identities':caches,
             'output_files':[record(p) for p in sorted(OUT.iterdir()) if p.is_file() and p!=destination],
             'frozen_historical_outputs_modified':False,
             'test_evidence':record(OUT/'native_validation.json'),
             'results_summary':record(OUT/'RESULTS_SUMMARY.json')}
    destination.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'manifest':str(destination),'source_files':len(source_names),
                      'output_files':len(payload['output_files']),'status':'pass'},indent=2))

if __name__=='__main__':main()
