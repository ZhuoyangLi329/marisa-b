"""模块大纲：独立Chebyshev递推回放全部660节点/88细网格探针和44-bin汇总。

验收已有source投影与系数文件，不重新生成模型、不校准p或代替post covariance。
物理p轴只是数值表示；按同一个原合同核验，不改变阈值或分箱集合。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

PART=Path(__file__).resolve().parents[1]
OUT=PART/'outputs/sto_marisa_post_rsd_mean100_v1/noir_paware_templates'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def basis(value,centre,radius,degree):
    x=(float(value)-centre)/radius; result=[1.,x]
    for i in range(2,degree+1):result.append(2*x*result[-1]-result[-2])
    return np.array(result[:degree+1])


def evaluate(coefficients,b,p,coordinates):
    bc,br=coordinates['b1'];pc,pr=coordinates['p']
    return np.einsum('i,j,ijab->ab',basis(b,bc,br,4),basis(p,pc,pr,2),coefficients)


def scaled_error(predicted,direct,powers):
    scale=np.maximum(1.,np.max(np.abs(direct)*powers,axis=1,keepdims=True))
    return float(np.max(np.abs(predicted-direct)*powers/scale))


def main():
    target=OUT/'INDEPENDENT_COEFFICIENT_AUDIT.json'
    if target.exists():raise RuntimeError('preserve the prior independent audit')
    report=json.loads((OUT/'COMPLETION_AUDIT.json').read_text())
    contract=json.loads((OUT/'CONTRACT.json').read_text())
    contract_sha=digest(OUT/'CONTRACT.json')
    if (report['status']!='ALL44_DETERMINISTIC_NOIR_P_AWARE_TEMPLATES_COMPLETE_NOT_LIKELIHOOD'
            or report['contract_sha256']!=contract_sha
            or digest(OUT/'DETERMINISTIC_COEFFICIENTS.npz')!=report['coefficients_sha256']):
        raise ValueError('complete aggregate and immutable contract required')
    indices=contract['indices']; powers=np.array(contract['fNL_range_coefficient_scaling'])[None,:]
    if len(indices)!=44 or len(set(indices))!=44 or len(report['rows'])!=44:raise ValueError('full original44 bins required')
    with np.load(OUT/'DETERMINISTIC_COEFFICIENTS.npz',allow_pickle=False) as z:
        coefficients=z['coefficients'].copy()
        np.testing.assert_array_equal(z['indices'],indices)
        if str(z['contract_sha256'])!=contract_sha:raise ValueError('aggregate contract marker differs')
    if coefficients.shape!=(5,3,3,3,44) or not np.isfinite(coefficients).all():raise ValueError('invalid tensor')
    coordinates=contract['coordinate_centers_and_scales']; rows=[]; nodes_seen=0; probes_seen=0
    source_rows={r['index']:r for r in report['rows']}
    for position,index in enumerate(indices):
        file=OUT/'bins'/f'bin{index:03d}.npz';meta=json.loads(file.with_suffix('.json').read_text())
        if meta['contract_sha256']!=contract_sha or digest(file)!=source_rows[index]['coefficients_sha256']:
            raise ValueError('individual bin artifact drift')
        with np.load(file,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['coefficients'],coefficients[...,position])
            design_nodes=z['node_designs'].copy()
        node_errors=[];probe_errors=[]
        for ib,b in enumerate(contract['b1_nodes']):
            for ip,p in enumerate(contract['physical_p_nodes']):
                node=json.loads((OUT/'nodes'/f'bin{index:03d}_b{ib}_p{ip}.json').read_text())
                if node['contract_sha256']!=contract_sha or node['b1']!=b or node['p_halo']!=p:raise ValueError('node provenance mismatch')
                direct=np.array(node['design']);np.testing.assert_array_equal(direct,design_nodes[ib,ip])
                node_errors.append(scaled_error(evaluate(coefficients[...,position],b,p,coordinates),direct,powers));nodes_seen+=1
        if len(meta['probes'])!=2:raise ValueError('both independent fine-grid probes required')
        for probe in meta['probes']:
            predicted=evaluate(coefficients[...,position],probe['b1'],probe['p_halo'],coordinates)
            direct=np.array(probe['direct']);error=scaled_error(predicted,direct,powers)
            if probe['mesh']!=contract['meshes'][str(index)][1]:raise ValueError('wrong validation grid')
            np.testing.assert_allclose(predicted,probe['predicted'],rtol=3e-13,atol=2e-5)
            probe_errors.append(error);probes_seen+=1
        row={'bin_index':index,'max_node_scaled_error':max(node_errors),'max_fine_probe_scaled_error':max(probe_errors)}
        if max(row.values())<0:raise ValueError('invalid diagnostic')
        if max(row['max_node_scaled_error'],row['max_fine_probe_scaled_error'])>contract['combined_nonnode_fine_grid_scaled_tolerance']:
            raise ValueError('original numerical gate failed')
        rows.append(row)
    if (nodes_seen,probes_seen)!=(660,88):raise ValueError('incomplete replay')
    result={'status':'ALL44_NOIR_COEFFICIENT_NODE_AND_FINE_GRID_REPLAY_PASS',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'bins':44,'nodes_replayed':nodes_seen,'fine_probes_replayed':probes_seen,
        'rows':rows,'max_node_scaled_error':max(r['max_node_scaled_error'] for r in rows),
        'max_fine_probe_scaled_error':max(r['max_fine_probe_scaled_error'] for r in rows),
        'original_scaled_tolerance':contract['combined_nonnode_fine_grid_scaled_tolerance'],
        'contract_sha256':contract_sha,'coefficients_sha256':report['coefficients_sha256'],
        'source_sha256':digest(__file__),'live_remote_source_hashes_checked_here':False,
        'physical_p_calibrated':False,'post_covariance_whitened_error_checked':False,'MCMC_released':False}
    with target.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
