"""模块大纲：只读实空间矩阵/认证索引/冻结似然；导出 RSD 对齐依据。

不重跑模型，不改实空间源数据，不提交作业；输出仅为 binning 审计。
认证表从现行 scan.py 的 AST 读取，不手抄、不加载历史复杂 runtime。
"""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

PART=Path(__file__).resolve().parents[1]
SOURCE=PART.parent/'03_bk_model/envclass_b'
OUT=PART/'outputs/realspace_binning_reference'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    scan=SOURCE/'vendor/py/scan.py'; parsed=ast.parse(scan.read_text())
    for node in parsed.body:
        if isinstance(node,ast.Assign):
            names=[n.id for n in node.targets if isinstance(n,ast.Name)]
            if 'CERTIFIED_INDICES' in names: certified=np.asarray(ast.literal_eval(node.value.args[0]),dtype=int)
            if 'EXPECTED_COUNTS' in names: expected=ast.literal_eval(node.value)
    matrix=SOURCE/'vendor/data/STANDARD_FNL250_MATRIX.npz'
    with np.load(matrix,allow_pickle=False) as data:
        k=data['B000_k'];pair_edges=data['B000_k_edges'];nmodes=data['B000_nmodes']
    with np.load(SOURCE/'vendor/data/MARISA_FNL250_MATRIX.npz') as data:
        np.testing.assert_allclose(data['B000_k'],k,atol=2e-6,rtol=0)
        np.testing.assert_array_equal(data['B000_k_edges'],pair_edges)
    edges=np.unique(pair_edges);pairs=np.asarray([(i,j) for i in range(15) for j in range(i,15)])
    np.testing.assert_array_equal(pair_edges,edges[np.stack((pairs,pairs+1),axis=-1)])
    # 中心只是说明原测量的请求参数；拟合筛选实际使用测得的 kavg，不用中心或边缘替代。
    centers=np.linspace(.01,.30,15);reconstructed=np.r_[0.,(centers[:-1]+centers[1:])/2,centers[-1]+(centers[1]-centers[0])/2]
    np.testing.assert_allclose(edges,reconstructed,rtol=0,atol=1e-15)
    cuts=sorted(set(expected)|{.04,.055,.07})
    selection={}
    for cut in cuts:
        indices=certified[np.max(k[certified],axis=1)<=cut+1e-12]
        if cut in expected:assert len(indices)==expected[cut]
        selection[str(cut)]={'count':len(indices),'indices':indices.tolist(),
                             'shell_pairs':pairs[indices].tolist(),'measured_kavg_pairs':k[indices].tolist()}
    contracts=[]
    for tag in ('paware_pb_mcmc_k012_k014_v1','paware_pb_ir_brecfix_k012_k014_v2'):
        for directory in sorted((SOURCE/'outputs'/tag).iterdir()):
            if not (directory/'contract.json').exists() or not (directory/'likelihood.npz').exists():continue
            meta=json.loads((directory/'contract.json').read_text())
            cut=meta['kmax_B']
            with np.load(directory/'likelihood.npz') as data:
                indices=data['indices'];dimension=len(data['target'])
            np.testing.assert_array_equal(indices,selection[str(cut)]['indices'])
            assert dimension==len(indices)+12
            contracts.append({'path':str(directory/'contract.json'),'sha256':sha(directory/'contract.json'),
                              'B_kmax':cut,'B_bins':len(indices),'P_bins_excluded_from_RSD_plan':12,
                              'indices_match':True,'data_dimension_P_plus_B':dimension})
    with np.load(SOURCE/'vendor/data/pre_tree_kmax0p20.npz') as data:
        projection={key:np.asarray(data[key]).tolist() for key in ('nmu','nradial','kfund_h_mpc','finite_box_projection_version')}
    value={'status':'PASS','scope':'binning reference, not an RSD model/data validation',
           'sources':{'matrix':{'path':str(matrix),'sha256':sha(matrix)},'scan':{'path':str(scan),'sha256':sha(scan)}},
           'observable':'Sugiyama B000(k1,k2), two radial shell legs; third closing leg unrestricted',
           'full_measured_bins':120,'radial_shells':15,'edges_h_Mpc':edges.tolist(),'requested_centers_h_Mpc':centers.tolist(),
           'radial_step_h_Mpc':float(centers[1]-centers[0]),'mask':'k1-shell <= k2-shell, upper triangle including diagonal',
           'selection_rule':'CERTIFIED_INDICES then max(measured kavg1,kavg2)<=cut+1e-12; NOT bin upper edge; NOT all-three-leg kmax',
           'certified_indices':certified.tolist(),'index000_measured_but_not_fitted':0 not in certified,
           'cut_selections':selection,'frozen_contracts_checked':contracts,'realspace_theory_projection':projection,
           'finite_box_rule':'k1,k2,k3>=2pi/L; excluded configurations contribute zero, preserve full-shell angular denominator',
           'RSD_alignment_required':['same exact radial edges','same shell-pair order and certified subset',
                                     'same two-leg observable and cutoff semantics','correct LOS/orientation average of RSD sources',
                                     'RSD ensemble covariance independently measured; do not reuse realspace covariance'],
           'warning':'at old proposed cuts .04/.055/.07/.08 only 2/5/5/9 B data points; six-parameter MCMC can be severely prior dominated',
           'paused_jobs':[59156128,59156137,59156138],
           'previous_strict_plan':'superseded before pilot/measurement/template/MCMC; only geometry128/256 completed'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'AUDIT.json').write_text(json.dumps(value,indent=2)+'\n')
    np.savez_compressed(OUT/'REALSPACE_BINNING.npz',B000_k=k,B000_k_edges=pair_edges,B000_nmodes=nmodes,
                        shell_pairs=pairs,edges=edges,certified_indices=certified)
    old=PART/'outputs/pre_rsd_bonly_v1'
    if old.exists():
        (old/'SUPERSEDED_BY_REALSPACE_BINNING_REVIEW.json').write_text(json.dumps({
            'status':'SUPERSEDED_NOT_FOR_PRODUCTION','reason':'user requires realspace fit point density and reference review',
            'superseded_contract_sha256':sha(old/'contract.json'),'audit_path':str(OUT/'AUDIT.json'),
            'data_measurement_started':False,'MCMC_submitted':False},indent=2)+'\n')
    print(json.dumps({'status':value['status'],'bin_width':value['radial_step_h_Mpc'],
                      'counts':{key:val['count'] for key,val in selection.items()},'frozen_contracts_checked':len(contracts)},indent=2))

if __name__=='__main__':main()
