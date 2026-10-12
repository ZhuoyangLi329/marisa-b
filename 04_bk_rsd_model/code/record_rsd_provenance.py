"""模块大纲：交付代码哈希；native 库匹配；实际输出与几何验收；禁止 PNG。"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import numpy as np

PART=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    code={}
    for p in sorted((PART/'code').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            if p.suffix=='.py':ast.parse(p.read_text())
            code[str(p.relative_to(PART))]=sha(p)
    native=json.loads((PART/'outputs/TREE_RSD_NATIVE_AUDIT.json').read_text())
    assert native['status']=='PASS'
    assert sha(PART/'work/build/libmarisa_tree_rsd.so')==native['library_sha256']
    out=PART/'outputs/pre_rsd_single_box';summary=json.loads((out/'SUMMARY.json').read_text())
    geometry=[]
    for name,stats in summary['samples'].items():
        with np.load(out/(name+'.npz')) as data:
            assert np.array_equal(data['exact_counts'],np.rint(data['exact_counts']))
            assert (data['exact_counts']>0).all()
            assert data['design_GLQ'].shape==(10,3,3)
            assert all(np.isfinite(data[k]).all() for k in data.files)
            np.testing.assert_allclose(data['B_raw'],data['direct_B_raw'],rtol=2e-10,atol=.005)
            geometry.append(data['exact_counts'].tolist())
        assert stats['max_P_mesh_change_in_diag_sigma']<.001
        assert stats['max_Braw_mesh_change_in_diag_sigma']<.001
        meta_path=PART/'outputs/pre_rsd_z0p5_losz'/name/'real000/metadata.json'
        meta=json.loads(meta_path.read_text())
        assert meta['data_has_rsd'] and not meta['reconstruction_applied']
        assert meta['exact_existing_position_and_mass_row_match']
        assert meta['fNL_truth']==stats['fNL_truth_fixed']
        assert sha(meta_path.parent/'positions_rsd_mpc_h.npy')==meta['output_positions_sha256']
    assert geometry[0]==geometry[1]
    assert not list((PART/'outputs').rglob('*.png'))
    files={str(p.relative_to(PART)):sha(p) for p in out.iterdir() if p.is_file()}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'numerical_delivery_status':'PASS',
        'scientific_status':'provisional single-box model diagnostic; fNL truth fixed, no ensemble covariance/recovery',
        'code_sha256':code,'model_outputs_sha256':files,'native_library_sha256':native['library_sha256'],
        'input_sample_position_hashes_verified':True,'no_PNG':True,'syntax_parse_all_python':'PASS',
        'matching_exact_counts':geometry[0]}
    (PART/'outputs/DELIVERY_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DELIVERY AUDIT PASS: '+str(PART/'outputs/DELIVERY_AUDIT.json'))

if __name__=='__main__':main()
