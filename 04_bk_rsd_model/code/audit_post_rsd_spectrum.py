"""模块大纲：实际重构 alpha 表与理论 1/M 独立核对；保存差异，不改旧谱表。"""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import sys
import numpy as np
from rsd_b000_contract import PART,load_spectrum,TABLE,digest
sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/marisa_gra'))
from png_recon_tools import compute_png_alpha_table


def main():
    k=np.geomspace(2*np.pi/1000.,.42,401)
    power,m=load_spectrum()(k)
    result=[];tables={}
    for n in (2048,8192):
        ak,av,meta=compute_png_alpha_table(kmin=1e-4,kmax=10.,nk=n,redshift=.5,cosmology='quijote')
        alpha=np.interp(np.log10(k),np.log10(ak),av)
        relative=alpha*m-1.
        result.append({'nk':n,'relative_alphaM_minus1_min':float(relative.min()),
                       'relative_alphaM_minus1_max':float(relative.max()),
                       'relative_alphaM_minus1_absmax':float(np.max(np.abs(relative))),
                       'metadata':asdict(meta)})
        tables[str(n)]=(ak,av,alpha)
    refinement=float(np.max(np.abs(tables['2048'][2]/tables['8192'][2]-1.)))
    payload={'created_UTC':datetime.now(timezone.utc).isoformat(),
        'status':'TABLES_MATCH' if result[-1]['relative_alphaM_minus1_absmax']<1e-6 else 'DISTINCT_RECONSTRUCTION_ALPHA_MUST_BE_EXPLICIT',
        'probe_k_range':[float(k.min()),float(k.max())],'probe_count':len(k),'rows':result,
        'alpha_interpolation_refinement_fraction_max':refinement,'physical_source_table':str(TABLE),
        'physical_source_table_sha256':digest(TABLE),'entry_sha256':digest(Path(__file__)),
        'old_tables_modified':False,'formal_calibration_performed':False}
    out=PART/'outputs/sto_marisa_post_rsd_mean100_v1';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'RECON_ALPHA_COMPARISON.npz',k_probe=k,transfer_M=m,
        alpha_k_2048=tables['2048'][0],alpha_values_2048=tables['2048'][1],
        alpha_k_8192=tables['8192'][0],alpha_values_8192=tables['8192'][1])
    payload['comparison_sha256']=digest(out/'RECON_ALPHA_COMPARISON.npz')
    (out/'SPECTRUM_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
