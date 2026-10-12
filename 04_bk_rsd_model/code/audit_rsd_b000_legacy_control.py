"""模块大纲：旧 real-space 重测差异按壳层/切割分解；不更改任何数据或 gate。

相同 edges 不保证不同后端的 float32 边界分类 bitwise 一致；禁止混用旧 covariance。
"""
import numpy as np
from rsd_b000_contract import OUT,reference,selected,CUTS,save_json
from run_rsd_b000_inference import read_measurement

def main():
    ref=reference();idx=ref['certified_indices'];control=np.load(OUT/'REALSPACE_MEASUREMENT_CONTROL_fiducial.npz')
    sigma=np.std([read_measurement('fiducial',i)[0]['B'] for i in range(20)],axis=0,ddof=1)
    changed=control['B000_nmodes']!=ref['B000_nmodes'];pairs=ref['shell_pairs'];diagonal=np.flatnonzero(pairs[:,0]==pairs[:,1])
    shells=[]
    for i in diagonal:
        if not changed[i]:continue
        shell=int(pairs[i,0]);old=int(round(np.sqrt(ref['B000_nmodes'][i])));new=int(round(np.sqrt(control['B000_nmodes'][i])))
        shells.append({'shell':shell,'edges_h_Mpc':ref['edges'][shell:shell+2].tolist(),
                       'legacy_modes':old,'new_modes':new,'change':new-old})
    proximity=[]
    for a,b in zip(shells,shells[1:]):
        if b['shell']!=a['shell']+1:continue
        edge=ref['edges'][b['shell']];kf=2*np.pi/1000.;n2=int(round((edge/kf)**2));radius=int(np.sqrt(n2))
        axis=np.arange(-radius,radius+1);x,y,z=np.meshgrid(axis,axis,axis,indexing='ij')
        k=float(np.sqrt(n2)*kf);ulp=float(np.spacing(np.float32(edge)))
        proximity.append({'edge_h_Mpc':float(edge),'integer_squared_radius':n2,'modes_at_this_radius':int(np.count_nonzero(x*x+y*y+z*z==n2)),
                          'physical_k_h_Mpc':k,'edge_minus_physical_k':float(edge-k),'float32_ULP_h_Mpc':ulp,
                          'gap_in_float32_ULP':float(abs(edge-k)/ulp)})
    value={'status':'EXPLAINED_BOUNDARY_CLASSIFICATION_DIFFERENCE','changed_shells':shells,'boundary_proximity':proximity,
           'affected_two_leg_bins':int(changed.sum()),'unchanged_count_bins_max_control_shift':float(np.max(10*abs(control['delta'][idx[~changed[idx]]])/sigma[idx[~changed[idx]]])),
           'max_control_shift_by_kmax':{str(k):float(np.max(10*abs(control['delta'][selected(k)])/sigma[selected(k)])) for k in CUTS},
           'units':'sigma of mean100, estimated from 20 fiducial RSD pilot boxes; not an old-realspace likelihood significance',
           'interpretation':'72 modes transfer across an edge less than one float32 ULP from a lattice radius; supports backend boundary-classification explanation, not a bitwise GPU rerun proof',
           'scientific_use':'current CPU RSD data and double-lattice theory counts agree. Do not directly reuse high-k legacy values/covariance or attribute this difference to RSD/IR.',
           'data_changed_by_this_audit':False}
    save_json(OUT/'LEGACY_BOUNDARY_AUDIT.json',value);print(value,flush=True)

if __name__=='__main__':main()
