"""模块大纲：只读原始 FoF；逐行核对旧位置；物理速度转换；周期 LOS 位移。"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np

PART=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PART/'code/vendor'))
from readfof import FoF_catalog

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def redshift_positions(position,group_velocity,z,omega_m,boxsize,axis=2):
    """GroupVel*(1+z) 是 km/s；v/(a H_h) 是 Mpc/h，不再额外乘 h。"""
    hubble_h=100.*np.sqrt(omega_m*(1+z)**3+1.-omega_m)
    velocity=np.asarray(group_velocity,dtype=float)*(1+z)
    displacement=velocity[:,axis]*(1+z)/hubble_h
    result=np.asarray(position,dtype=float).copy()
    result[:,axis]=(result[:,axis]+displacement)%boxsize
    return result,velocity,displacement,hubble_h

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cosmologies',nargs='+',default=['fiducial','LC_p'])
    parser.add_argument('--realization',type=int,default=0)
    args=parser.parse_args()
    for cosmo in args.cosmologies:
        if cosmo not in ('fiducial','LC_p'):raise ValueError('only verified cosmologies supported')
        tag=f'real{args.realization:03d}'
        raw=Path('/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5')/cosmo/tag
        existing=Path('/pscratch/sd/l/lzy/quijote-png/halos/FoF_z0p5_mmin1e13_positions')/cosmo/tag
        fof=FoF_catalog(str(raw),3,read_IDs=False)
        mass=fof.GroupMass.astype(float)*1e10;mask=mass>=1e13
        position=fof.GroupPos[mask]/np.float32(1e3)
        assert np.array_equal(position,np.load(existing/'positions_mpc_h.npy')),'position row mismatch'
        assert np.array_equal(mass[mask],np.load(existing/'masses_msun_h.npy')),'mass row mismatch'
        rsd,velocity,displacement,hubble=redshift_positions(position,fof.GroupVel[mask],.5,.3175,1000.)
        assert np.isfinite(rsd).all() and np.isfinite(velocity).all()
        assert np.array_equal(rsd[:,:2],position[:,:2])
        out=PART/'outputs/pre_rsd_z0p5_losz'/cosmo/tag;out.mkdir(parents=True,exist_ok=True)
        for name,value in [('positions_rsd_mpc_h',rsd),('positions_real_mpc_h',position),
                           ('velocities_physical_km_s',velocity),('masses_msun_h',mass[mask])]:
            np.save(out/(name+'.npy'),value)
        metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'cosmology':cosmo,
            'realization':args.realization,'fNL_truth':0 if cosmo=='fiducial' else 100,
            'redshift':.5,'Omega_m0':.3175,'boxsize_mpc_h':1000.,'volume_Gpc_h3':1.,
            'los':[0,0,1],'axis':2,'stage':'pre_recon','data_has_rsd':True,
            'reconstruction_applied':False,'n_halo':len(rsd),'nbar_h3_Mpc3':len(rsd)/1e9,
            'mass_threshold_Msun_h':1e13,'f_growth':.7625790037720044,
            'velocity_rule':'v_physical_km_s=GroupVel*(1+z); ds_Mpc_h=v_physical*(1+z)/H_h',
            'H_h_km_s_Mpc_h':hubble,'exact_existing_position_and_mass_row_match':True,
            'displacement_los_rms_Mpc_h':float(np.sqrt(np.mean(displacement**2))),
            'displacement_los_min_max_Mpc_h':[float(displacement.min()),float(displacement.max())],
            'number_wrapped':int(np.count_nonzero((position[:,2]+displacement<0)|(position[:,2]+displacement>=1000))),
            'source_raw':str(raw/'groups_003/group_tab_003.0'),'source_raw_sha256':sha(raw/'groups_003/group_tab_003.0'),
            'source_positions_sha256':sha(existing/'positions_mpc_h.npy'),
            'output_positions_sha256':sha(out/'positions_rsd_mpc_h.npy'),
            'velocity_documentation':'https://quijote-simulations.readthedocs.io/en/latest/Examples/Reading_FoF.html'}
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__':main()
