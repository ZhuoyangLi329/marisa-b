"""模块大纲：实际 Marisa/CIC 算子对独立 numpy paint-FFT-readout；固定参数与符号。

只使用小型合成加权粒子，不生产模拟 catalog。通过此门仍不证明连续 sinc^4
近似包含实际 alias，也不证明 weighted shot noise 或物理 p 校准。
"""
from pathlib import Path
from itertools import product
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
import socket
import sys
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from jaxpower import ParticleField,MeshAttrs,create_sharding_mesh
from jaxrecon.zeldovich import estimate_particle_delta
import jaxrecon.zeldovich as zeldovich

PART=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PART.parent/'01_reconstruction_methods/marisa_gra'))
from png_recon_tools import MarisaPlaneParallelFFTReconstruction
import png_recon_tools


def cic_weights(position,n,box):
    u=np.asarray(position,dtype=float)*n/box;lower=np.floor(u).astype(int);frac=u-lower
    for bits in product((0,1),repeat=3):
        idx=(lower+bits)%n
        weight=np.prod(np.where(np.array(bits)[None,:],frac,1.-frac),axis=1)
        yield (idx[:,0],idx[:,1],idx[:,2]),weight


def paint(position,weights,n,box):
    out=np.zeros((n,n,n))
    for idx,w in cic_weights(position,n,box):np.add.at(out,idx,weights*w)
    return out


def read(mesh,position,box):
    return sum(w*mesh[idx] for idx,w in cic_weights(position,len(mesh),box))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rng=np.random.default_rng(2026100803);rows=[];all_pass=True
    n=16;box=128.;radius=10.;brec=2.2280546353635335;prec=1.2822;bp=2*1.686*(brec-prec)
    ak=np.geomspace(1e-4,10.,2048);av=1e-3/(1+(ak/.04)**2)
    positions=rng.uniform(.01,box-.01,size=(3000,3));weights=rng.uniform(.4,2.,size=3000)
    probes=rng.uniform(.01,box-.01,size=(701,3))
    for dtype in ('f8','f4'):
        pos=positions.astype(dtype);wt=weights.astype(dtype);probe=probes.astype(dtype)
        attrs=MeshAttrs(meshsize=n,boxsize=np.full(3,box,dtype=dtype),
                        boxcenter=np.full(3,box/2,dtype=dtype),dtype=dtype)
        with create_sharding_mesh(meshsize=n):
            field=ParticleField(jnp.asarray(pos),weights=jnp.asarray(wt),attrs=attrs)
            delta=estimate_particle_delta(field,smoothing_radius=radius)
            # 独立 full-complex Fourier oracle；实际代码使用其自有 r2c/c2r 流程。
            mass=paint(pos,wt,n,box);raw=mass/mass.mean()-1.
            axis=np.fft.fftfreq(n)*n*(2*np.pi/box)
            q=np.meshgrid(axis,axis,axis,indexing='ij',sparse=True);q2=sum(a*a for a in q)
            smooth=np.fft.fftn(raw)*np.exp(-.5*q2*radius**2)
            expected_delta=np.fft.ifftn(smooth).real
            d_actual=np.asarray(delta.value)
            delta_error=float(np.max(np.abs(d_actual-expected_delta))/max(1e-20,np.max(np.abs(expected_delta))))
            for fnlrec in (0.,100.):
                rec=MarisaPlaneParallelFFTReconstruction(delta,growth_rate=0.,bias=brec,los='z',halo_add=0,
                    png_alpha_k=ak,png_alpha_values=av,fnl_loc=fnlrec,png_p=prec,png_delta_c=1.686,bias_floor=1e-3)
                observed=np.asarray(rec.read_shifts(jnp.asarray(probe),field='disp'))
                newpos=np.asarray(rec.read_shifted_positions(jnp.asarray(probe),field='disp'))
                kval=np.sqrt(np.where(q2>0,q2,1.));alpha=np.interp(np.log10(kval),np.log10(ak),av)
                denom=brec+fnlrec*bp*alpha
                predicted=[]
                for a in range(3):
                    meshhat=1j*q[a]/np.where(q2>0,q2,1.)*smooth/denom
                    # 标准 even-grid Nyquist gradient 为零，避免非 Hermitian 导数。
                    ix=[slice(None)]*3;ix[a]=n//2;meshhat[tuple(ix)]=0.
                    psi=np.fft.ifftn(meshhat).real
                    predicted.append(read(psi,probe,box))
                predicted=np.column_stack(predicted)
                scale=max(1e-20,float(np.max(np.abs(predicted))))
                shift_error=float(np.max(np.abs(observed-predicted))/scale)
                position_error=float(np.max(np.abs(newpos-(probe-observed))))
                # RSD removal 不属于本算法；同一 instance 对 data/random 都读 disp。
                tolerance=3e-11 if dtype=='f8' else 3e-5
                passed=(delta_error<tolerance and shift_error<tolerance and position_error<(2e-12 if dtype=='f8' else 2e-5))
                all_pass=all_pass and passed
                rows.append({'dtype':dtype,'fNL_rec':fnlrec,'density_relative_max_error':delta_error,
                             'read_shift_relative_max_error':shift_error,'position_minus_shift_max_error':position_error,
                             'relative_tolerance':tolerance,'pass':passed})
    sources=[Path(__file__),Path(zeldovich.__file__),Path(png_recon_tools.__file__),
             Path(inspect.getfile(ParticleField)),Path(inspect.getfile(MeshAttrs))]
    payload={'status':'DISCRETE_OPERATOR_ORACLE_PASS' if all_pass else 'REQUIRES_INVESTIGATION',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),
        'CPU_affinity':sorted(os.sched_getaffinity(0)),'mesh':n,'boxsize':box,'smoothing_radius':radius,
        'b_rec':brec,'p_rec':prec,'reconstruction_growth_rate':0.,'data_field':'disp','random_field':'disp',
        'weighted_particles':len(positions),'independent_read_probes':len(probes),'rows':rows,
        'alpha_table':'synthetic smooth table, not physical calibration','sources':{str(p):sha(p) for p in sources},
        'catalog_production_released':False,'remaining':'actual spectral table parity, finite-grid alias, weighted noise and independent Fid500'}
    target=PART/'outputs/sto_marisa_post_rsd_mean100_v1/OPERATOR_AUDIT.json'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
    if not all_pass:raise SystemExit(1)


if __name__=='__main__':main()
