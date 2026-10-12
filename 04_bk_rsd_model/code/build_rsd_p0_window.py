"""模块大纲：从生产 jaxpower binner 导出实际 P window；不另写分箱算法。

仅保留低 k 格点的整数坐标/同一个 binner 的标签和 Hermitian 权重。
避免 CPU/GPU 或 float32/64 边界舍入导致 P 数据与模型 window 不一致。
"""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from jaxpower import MeshAttrs,BinMesh2SpectrumPoles
import jaxpower.mesh2 as source
from rsd_pb_common import OUT,p_geometry,p_lattice,digest,save_json
from rsd_b000_contract import load_spectrum

def main():
    _,edges=p_geometry();OUT.mkdir(parents=True,exist_ok=True)
    mattrs=MeshAttrs(meshsize=256,boxsize=np.ones(3,dtype='f4')*1000,
                    boxcenter=np.ones(3,dtype='f4')*500,dtype='f4')
    bin=BinMesh2SpectrumPoles(mattrs,edges=edges,ells=(0,))
    if len(bin.ibin)!=1:raise ValueError('unexpected mode oversampling')
    coords=mattrs.kcoords(kind='index',sparse=True);shape=np.broadcast_shapes(*[q.shape for q in coords])
    ibin=np.asarray(bin.ibin[0]).reshape(shape)
    # jaxpower 的 bincount 以 0/nbin+1 为范围外 sentinel；有效标签从1开始。
    mask=(ibin>=1)&(ibin<=12);labels=ibin[mask]-1
    w=np.ones(shape,dtype='f8') if bin.wmodes is None else np.asarray(bin.wmodes).reshape(shape)
    vectors=np.column_stack([np.rint(np.broadcast_to(np.asarray(q),shape)[mask]).astype('i2')
                             for q in coords])
    weights=np.asarray(w[mask],dtype=float)
    nmodes=np.asarray(bin.nmodes);k=np.asarray(bin.xavg)
    np.testing.assert_array_equal(np.bincount(labels,weights=weights,minlength=12),nmodes)
    first=OUT/'measurements_m256/fiducial_real000.npz'
    if first.exists():
        with np.load(first) as z:
            np.testing.assert_array_equal(nmodes,z['nmodes']);np.testing.assert_allclose(k,z['k'],rtol=1e-7)
    meta={'production_binner':'jaxpower.BinMesh2SpectrumPoles, same as shared.measure_pk',
          'production_mesh':256,'dtype':'float32','fft_backend':mattrs.fft_backend,
          'binner_source':str(Path(source.__file__)),'binner_source_sha256':digest(source.__file__),
          'export_entry_sha256':digest(Path(__file__)),'boundary_policy':'exact frozen production binner labels; no approximate shell repair'}
    np.savez_compressed(OUT/'P0_WINDOW.npz',vectors_integer=vectors,labels=labels,weights=weights,
                       nmodes=nmodes,k=k,edges=edges,metadata_json=np.array(json.dumps(meta)))
    t,_,_=p_lattice(edges)
    # 六个独立谱函数直接喂给原生产 binner，验证压缩 window 的每个方向权重。
    full_vectors=[np.asarray(q,dtype='f8')*(2*np.pi/1000.) for q in mattrs.kcoords(kind='index',sparse=True)]
    q2=sum(v*v for v in full_vectors);q=np.sqrt(q2);active=q>0;p=np.zeros_like(q);m=np.ones_like(q)
    p[active],m[active]=load_spectrum()(q[active]);u=np.divide(full_vectors[2]**2,q2,out=np.zeros_like(q2),where=active)
    values=(p,p*u,p*u*u,p/m,p*u/m,p/m**2);errors=[]
    for expected,value in zip(t,values):
        direct=np.asarray(bin(jnp.asarray(value)));np.testing.assert_allclose(direct,expected,rtol=4e-13,atol=1e-11)
        errors.append(float(np.max(np.abs(direct-expected)/np.maximum(np.abs(expected),1e-25))))
    print('PRODUCTION_P_WINDOW_PASS',json.dumps({'nmodes':nmodes.tolist(),'mode_count':len(labels),
          'max_relative_error':max(errors),'fft_backend':mattrs.fft_backend}),flush=True)
    save_json(OUT/'P0_WINDOW_AUDIT.json',{'status':'PASS','window_sha256':digest(OUT/'P0_WINDOW.npz'),
              'binner_source_sha256':digest(source.__file__),'entry_sha256':digest(Path(__file__)),
              'all_six_templates_against_original_production_operator':True,'max_relative_error':max(errors),
              'measurement_nmodes_match':first.exists(),'nmodes':nmodes.tolist(),'mode_count':len(labels)})

if __name__=='__main__':main()
