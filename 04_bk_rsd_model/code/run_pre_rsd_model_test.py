"""模块大纲：单盒 P0/P2；strict B0；离散有向 native 投影；固定真值诊断。

本例不使用 real-space covariance，不作 MCMC，不宣称 fNL 无偏验收。
共享 strict FFT estimator 已通过独立 brute-force 测试；Gaussian/PNG 源项均
使用真实 Cartesian 闭合三角形，不把连续 monopole 或旧两腿 B000 混进来。
"""
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from tree_png_rsd_native import NativeTreeRSD,HaloBiasV1,RsdConfig
from periodic_rsd_measure import catalog_fourier,power_multipoles

PART=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PART/'code/vendor'))
import quijote_strict_triangle_fft_estimator as fft

from rsd_bonly_common import TABLE, TABLE_HASH, F_GROWTH, P_CAL, bias, load_spectrum
EDGES=np.array([.01,.025,.04,.055])

def pk_theory(n,edges,sample,b1,fnl,f_growth):
    axis=2*np.pi*np.fft.fftfreq(n,d=1000./n)
    kx,ky,kz=np.meshgrid(axis,axis,axis,indexing='ij',sparse=True)
    k=np.sqrt(kx*kx+ky*ky+kz*kz)
    output=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(k>=lo)&(k<hi);kk=k[mask];mu=np.broadcast_to(kz,k.shape)[mask]/kk
        p,m=sample(kk);z1=b1+f_growth*mu**2+fnl*2*1.686*(b1-P_CAL)/m
        value=z1*z1*p
        output.append([np.mean(value),np.mean(5*.5*(3*mu**2-1)*value)])
    return np.array(output)

def fit_b1(measured,n,edges,sample,fnl):
    result=least_squares(lambda x:((pk_theory(n,edges,sample,x[0],fnl,F_GROWTH)-measured[:,2:4])/measured[:,4:6]).ravel(),[2.],bounds=(.5,4.))
    return float(result.x[0])

def exact_project(geometry,field,sample,b1):
    """源项和测量投影同一个 ordered triangle list；块化限制内存。"""
    n=geometry.meshsize;axis=np.rint(np.fft.fftfreq(n)*n).astype(int)
    mode_grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    modes=[mode_grid[geometry.shell_index==i] for i in range(geometry.nshell)]
    native=NativeTreeRSD();rsd=RsdConfig(F_GROWTH)
    bs=[bias(b1),bias(b1,1.),bias(b1,0.,1.)]
    # 额外 f_growth=0 在同一真实 lattice 上给出控制，不重新拟合模型。
    projected=[];direct_b=[];pair_shot=[];counts=[]
    kf=2*np.pi/geometry.boxsize_mpc_h
    for shells,expected in zip(geometry.triangles,geometry.triangle_counts):
        i,j,k=map(int,shells);sum_raw=np.zeros((4,39));sum_b=0j;sum_pair=0.;count=0
        for start in range(0,len(modes[i]),24):
            aa=modes[i][start:start+24];bb=modes[j]
            first=np.repeat(aa,len(bb),axis=0);second=np.tile(bb,(len(aa),1));third=-first-second
            # 本例 all legs <=.055 << Nyquist，绝不做 modulo alias 闭合。
            inside=np.all(np.abs(third)<n//2,axis=1)
            labels=np.full(len(third),-1,dtype=int)
            labels[inside]=geometry.shell_index[tuple((third[inside]%n).T)]
            selected=labels==k
            if not selected.any():continue
            vectors=np.stack([first[selected],second[selected],third[selected]],axis=1)
            norms=np.linalg.norm(vectors,axis=2)*kf;p,m=sample(norms)
            vv=np.ascontiguousarray(vectors*kf,dtype=float)
            for a,b in enumerate(bs):sum_raw[a]+=native.evaluate(vv,p,m,b,rsd).raw.sum(axis=0)
            sum_raw[3]+=native.evaluate(vv,p,m,bs[0],RsdConfig(0.)).raw.sum(axis=0)
            vals=field[tuple((vectors%n).transpose(2,0,1))]
            sum_b+=np.prod(vals,axis=1).sum()
            sum_pair+=np.sum(np.abs(vals)**2)
            count+=len(vectors)
        assert count==expected,(shells,count,expected)
        projected.append(sum_raw/count);direct_b.append(geometry.volume_mpc3_h3**2*sum_b.real/count)
        pair_shot.append(geometry.volume_mpc3_h3*sum_pair/count)
        counts.append(count)
        print(f'exact native shells={shells.tolist()} count={count}',flush=True)
    projected=np.array(projected)
    return {'design_GLQ':np.stack([projected[:,0,:3],projected[:,1,:3]-projected[:,0,:3],
                                  projected[:,2,:3]-projected[:,0,:3]],axis=1),
            'realspace_base_GLQ':projected[:,3,:3],
            'gaussian_stochastic':projected[:,0,36:39],
            'direct_B_raw':np.array(direct_b),'pair_P_raw_sum':np.array(pair_shot),'exact_counts':np.array(counts)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meshsize',type=int,default=128)
    parser.add_argument('--cosmologies',nargs='+',default=['fiducial','LC_p'])
    args=parser.parse_args();t0=time.time();sample=load_spectrum()
    geometry=fft.build_geometry(args.meshsize,1000.,EDGES)
    out=PART/'outputs/pre_rsd_single_box';out.mkdir(parents=True,exist_ok=True)
    summary={'status':'single_box_diagnostic','geometry':'strict three-leg angle-average B0; not legacy two-leg TriPoSH B000',
        'meshsize':args.meshsize,'B_edges_h_Mpc':EDGES.tolist(),'all_legs_kmax_h_Mpc':float(EDGES[-1]),
        'P_edges_h_Mpc':np.arange(.01,.061,.01).tolist(),'f_growth':F_GROWTH,'p_fixed':P_CAL,
        'IR':'off: strict tree source test','FoG':'off','loops':'off','counterterms':'off',
        'covariance':'none; displayed Gaussian diagonal errors are approximate single-box diagnostics',
        'finite_PNG_stochastic_closure':'off; analytic unit-weight point-catalog self-pair/triple subtraction; CIC approximation controlled by mesh convergence',
        'fNL_inference':'not performed; sample truth fixed while profiling only b2_native and bK2',
        'spectrum_path':str(TABLE),'spectrum_sha256':TABLE_HASH,'samples':{}}
    pedges=np.arange(.01,.061,.01)
    for cosmo in args.cosmologies:
        source=PART/'outputs/pre_rsd_z0p5_losz'/cosmo/'real000'
        meta=json.loads((source/'metadata.json').read_text());nbar=meta['nbar_h3_Mpc3'];truth=meta['fNL_truth']
        position=np.load(source/'positions_rsd_mpc_h.npy');field=catalog_fourier(position,args.meshsize)
        realfield=catalog_fourier(np.load(source/'positions_real_mpc_h.npy'),args.meshsize)
        pk=power_multipoles(field,pedges,nbar);pkreal=power_multipoles(realfield,pedges,nbar)
        fine=catalog_fourier(position,2*args.meshsize)
        pkfine=power_multipoles(fine,pedges,nbar)
        b1=fit_b1(pk,args.meshsize,pedges,sample,truth)
        pmodel=pk_theory(args.meshsize,pedges,sample,b1,truth,F_GROWTH)
        filtered=fft.filter_normalized_fourier(field,geometry)
        braw=fft.measure_bispectrum(filtered,filtered,filtered,geometry)
        finegeom=fft.build_geometry(2*args.meshsize,1000.,EDGES)
        assert np.array_equal(finegeom.triangle_counts,geometry.triangle_counts)
        finefiltered=fft.filter_normalized_fourier(fine,finegeom)
        bfine=fft.measure_bispectrum(finefiltered,finefiltered,finefiltered,finegeom)
        del fine,finefiltered,finegeom,filtered,realfield
        exact=exact_project(geometry,field,sample,b1)
        np.testing.assert_allclose(braw,exact['direct_B_raw'],rtol=2e-10,atol=.005)
        # 单位权重同 catalog 的 factorial auto B：Braw-sum(Praw)/n+2/n²。
        # 这是理想 point Fourier field 的精确关系；CIC 插值误差需单独验证收敛。
        bdata=braw-exact['pair_P_raw_sum']/nbar+2./nbar**2
        # Gaussian 对角方差只用来展示和稳定 profile，不报告正式 p-value。
        sym=np.array([6 if i==j==k else (2 if i==j or j==k else 1) for i,j,k in geometry.triangles])
        pp=[]
        for km in geometry.k_mean_h_mpc:
            p,m=sample(km);bp=2*1.686*(b1-P_CAL)
            z=b1+truth*bp/m
            pp.append((z*z+2*z*F_GROWTH/3+F_GROWTH**2/5)*p+1/nbar)
        sigma=np.sqrt(sym*1e9*np.prod(pp,axis=1)/geometry.triangle_counts)
        response=exact['design_GLQ']@np.array([1.,truth,truth**2])
        parameters=np.linalg.lstsq(response[:,1:]/sigma[:,None],(bdata-response[:,0])/sigma,rcond=None)[0]
        fitted=response[:,0]+response[:,1:]@parameters
        residual=(bdata-fitted)/sigma
        stats={'N_halo':meta['n_halo'],'fNL_truth_fixed':truth,'b1_from_P0_P2':b1,
            'b2_native_profile':float(parameters[0]),'bK2_profile':float(parameters[1]),
            'b2_EFT_profile':float(parameters[0]+4*parameters[1]/3),
            'B_diagonal_diagnostic_chi2':float(residual@residual),'B_bins':len(bdata),'profile_parameters':2,
            'max_P_mesh_change_in_diag_sigma':float(np.max(np.abs(pkfine[:,2:4]-pk[:,2:4])/pk[:,4:6])),
            'max_Braw_mesh_change_in_diag_sigma':float(np.max(np.abs(bfine-braw)/sigma)),
            'max_fft_vs_direct_Braw_relative_error':float(np.max(np.abs(braw-exact['direct_B_raw'])/np.maximum(1,np.abs(braw)))),
            'PK_diagonal_diagnostic_chi2':float(np.sum(((pmodel-pk[:,2:4])/pk[:,4:6])**2)),
            'B_source_response_fNL100_over_G_median':float(np.median((100*exact['design_GLQ'][:,0,1]+1e4*exact['design_GLQ'][:,0,2])/exact['design_GLQ'][:,0,0]))}
        summary['samples'][cosmo]=stats
        np.savez_compressed(out/(cosmo+'.npz'),pk=pk,pkreal=pkreal,pkfine=pkfine,pmodel=pmodel,
            triangles=geometry.triangles,kmean=geometry.k_mean_h_mpc,B_raw=braw,B_factorial=bdata,
            B_fitted=fitted,B_sigma_diag=sigma,B_raw_fine=bfine,profile=parameters,**exact)
        print(json.dumps(stats,indent=2),flush=True)
    summary['elapsed_seconds']=time.time()-t0
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
