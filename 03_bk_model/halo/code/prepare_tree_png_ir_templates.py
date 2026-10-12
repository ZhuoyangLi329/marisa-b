#!/usr/bin/env python3
"""完整 local-PNG tree IR 模板：共享 smooth split，逐三角形处理，再投影。

模块大纲
1. 冻结同红移 P/M/Pnw/Mnw，沿用仓库 EH-ratio Gaussian split。
2. Gaussian 使用 exact-lattice measure；PNG 使用原 finite-box continuum measure。
3. 保存所有 native 原子项、阻尼参数和内容哈希；不写入历史仓库或旧缓存。
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
import json
import subprocess
import sys
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import convolve1d

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'marisa_b_core'
OLD=Path('/pscratch/sd/l/lzy/reconstruction-png')
OUT=ROOT/'halo/outputs/local_png_tree_ir_20260930'
sys.path.insert(0,str(CORE/'python'))
sys.path.insert(0,str(CORE/'scripts/production'))
from marisa_b.tree_png_ir_native import NativeTreeIR,NATIVE_TERMS
from marisa_b.tree_png_ir_resummation import HaloTreeBias,bao_displacement_variance
from run_pre_recon_halo_png_uplift_v1 import tree_work

MATRIX=OLD/'analysis/quijote_halo_z0p5_r10_lam1_dm_nenv_fid500_lcp100/matrices/baseline_pre_standard_matrix.npz'
TREE_MANIFEST=OLD/'analysis/quijote_halo_z0p5_pre_direct_finite_tree_kmax0p20_v1/theory/pre_halo_tree_finite_png_z0p5_kmax0p20_nrad4_nmu80.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return {'path':str(path),'sha256':sha(path)}

def eh_shape(k):
    # 与 src/eft_v2/ir_resummation.cpp 相同的 Quijote 无振荡参考形状。
    h,om,ob,ns,theta=.6711,.3175,.049,.9624,2.7255/2.7
    fb=ob/om;omh2=om*h*h
    ag=1-.328*np.log(431*omh2)*fb+.38*np.log(22.3*omh2)*fb**2
    sound=44.5*np.log(9.83/omh2)/np.sqrt(1+10*(ob*h*h)**.75)*h
    gamma=om*h*(ag+(1-ag)/(1+(.43*k*sound)**4))
    q=k*theta**2/gamma;logarithm=np.log(2*np.e+1.8*q)
    transfer=logarithm/(logarithm+(14.2+731/(1+62.5*q))*q*q)
    return k**ns*transfer**2

class Spectra:
    def __init__(self,table,smoothing=.25):
        raw=np.loadtxt(table);self.raw=raw
        self.kmin=float(raw[0,0]);self.kmax=float(raw[-1,0])
        self.p=CubicSpline(np.log(raw[:,0]),raw[:,1])
        self.m=CubicSpline(np.log(raw[:,0]),raw[:,3])
        grid=np.geomspace(self.kmin,self.kmax,8192)
        ratio=self.p(np.log(grid))/eh_shape(grid)
        dx=np.log(grid[1]/grid[0]);radius=int(np.ceil(4*smoothing/dx))
        offsets=np.arange(-radius,radius+1)
        weights=np.exp(-.5*(offsets*dx/smoothing)**2);weights/=weights.sum()
        padded=np.r_[ratio[0]+np.arange(-radius,0)*(ratio[1]-ratio[0]),ratio,
                     ratio[-1]+np.arange(1,radius+1)*(ratio[-1]-ratio[-2])]
        smooth=convolve1d(padded,weights,mode='nearest')[radius:-radius]
        self.nw_ratio=CubicSpline(np.log(grid),np.log(smooth))
        self.grid=grid
    def power(self,k):
        x=np.asarray(k);value=self.p(np.log(np.maximum(x,self.kmin)))
        return np.where(x<self.kmin,self.raw[0,1]*(x/self.kmin)**.9624,value)
    def nowiggle(self,k):
        x=np.asarray(k);return np.exp(self.nw_ratio(np.log(np.maximum(x,self.kmin))))*eh_shape(x)
    def transfer(self,k):
        return self.m(np.log(k))

def vectors_from_triangles(t):
    t=np.asarray(t);v=np.zeros((len(t),3,3))
    v[:,0,2]=t[:,0];v[:,1,0]=t[:,1]*np.sqrt(1-t[:,2]**2)
    v[:,1,2]=t[:,1]*t[:,2];v[:,2]=-v[:,0]-v[:,1]
    return v

def accumulate(values,indices,ids,weights):
    out=np.zeros((120,len(NATIVE_TERMS)))
    for column in range(values.shape[1]):
        out[indices,column]=np.bincount(ids,weights*values[:,column],minlength=len(indices))
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nrad',type=int,default=4);parser.add_argument('--nmu',type=int,default=80)
    parser.add_argument('--r-bao',type=float,default=110.);parser.add_argument('--smoothing',type=float,default=.25)
    parser.add_argument('--skip-lattice',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    table=Path(json.loads(TREE_MANIFEST.read_text())['inputs']['png_table']['path'])
    spectrum=Spectra(table,args.smoothing)
    with np.load(MATRIX) as z:
        geometry=SimpleNamespace(weighted_pairs=z['b000_k'],shell_edges=z['b000_k_edges'])
    indices,triangles,meta,work=tree_work(geometry,nmu=args.nmu,nradial=args.nrad,kfund=2*np.pi/1000,kmax=.20)
    meta=np.asarray(meta);ids=meta[:,0].astype(int);weights=.5*meta[:,1]
    vectors=vectors_from_triangles(triangles)
    norms=np.linalg.norm(vectors,axis=2)
    p=spectrum.power(norms);nw=spectrum.nowiggle(norms);m=spectrum.transfer(norms)
    native=NativeTreeIR(CORE/'build/tree_png_ir/libmarisa_tree_png_ir.so')
    bias=HaloTreeBias(1,1,1,1,1,1)
    lattice=None
    lattice_binary=CORE/'build/tree_png_ir/tree_ir_lattice_nodes'
    if not args.skip_lattice:
        nodefile=OUT/f'lattice_nodes_r{args.nrad}_m{args.nmu}.npz'
        if nodefile.exists():
            with np.load(nodefile) as z:lattice=np.asarray(z['nodes'])
        else:
            bins=''.join(f'{i} '+' '.join(f'{x:.17g}' for x in geometry.shell_edges[i].ravel())+'\n' for i in indices)
            result=subprocess.run([str(lattice_binary),str(args.nrad),str(args.nmu)],input=bins,text=True,capture_output=True,check=True)
            lattice=np.loadtxt(io.StringIO(result.stdout))
            np.savez_compressed(nodefile,nodes=lattice)
            (OUT/f'lattice_nodes_r{args.nrad}_m{args.nmu}.log').write_text(result.stderr)
        lv=lattice[:,2:].reshape(-1,3,3);lk=np.linalg.norm(lv,axis=2)
        lp=spectrum.power(lk);lnw=spectrum.nowiggle(lk);lm=spectrum.transfer(lk)
        lookup={int(i):j for j,i in enumerate(indices)}
        lids=np.array([lookup[int(i)] for i in lattice[:,0]])
    sigmas={}
    for cut in (.05,.10,.15,.20):
        # 精确包含积分上限；低端外推至 1e-6，避免把网格截断当作 k_IR。
        q=np.geomspace(1.e-6,cut,8193)
        sigmas[f'{cut:.2f}']=bao_displacement_variance(q,spectrum.nowiggle(q),r_bao=args.r_bao,k_ir=cut)
    variants=[('fixed_order','fixed_order',0.),('strict_zero','strict_lo',0.)]
    variants += [(f'strict_kir{cut:.2f}','strict_lo',sigmas[f'{cut:.2f}']) for cut in (.05,.10,.15,.20)]
    variants += [('factorized_kir0.15','factorized_closure',sigmas['0.15'])]
    arrays={'indices':indices,'geometry':geometry.weighted_pairs,'sigma2':np.array(list(sigmas.values()))}
    for name,scheme,sigma in variants:
        values=native.evaluate(vectors,p,nw,m,bias,sigma2=sigma,scheme=scheme)
        arrays[name+'__continuum']=accumulate(values,indices,ids,weights)
        if lattice is not None:
            values=native.evaluate(lv,lp,lnw,lm,bias,sigma2=sigma,scheme=scheme)
            arrays[name+'__lattice']=accumulate(values,indices,lids,lattice[:,1])
        print(f'{name}: Sigma2={sigma:.8f}, continuum triangles={len(vectors)}',flush=True)
    tag=f'r{args.nrad}_m{args.nmu}_s{args.smoothing:.2f}_rbao{args.r_bao:.1f}'
    output=OUT/f'tree_ir_templates_{tag}.npz'
    np.savez_compressed(output,**arrays)
    k=spectrum.grid;full=spectrum.power(k);smooth=spectrum.nowiggle(k);transfer=spectrum.transfer(k)
    np.savez_compressed(OUT/f'spectrum_split_s{args.smoothing:.2f}.npz',k=k,plin=full,pnw=smooth,pwiggle=full-smooth,
                        transfer=transfer,transfer_nw=transfer*np.sqrt(smooth/full))
    metadata={'status':'computed','redshift':.5,'units':{'k':'h/Mpc','r':'Mpc/h','sigma2':'(Mpc/h)^2'},
              'sigma2_by_kir':sigmas,'r_bao':args.r_bao,'smoothing_lambda_ln_k':args.smoothing,
              'split':'Gaussian Eisenstein-Hu ratio, Quijote fiducial, 8192 log nodes, 4-sigma kernel',
              'variants':variants,'terms':NATIVE_TERMS,'projection':work,
              'lattice_nodes':0 if lattice is None else len(lattice),
              'inputs':{'png_table':record(table),'geometry':record(MATRIX),'native_library':record(native.path),
                        'native_source':record(CORE/'src/marisa_b/tree_png_ir.cpp'),'producer':record(Path(__file__)),
                        'node_exporter':record(lattice_binary)},'output':record(output)}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output,flush=True)

if __name__=='__main__':main()
