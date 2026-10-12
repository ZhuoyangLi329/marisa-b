"""模块大纲：独立只读诊断两腿 cut 的 closing-leg 分布；不改任何似然。

mode-count 加权分布不是 B 信号或 covariance 加权分布，不作唯一物理归因。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
from numpy.fft import rfftn,irfftn
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from quijote_strict_triangle_fft_estimator import assign_shells

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mesh',type=int,default=144);args=parser.parse_args()
    with np.load(args.reference) as z:
        edges=np.array(z['edges']);idx=np.array(z['certified_indices']);pairs=np.array(z['shell_pairs']);means=np.array(z['B000_k'])
    n=args.mesh;kf=2*np.pi/1000.;axis=np.rint(np.fft.fftfreq(n)*n)*kf
    x,y,z=np.meshgrid(axis,axis,axis,indexing='ij',sparse=True);k=np.sqrt(x*x+y*y+z*z);active=k>0
    labels=assign_shells(k,edges);shells=sorted(set(pairs[idx].ravel()));transforms={};counts={};counts_full={}
    if 2*max(edges[np.array(shells)+1])>=np.pi*n/1000:raise ValueError('closing mode alias; increase diagnostic FFT mesh')
    for s in shells:
        full=labels==s;mask=full&active;counts_full[s]=int(full.sum());counts[s]=int(mask.sum())
        transforms[s]=rfftn(mask.astype(float))
    cuts=(.08,.10,.12,.14,.18,.20);rows=[];hist_edges=np.arange(0,.4201,.0005)
    for index in idx:
        i,j=map(int,pairs[index]);conv=np.rint(irfftn(transforms[i]*transforms[j],s=(n,n,n)))
        if np.min(conv)<0 or int(np.sum(conv))!=counts[i]*counts[j]:raise ValueError('lattice convolution count failure')
        valid=int(np.sum(conv[active]));hist=np.histogram(k[active],bins=hist_edges,weights=conv[active])[0]
        quantiles=[float(hist_edges[min(int(np.searchsorted(np.cumsum(hist),q*valid)),len(hist_edges)-2)]) for q in (.5,.9,.99)]
        maximum=float(np.max(k[conv>0]));fractions={f'{cut:.2f}':float(np.sum(conv[k>cut+1e-12])/valid) for cut in cuts}
        rows.append({'index':int(index),'shell_pair':[i,j],'k1_k2_mean':means[index].tolist(),
                     'full_two_shell_denominator':counts_full[i]*counts_full[j],
                     'all_three_nonzero_triangle_count':valid,'closing_k3_max':maximum,
                     'closing_k3_q50_q90_q99_approx':quantiles,'quantile_histogram_width':.0005,
                     'fraction_valid_triangles_k3_above_threshold':fractions,'hist':hist})
    pooled=[]
    for cut in cuts:
        chosen=[r for r in rows if max(r['k1_k2_mean'])<=cut+1e-12]
        hist=sum((r['hist'] for r in chosen),start=np.zeros(len(hist_edges)-1));total=sum(r['all_three_nonzero_triangle_count'] for r in chosen)
        frac=sum(r['all_three_nonzero_triangle_count']*r['fraction_valid_triangles_k3_above_threshold'][f'{cut:.2f}'] for r in chosen)/total
        q90=float(hist_edges[min(int(np.searchsorted(np.cumsum(hist),.9*total)),len(hist_edges)-2)])
        pooled.append({'kmax_B':cut,'B_points':len(chosen),'closing_k3_max':max(r['closing_k3_max'] for r in chosen),
                       'pooled_mode_count_fraction_k3_above_nominal_kmax':frac,'pooled_closing_k3_q90_approx':q90,
                       'minimum_and_maximum_bin_fraction':[min(r['fraction_valid_triangles_k3_above_threshold'][f'{cut:.2f}'] for r in chosen),
                                                            max(r['fraction_valid_triangles_k3_above_threshold'][f'{cut:.2f}'] for r in chosen)]})
    for row in rows:row.pop('hist')
    result={'status':'PASS','reference_sha256':digest(args.reference),'source_sha256':digest(Path(__file__)),
            'mesh':n,'FFT_workers':1,'scope':'geometric diagnostic only; no model, covariance, MCMC or data changes',
            'interpretation':'The published kmax cuts two measured legs, not the closing third leg.',
            'weighting':'pure valid lattice-mode counts pooled over bins, NOT B-signal or likelihood weights',
            'realspace_has_same_geometry':True,'unique_RSD_cause_identified':False,'by_cut':pooled,'per_bin':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','by_cut':pooled},indent=2),flush=True)

if __name__=='__main__':main()
