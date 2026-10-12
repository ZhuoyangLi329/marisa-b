"""模块大纲：44 个认证两腿 bins；exact lattice 源；三次 b1 模板；随机项秩依据。

每个 bin 独立 checkpoint；degree=3 是已逐项验证的 pre-recon tree/PBS 代数。
"""
import argparse
import gc
import time
import numpy as np
from rsd_b000_contract import *
from rsd_b000_fft_projection import FFTProjection,integrate_design

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,required=True);parser.add_argument('--meshsize',type=int,default=144)
    parser.add_argument('--probe',action='store_true');args=parser.parse_args();t=time.time()
    ref=reference();index=int(ref['certified_indices'][args.index]);pair=ref['shell_pairs'][index]
    directory=OUT/f'template_bins_m{args.meshsize}';directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'bin{index:03d}.npz'
    if target.exists():
        with np.load(target) as z:
            if str(z['contract_sha256'])!=digest(OUT/'contract.json'):raise ValueError('template contract drift')
        print('REUSE '+str(target),flush=True);return
    sampled=[];peaks=[];counts=None;noise=None
    for b1 in BNODES:
        projector=FFTProjection(args.meshsize,ref['edges'],pair,b1)
        counts=projector.counts;sampled.append(integrate_design(projector));peaks.append(projector.cache_peak_bytes)
        if noise is None:noise=projector.stochastic()
        del projector;gc.collect()
        print(f'bin={index} b1={b1:.5f} elapsed={time.time()-t:.1f}',flush=True)
    vander=np.polynomial.chebyshev.chebvander((BNODES-2.25)/1.75,3)
    coef=np.linalg.solve(vander,np.asarray(sampled).reshape(4,-1)).reshape(4,3,3)
    # Chebyshev 四节点与独立 direct FFT 的第五点；正式 whitened 误差在协方差完成后验收。
    probe_error=np.nan;probe_direct=np.zeros((3,3));probe_prediction=np.zeros((3,3))
    if args.probe:
        b1=1.83641;projector=FFTProjection(args.meshsize,ref['edges'],pair,b1)
        probe_direct=integrate_design(projector)
        probe_prediction=np.polynomial.chebyshev.chebval((b1-2.25)/1.75,coef)
        scale=np.max(np.abs(probe_direct),axis=1,keepdims=True)
        probe_error=float(np.max(np.abs(probe_prediction-probe_direct)/np.maximum(1.,scale)))
        if probe_error>2e-10:raise ValueError(('b1 polynomial mismatch',index,probe_error))
    np.savez_compressed(target,coefficients=coef,stochastic_geometry=noise,shell_pair=pair,counts=np.array(counts),
                        theory_meshsize=args.meshsize,cache_peak_bytes=max(peaks),probe_relative_error=probe_error,
                        probe_direct=probe_direct,probe_prediction=probe_prediction,contract_sha256=digest(OUT/'contract.json'))
    print(f'COMPLETE bin={index} seconds={time.time()-t:.1f} cache_GB={max(peaks)/1e9:.2f}',flush=True)

if __name__=='__main__':main()
