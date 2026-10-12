#!/usr/bin/env python3
"""模块大纲：加速投影/PNG 指数混合与独立 native 逐节点结果交叉验收。"""
import numpy as np
from paware_pb_ir_templates import *


def main():
    sp,_=spectrum();src=Sources();var=Variance();engine=NativePostTreeIR(LIBRARY)
    rng=np.random.default_rng(301027)
    v=rng.normal(size=(13,3,3))*.06;v[:,2]=-v[:,0]-v[:,1]
    w=rng.uniform(.1,1.,len(v));w/=w.sum()
    unit=src.evaluate(v,sp)
    q=np.geomspace(QMIN,KIR,2049)
    settings=PostReconIrSettings(RBAO,KIR,QMIN,angular_mu_order=48,angular_phi_order=96)
    rec=ReconstructionConfig(10.,BREC,8.,4)
    adaptive=AdaptiveReconstructionConfig(2*DC*(BREC-PREC),1.)
    errors=[]
    for b in (.2,1.93,BREC,4.2,7.8):
        b2,gamma,bs=.35,-.17,.73
        bn=b2-4*gamma/3;bp=2*DC*(b-PREC)
        bpd=2*DC*(b-1)+2*(DC*(bn-8*(b-1)/21)-b+1)
        bp2=4*DC*(DC*(bn-8*(b-1)/21)-2*(b-1))
        bias=HaloTreeBias(b,bn,gamma,bp,bpd,bp2)
        direct=engine.evaluate(v,*sample(sp,v),bias,rec,q,sp.power(q),sp.nowiggle(q),sp.transfer(q),settings,adaptive)
        integ,_=project_raw(v,w,unit,var,[b],"sto_marisa")
        design=combine(integ[0],b,PREC,1.,np.ones(3))
        predicted=np.array([1.,b2,gamma,0.,bs])@design
        fixed=direct.fixed_order
        fixed_d=np.stack([fixed[:,0]+fixed[:,1],fixed[:,LCOLS].sum(1),fixed[:,QCOLS].sum(1)],axis=1)
        fixed_s=fixed[:,17:20]
        ir_s=direct.source[:,17:20].copy()
        ir_s[:,1]+=direct.stochastic_mixing[:,0]
        ir_s[:,2]+=direct.stochastic_mixing[:,1]+direct.stochastic_mixing[:,2]
        expected=w@(direct.deterministic-fixed_d+bs*(ir_s-fixed_s))
        scale=np.maximum(1.,abs(w@(fixed_d+bs*fixed_s)))
        error=float(max(abs(predicted-expected)/scale))
        errors.append({"b1":b,"scaled_error":error})
    assert max(x["scaled_error"] for x in errors)<1.e-8,errors
    save_json(OUT/"IR_DRESSING_AUDIT.json",{"status":"pass","errors":errors,
        "includes":"deterministic source, all exponent mixing, inherited common-amplitude stochastic mixing",
        "native_validation":"48x96 angles,2049 radial nodes, independent triangles"})
    print("DRESSING PASS",errors,flush=True)


if __name__=="__main__":main()
