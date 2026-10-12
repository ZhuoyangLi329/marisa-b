#!/usr/bin/env python3
"""模块大纲：独立 kernel 缩放、缓存替换、bias 组合与新旧合同数值验收。"""
import copy
import json
import numpy as np
from paware_pb_corrected_model import CorrectedModel, BREC, PREC, BNAMES
from paware_pb_ir_templates import Sources, spectrum, factors, OUT, digest, save_json, DC


def relative(a,b):
    return float(np.max(abs(a-b)/np.maximum(1.,abs(b))))


def main():
    model=CorrectedModel("sto_marisa")
    rng=np.random.default_rng(264930)
    v=rng.normal(size=(37,3,3))*.06;v[:,2]=-v[:,0]-v[:,1]
    sp,_=spectrum();src=Sources()
    ratio=model.ratio
    before=src.evaluate(v,sp,brec=model.old_brec)
    after=src.evaluate(v,sp,brec=BREC)
    expected=before.copy()
    expected[..., [1,8,14]]*=ratio
    expected[..., [24,26]]*=ratio**2
    expected[...,25]*=ratio**3
    native_scaling=relative(after,expected)
    assert native_scaling<2.e-12,native_scaling
    bias_errors=[]
    for b in (.2,1.93,2.228,7.8):
        b2,gamma=.32,-.19
        fac=factors(b,b2,gamma,PREC)
        bp=2*DC*(b-PREC);bn=b2-4*gamma/3
        bpd=2*DC*(b-1)+2*(DC*(bn-8*(b-1)/21)-b+1)
        bp2=4*DC*(DC*(bn-8*(b-1)/21)-2*(b-1))
        direct=src.evaluate(v,sp,bias=[b,bn,gamma,bp,bpd,bp2])
        product=after*fac
        product[...,0]=product[...,21]+product[...,22]+product[...,23]
        bias_errors.append(relative(product,direct))
    assert max(bias_errors)<1.e-10,bias_errors
    mappings=[{"b1":b,"b2":.31,"gamma2":-.19,"Ashot_residual":-.7,"Bshot_residual":.23}
              for b in (.2,1.93,2.228,7.8)]
    wrapped=[model.parts(mp) for mp in mappings]
    # 独立实现：直接替换仅存于本进程的源缓存，然后走原 legacy 求值器。
    gaussian=model.assets.post_templates.tied._compiled["diagram:tree"]
    gaussian["b1^4"]=gaussian["b1^4"]*ratio
    for key in ("dBdfNL_local_bphi_reconstruction","Bhalo_tree_fNL2_bphi_sq_reconstruction"):
        model.assets.png.tree[key]=model.assets.png.tree[key]*ratio
    for key in ("gaussian_linear_basis","gaussian_quadratic_basis","png_linear_cross_basis"):
        model.runtime.marisa_finite[key]=model.runtime.marisa_finite[key]*ratio
    errors=[]
    for mp,expected_parts in zip(mappings,wrapped):
        direct=model.legacy.parts(mp)
        errors.append({key:relative(direct[key],expected_parts[key]) for key in direct})
    assert max(v for row in errors for v in row.values())<2.e-12,errors
    report={"status":"pass","native_inverse_brec_scaling_max_error":native_scaling,
            "unit_bias_source_jet_mapping_max_errors":bias_errors,"independent_cache_replacement_errors":errors,
            "scope":"all full/nw/leg-wiggle source coefficients; no MCMC modification"}
    save_json(OUT/"BASELINE_KERNEL_AUDIT.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
