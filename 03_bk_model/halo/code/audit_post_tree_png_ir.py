#!/usr/bin/env python3
"""模块大纲：真实 z=0.5 谱/current-p 示例；三方法；IR 积分收敛；有限盒真实方向 shell；PDF。

只做理论数值验收，不使用被隔离的 post 目录，不声明新的 fNL/kmax 有效范围。
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import CubicSpline
ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "marisa_b_core"
sys.path[:0] = [str(CORE / "python"), str(CORE / "scripts/examples"), str(CORE / "scripts/production")]
from reconstruction_current_p import tracer_biases
from marisa_b.tree_png_ir_resummation import HaloTreeBias, ReconstructionConfig, triangle_vectors
from marisa_b.post_recon_tree_png_ir import AdaptiveReconstructionConfig, PostReconIrSettings, post_recon_displacement_variance_at_fnl
from marisa_b.post_recon_tree_png_ir_native import NativePostTreeIR, project_post_tree_nodes
from generate_post_tree_png_ir import generate, record


def scaled_error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.max(np.abs(a-b)/np.maximum(1.,np.abs(b))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"halo/outputs/local_png_post_tree_ir_20260930/audit")
    args=parser.parse_args();out=args.output
    if out.exists():
        raise FileExistsError("use a fresh audit directory; prior diagnostics are preserved")
    out.mkdir(parents=True)
    spectrum=ROOT/"halo/outputs/local_png_tree_ir_20260930/spectrum_split_s0.25.npz"
    configpath=CORE/"configs/reconstruction_current_p_example.json"
    config=json.loads(configpath.read_text())
    with np.load(spectrum) as z:
        logk=np.log(z["k"])
        interp=[CubicSpline(logk,z[key],extrapolate=False) for key in ("plin","pnw","transfer")]
    def sample(k): return [f(np.log(k)) for f in interp]
    qmin=2*np.pi/1000
    def grid(n=1025, lo=qmin, hi=.15):
        q=np.geomspace(lo,hi,n);return [q,*sample(q)]
    library=CORE/"build/tree_png_ir/libmarisa_post_tree_png_ir.so"
    engine=NativePostTreeIR(library)
    settings=PostReconIrSettings(110.,.15,qmin,angular_mu_order=32,angular_phi_order=64)
    v=np.array([triangle_vectors(.04,.065,-.42),triangle_vectors(.10,.13,-.61),triangle_vectors(.17,.16,-.55)])
    # 非固定平面：用明确正交矩阵旋转，保持闭合和边长不变。
    rotation=np.array([[.36,-.48,.8],[.8,.60,0.],[-.48,.64,.60]])
    v=v@rotation.T
    external=sample(np.linalg.norm(v,axis=2))
    # 第一个非零 FFT shell 的实际整数格点，不使用 Haar/固定平面替代方向。
    modes=np.array([x for x in itertools.product(range(-1,2),repeat=3) if .5<=np.linalg.norm(x)<1.5])
    pairs=[(a,b) for a in modes for b in modes if np.any(a+b)]
    shell=qmin*np.array([[a,b,-a-b] for a,b in pairs],dtype=float)
    weights=np.full(len(shell),1./len(modes)**2)
    assert len(modes)==18 and len(shell)==306
    report={"scope":"implementation/numerical audit; not a post-catalogue or kmax validation",
            "inputs":{"spectrum":record(spectrum),"current_p":record(configpath),"library":record(library)},
            "ir_settings":asdict(settings),"radial_nodes":1025,"shift_noise":"disabled; no assumed 1/nbar for weighted tracer",
            "calibration_warning":"parent pre/STO p values are examples, not dedicated post calibrations",
            "illustrative_nuisance":{"b2_likelihood":.3,"gamma2":-.15},"methods":{}}
    pdf_values={}
    for method,tracer,eta in [("standard_gra","halo",0.),("marisa_gra","halo",1.),("sto_marisa","sto",1.)]:
        t=config["tracers"][tracer];r=t["reconstruction_example"]
        bb=tracer_biases(t["b1_theory"],t["p_halo"],.3,-.15)
        bias=HaloTreeBias(bb["b1"],bb["b2_native"],-.15,bb["bphi"],bb["bphidelta"],bb["bphi2"])
        rec=ReconstructionConfig(r["R_mpc_h"],r["b_rec"],r["cell_size_mpc_h"],r["cic_power_per_axis"])
        adaptive=AdaptiveReconstructionConfig(2*1.686*(r["b_rec"]-r["p_rec"]),eta)
        def evaluate(s=settings,n=1025):
            return engine.evaluate(v,*external,bias,rec,*grid(n,s.q_min,s.k_ir),s,adaptive)
        base=evaluate()
        angular={}
        for order in (24,48):
            candidate=evaluate(replace(settings,angular_mu_order=order,angular_phi_order=2*order))
            angular[str(order)]={"variance":scaled_error(candidate.variance,base.variance),
                                 "G_L_Q":scaled_error(candidate.deterministic,base.deterministic)}
        radial={}
        for n in (257,513,2049):
            candidate=evaluate(n=n)
            radial[str(n)]={"variance":scaled_error(candidate.variance,base.variance),
                            "G_L_Q":scaled_error(candidate.deterministic,base.deterministic)}
        sensitivity={}
        for label,s in [("qmin_half",replace(settings,q_min=qmin/2)),("qmin_double",replace(settings,q_min=qmin*2)),
                        ("kIR_0.10",replace(settings,k_ir=.10)),("kIR_0.20",replace(settings,k_ir=.20)),
                        ("rbao_100",replace(settings,r_bao=100.)),("rbao_120",replace(settings,r_bao=120.))]:
            test=evaluate(s)
            sensitivity[label]={"variance":test.variance.tolist(),"G_L_Q":test.deterministic.tolist()}
        exact_checks={}
        for fnl in (-100.,100.):
            exact=np.array(post_recon_displacement_variance_at_fnl(v[1],*grid(),bias,rec,settings,fnl,adaptive))
            approx=base.variance[1,0]+fnl*base.variance[1,1]+fnl**2*base.variance[1,2]
            exact_checks[str(fnl)]={"exact_variance":exact.tolist(),"f2_variance":approx.tolist(),
                                    "scaled_variance_truncation":scaled_error(approx,exact)}
        sgrid=grid(129)
        ss=replace(settings,angular_mu_order=16,angular_phi_order=32)
        sbatch=engine.evaluate(shell,*sample(np.linalg.norm(shell,axis=2)),bias,rec,*sgrid,ss,adaptive)
        projected=project_post_tree_nodes(sbatch,weights,np.zeros(len(shell),dtype=int),number_of_bins=1,
                                          weights_include_original_denominator=True)
        # 真实 shell 测试只代表首个窄 shell；不冒充全部 kmax bins 的收敛验收。
        forder=replace(ss,angular_mu_order=24,angular_phi_order=48)
        fine=engine.evaluate(shell,*sample(np.linalg.norm(shell,axis=2)),bias,rec,*sgrid,forder,adaptive)
        finep=project_post_tree_nodes(fine,weights,np.zeros(len(shell),dtype=int),number_of_bins=1,
                                      weights_include_original_denominator=True)
        nodefile=out/f"{method}_exact_shell_nodes.npz"
        np.savez_compressed(nodefile,vectors=shell,power=sample(np.linalg.norm(shell,axis=2))[0],
            no_wiggle=sample(np.linalg.norm(shell,axis=2))[1],transfer=sample(np.linalg.norm(shell,axis=2))[2],
            q=sgrid[0],qpower=sgrid[1],qnw=sgrid[2],qm=sgrid[3],weights=weights,bin_ids=np.zeros(len(shell),dtype=int))
        jobconfig={"method":method,"bias":asdict(bias),"reconstruction":asdict(rec),"adaptive":asdict(adaptive),
                   "ir":asdict(ss),"reconstruction_parameter_path":"f_rec=eta*f",
                   "weights_include_original_denominator":True,"absolute_orientation_nodes":True,"number_of_bins":1,
                   "measure":"exact enumeration: 18 x 18 FFT modes, closing zero pairs omitted, denominator 324"}
        jobpath=out/f"{method}_config.json";jobpath.write_text(json.dumps(jobconfig,indent=2)+"\n")
        resultpath=out/f"{method}_shell_templates.npz"
        generate(nodefile,jobpath,resultpath,library,chunk_size=77)
        with np.load(resultpath) as generated:
            chunk_error=scaled_error(generated["deterministic"],projected["deterministic"])
            assert chunk_error<1e-12
        row={"bias":asdict(bias),"reconstruction":asdict(rec),"adaptive":asdict(adaptive),
             "variance":base.variance.tolist(),"G_L_Q":base.deterministic.tolist(),"mixing":base.mixing.tolist(),
             "angular_scaled_errors":angular,"radial_scaled_errors":radial,"cutoff_sensitivity":sensitivity,
             "finite_f_variance_checks":exact_checks,"shell":{"nodes":len(shell),"full_pairs":324,
                 "weight_sum":float(weights.sum()),"G_L_Q":projected["deterministic"].tolist(),
                 "angular_scaled_error":scaled_error(finep["deterministic"],projected["deterministic"]),
                 "chunk_scaled_error":chunk_error}}
        report["methods"][method]=row
        pdf_values[method]=(base.variance[1],base.mixing[1])
        print(method, json.dumps({"angular":angular,"radial":radial,"shell":row["shell"]}),flush=True)
    assert all(max(x["G_L_Q"] for x in row["angular_scaled_errors"].values())<1e-4 for row in report["methods"].values())
    assert all(row["shell"]["angular_scaled_error"]<1e-4 for row in report["methods"].values())
    assert all(row["radial_scaled_errors"]["2049"]["variance"]<1e-8 for row in report["methods"].values())
    assert all(row["radial_scaled_errors"]["2049"]["G_L_Q"]<1e-9 for row in report["methods"].values())
    report["status"]="PASS_selected_nodes_and_first_exact_shell; full_shell_likelihood_acceptance_pending"
    (out/"POST_TREE_IR_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for method,(sigma,mix) in pdf_values.items():
        for order,ax in enumerate(axes):
            ax.plot([1,2,3],sigma[order],"o-",label=method)
            ax.set(xlabel="triangle leg",ylabel=f"Sigma coefficient f^{order} [(Mpc/h)^2]",xticks=[1,2,3])
    axes[0].legend(fontsize=8)
    fig.suptitle("Post tree IR: current-p parent-tracer examples (not a catalogue fit)")
    fig.savefig(out/"post_tree_ir_current_p.pdf");plt.close(fig)
    print(out/"POST_TREE_IR_AUDIT.json",flush=True)


if __name__=="__main__":main()
