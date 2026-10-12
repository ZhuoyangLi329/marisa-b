#!/usr/bin/env python3
"""模块大纲：六组链的冻结似然复现、收敛、旧结果不变检查与对照 JSON。"""
from pathlib import Path
import json
import h5py
import numpy as np
import run_paware_pb_ir_mcmc as run


def main():
    results=[]
    for method,ir in (("sto_marisa",False),("pre",True),("sto_marisa",True)):
        for cut in (.12,.14):
            name=run.case_name(method,ir,cut);directory=run.OUT/name
            case=run.legacy.Case(directory)
            result=json.loads((directory/"RESULTS.json").read_text())
            assert run.legacy.digest(directory/"likelihood.npz")==case.meta["likelihood_sha256"]
            baseline=run.OLD/run.legacy.tag(method,cut)
            assert run.legacy.digest(baseline/"likelihood.npz")==case.meta["baseline_likelihood_sha256"]
            assert run.legacy.digest(baseline/"contract.json")==case.meta["baseline_contract"]
            with np.load(baseline/"likelihood.npz") as old_arrays:
                unchanged={key:bool(np.array_equal(getattr(case,key),old_arrays[key]))
                           for key in ("target","covariance","chol","prior_mean","prior_whitener","bounds","indices")}
            assert all(unchanged.values()),unchanged
            baseline_meta=json.loads((baseline/"contract.json").read_text())
            for key in ("kmax_P","volume","p_halo","p_fixed_not_marginalized","n_target","n_fid",
                        "prior_weight","full_PB_cross_covariance","hartlap","parameter_names"):
                assert case.meta[key]==baseline_meta[key],key
            if ir:
                # 逐个复核实际使用的冻结资产，不只信任合同中的 pass 标签。
                for path,expected in case.meta["IR_contract"]["frozen_assets_sha256"].items():
                    assert run.legacy.digest(Path(path))==expected,path
            errors=[]
            with h5py.File(directory/"chain.h5","r") as h:
                for en in range(4):
                    g=h[f"ensemble_{en}"]
                    for step in (0,len(g["chain"])//2,len(g["chain"])-1):
                        errors.extend(abs(case.logp(g["chain"][step,::4])-g["log_prob"][step,::4]))
            assert max(errors)<1.e-7,max(errors)
            assert all(case.meta["unchanged_arrays"].values())
            row={"case":name,"method":method,"IR":ir,"kmax_B":cut,
                 "fNL":result["posterior"]["fNL"],"converged":result["diagnostics"]["passed"],
                 "max_Rhat":max(result["diagnostics"]["rank_normalized_split_rhat"]),
                 "GoF":result["map_gof_p_value"],"logp_reproduction_max_error":max(errors),
                 "unchanged_arrays_rechecked":unchanged,
                 "steps_per_ensemble":result["diagnostics"]["steps"],
                 "boundary_warning":result["diagnostics"]["boundary_warning"],
                 "truth_in_68":result["truth_in_68"],"truth_in_95":result["truth_in_95"],
                 "likelihood_sha256":case.meta["likelihood_sha256"]}
            results.append(row)
    old=json.loads((run.OLD/"SUMMARY.json").read_text())
    old_hash=run.legacy.digest(run.OLD/"SUMMARY.json")
    assert old_hash=="64829ca8519bdf541904491b862d5c1bc2d577e8712ec096478196b3f3753ae6"
    differences=[]
    for method in ("pre","sto_marisa"):
        for cut in (.12,.14):
            old_row=next(r for r in old["rows"] if r["method"]==method and r["kmax_B"]==cut)
            ir_row=next(r for r in results if r["method"]==method and r["kmax_B"]==cut and r["IR"])
            baseline=old_row
            if method=="sto_marisa":
                baseline=next(r for r in results if r["method"]==method and r["kmax_B"]==cut and not r["IR"])
                differences.append({"effect":"source_kernel_correction","method":method,"kmax_B":cut,
                                    "delta_fNL_median":baseline["fNL"]["median"]-old_row["fNL"]["median"],
                                    "MAP_GoF_before":old_row["MAP_GoF"],"MAP_GoF_after":baseline["GoF"]})
            width_before=(baseline["fNL"]["q84"]-baseline["fNL"]["q16"])/2
            width_after=(ir_row["fNL"]["q84"]-ir_row["fNL"]["q16"])/2
            differences.append({"effect":"IR_only","method":method,"kmax_B":cut,
                                "delta_fNL_median":ir_row["fNL"]["median"]-baseline["fNL"]["median"],
                                "half_width_ratio_after_before":width_after/width_before,
                                "MAP_GoF_before":baseline.get("GoF",baseline.get("MAP_GoF")),
                                "MAP_GoF_after":ir_row["GoF"]})
    report={"status":"complete" if all(r["converged"] for r in results) else "needs_more_sampling",
            "six_new_cases":results,"old_comparison_rows":old["rows"],"old_summary_sha256":old_hash,
            "separated_effects":differences,
            "comparison":"STO old to corrected no-IR isolates b_rec correction; corrected no-IR to IR isolates IR; pre uses retained old no-IR",
            "scope_warning":"inherited f_rec=f theory path retained; catalogue f_rec=100 fixed; no claim of fixed-f_rec inference or kmax extension"}
    run.legacy.save_json(run.OUT/"SUMMARY.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
