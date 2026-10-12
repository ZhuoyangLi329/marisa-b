#!/usr/bin/env python3
"""模块大纲：native source jets；方向依赖方差；逐节点 IR；保留旧合同的理论差。

源项保持完整 native 27 项。位移方差对 b1 精确二次（p 固定），方向采用
立方对称不变量插值并与独立 native 方向/更密积分验收。投影先 IR 后平均。
不以平均 Sigma 代替逐方向阻尼；不默认添加 STO 的 1/nbar 位移噪声。
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import concurrent.futures
import ctypes
from dataclasses import replace
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn

ROOT = Path("/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model")
CORE = ROOT/"marisa_b_core"
OUT = ROOT/"envclass_b/outputs/paware_pb_ir_brecfix_k012_k014_v2"
TEMPLATES = OUT/"templates"
LIBRARY = OUT/"build/libpaware_ir.so"
sys.path[:0] = [str(CORE/"python"), str(CORE/"scripts/production"), str(ROOT/"halo/code")]
from marisa_b.post_recon_tree_png_ir_native import NativePostTreeIR
from marisa_b.post_recon_tree_png_ir import AdaptiveReconstructionConfig, PostReconIrSettings
from marisa_b.tree_png_ir_resummation import HaloTreeBias, ReconstructionConfig
from prepare_tree_png_ir_templates import Spectra, TREE_MANIFEST, tree_work, vectors_from_triangles
from paware_pb_corrected_model import CorrectedModel, BREC, PREC, BNAMES

DEGREE = 32
CHECK_DEGREE = 48
DC = 1.686
QMIN = 2*np.pi/1000.
KIR = .15
RBAO = 110.
GCOLS = [1, 21, 22, 23]
LCOLS = [2, 3, 4, 5, 6, 7, 8, 24]
QCOLS = [9, 10, 11, 12, 13, 14, 15, 16, 25, 26]
PRE_ZERO = [1, 8, 14, 24, 25, 26]
ORDERS = {"primary": (4, 32, 4, 3, 4), "fine": (6, 48, 4, 4, 4)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2)+"\n")


def spectrum():
    path = Path(json.loads(TREE_MANIFEST.read_text())["inputs"]["png_table"]["path"])
    return Spectra(path), path


def sample(sp, vectors):
    k = np.linalg.norm(vectors, axis=-1)
    return sp.power(k), sp.nowiggle(k), sp.transfer(k)


class Sources:
    def __init__(self):
        self.lib = ctypes.CDLL(str(LIBRARY))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.paware_source_batch.argtypes = [ctypes.c_size_t]+[ptr]*7
        self.lib.paware_source_batch.restype = ctypes.c_int
        self.lib.paware_ir_error.restype = ctypes.c_char_p

    def evaluate(self, vectors, sp, *, brec=BREC, bpr=None, bias=None, pre=False):
        v = np.ascontiguousarray(vectors)
        legs = [np.ascontiguousarray(x) for x in sample(sp, v)]
        b = np.ascontiguousarray(np.ones(6) if bias is None else bias, dtype=float)
        cfg = np.array([10., brec, 8., 4., 2*DC*(BREC-PREC) if bpr is None else bpr, 1.])
        result = np.empty((len(v), 5, 27))
        status = self.lib.paware_source_batch(len(v), v, *legs, b, cfg, result)
        if status:
            raise RuntimeError(self.lib.paware_ir_error().decode())
        if pre:
            result[..., PRE_ZERO] = 0.
        assert np.isfinite(result).all()
        return result


def features(vectors):
    direction = np.asarray(vectors)/np.linalg.norm(vectors, axis=-1, keepdims=True)
    squared = direction**2
    u = (3*np.sum(squared**2, axis=-1)-1)/2
    v = 27*np.prod(squared, axis=-1)
    powers = [(a, b) for a in range(5) for b in range(3) if 4*a+6*b <= 16]
    return np.stack([u**a*v**b for a, b in powers], axis=-1)


class Variance:
    def __init__(self, path=None):
        with np.load(path or TEMPLATES/"variance.npz") as z:
            self.coefficients = z["coefficients"]
            self.pre_sigma = float(z["pre_sigma"])

    def evaluate(self, vectors, b, method="sto_marisa"):
        if method == "pre":
            result = np.zeros((*np.shape(vectors)[:-1], 3))
            result[..., 0] = self.pre_sigma
            return result
        # 三个 b1 节点 0,2,4 的 Lagrange 插值是精确二次，不是近似拟合。
        basis = np.array([(b-2)*(b-4)/8, -b*(b-4)/4, b*(b-2)/8])
        coef = np.einsum("b,bfc->fc", basis, self.coefficients)
        return features(vectors) @ coef


def build_variance():
    if (TEMPLATES/"variance.npz").exists():
        raise FileExistsError("方差资产已存在，禁止覆盖")
    sp, table = spectrum()
    engine = NativePostTreeIR(LIBRARY)
    rec = ReconstructionConfig(10., BREC, 8., 4)
    adaptive = AdaptiveReconstructionConfig(2*DC*(BREC-PREC), 1.)
    settings = PostReconIrSettings(RBAO, KIR, QMIN, angular_mu_order=24, angular_phi_order=48)
    rng = np.random.default_rng(930261)
    vectors = rng.normal(size=(24, 3, 3))*.06
    vectors[:, 2] = -vectors[:, 0]-vectors[:, 1]
    h = features(vectors).reshape(-1, features(vectors).shape[-1])
    def native(v, b, fine=False):
        q = np.geomspace(QMIN, KIR, 2049 if fine else 1025)
        bias = HaloTreeBias(b, .2, -.1, 2*DC*(b-PREC), .3, -.2)
        s = replace(settings, angular_mu_order=48, angular_phi_order=96) if fine else settings
        return engine.evaluate(v, *sample(sp, v), bias, rec, q, sp.power(q), sp.nowiggle(q),
                               sp.transfer(q), s, adaptive)
    coefs = []
    for b in (0., 2., 4.):
        values = native(vectors, b).variance.transpose(0, 2, 1).reshape(-1, 3)
        coefs.append(np.linalg.lstsq(h, values, rcond=1.e-13)[0])
    q = np.geomspace(QMIN, KIR, 8193)
    sigma = simpson(sp.nowiggle(q)*(1-spherical_jn(0,q*RBAO)+2*spherical_jn(2,q*RBAO)), x=q)/(6*np.pi**2)
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    candidate = TEMPLATES/"variance.candidate.npz"
    np.savez(candidate, coefficients=np.array(coefs), pre_sigma=sigma)
    approximate = Variance(candidate)
    holdout = rng.normal(size=(9, 3, 3))*.07
    holdout[:, 2] = -holdout[:, 0]-holdout[:, 1]
    errors = []
    for b in (.13, 1.93, BREC, 5.1, 7.95):
        exact = native(holdout, b, fine=True)
        predicted = approximate.evaluate(holdout, b).transpose(0, 2, 1)
        weighted = abs(predicted-exact.variance)*np.array([1., 300., 90000.])[None, :, None]
        error = float(np.max(np.sum(weighted, axis=1)))
        errors.append({"b1": b, "max_variance_error_fnl_bound300": error})
    if max(x["max_variance_error_fnl_bound300"] for x in errors) > 2.e-6:
        raise RuntimeError(f"方差方向插值未通过: {errors}")
    candidate.rename(TEMPLATES/"variance.npz")
    save_json(TEMPLATES/"variance.json", {"status": "pass", "independent_native_holdout": errors,
        "radial_nodes": 1025, "validation_radial_nodes": 2049, "angular_orders": [24,48],
        "validation_angular_orders": [48,96], "direction_representation": "cubic invariant polynomials through degree16",
        "b1_representation": "exact quadratic for fixed p", "q_min": QMIN, "kIR": KIR, "rBAO": RBAO,
        "shift_noise": "disabled; weighted noise not assumed", "table": str(table), "table_sha256": digest(table),
        "library_sha256": digest(LIBRARY), "variance_sha256": digest(TEMPLATES/"variance.npz")})
    print("VARIANCE PASS", errors, flush=True)


def factors(b, b2, gamma, p):
    bn = b2-4*gamma/3
    bp = 2*DC*(b-p)
    bpd = 2*DC*(b-1)+2*(DC*(bn-8*(b-1)/21)-b+1)
    bp2 = 4*DC*(DC*(bn-8*(b-1)/21)-2*(b-1))
    return np.array([0., b**4, b**3, b*b*bp, b*b*bp, b*b*bpd, b*bn*bp,
        b*gamma*bp, b**3*bp, b*b*bp, b*bp*bp, b*bp*bp, bn*bp*bp, gamma*bp*bp,
        b*b*bp*bp, b*bp*bpd, b*b*bp2, b*b, b*bp, bp*bp, 1., b**3, b*b*bn,
        b*b*gamma, b**4, b**4, b**3*bp])


def combine(integrals, b, p, nbar, adapter):
    """三个数组分别为 IR-fixed、D*d1*w、D*d2*w；返回 nuisance x f阶。"""
    src, d1, d2 = integrals
    def deterministic(b2, gamma):
        fac = factors(b, b2, gamma, p)
        return np.array([src[GCOLS] @ fac[GCOLS],
            src[LCOLS] @ fac[LCOLS] + d1[GCOLS] @ fac[GCOLS],
            src[QCOLS] @ fac[QCOLS] + d1[LCOLS] @ fac[LCOLS] + d2[GCOLS] @ fac[GCOLS]])
    base = deterministic(0., 0.)
    bp = 2*DC*(b-p)
    stochastic = np.array([adapter[0]*b*b*src[17],
        adapter[1]*b*bp*src[18]+adapter[0]*b*b*d1[17],
        adapter[2]*bp*bp*src[19]+adapter[1]*b*bp*d1[18]+adapter[0]*b*b*d2[17]])/nbar
    return np.array([base, deterministic(1.,0.)-base, deterministic(0.,1.)-base,
                     np.zeros(3), stochastic])


def project_raw(v, weights, source, variance, bnodes, method):
    fixed = weights @ source[:, 0]
    nw = weights @ source[:, 1]
    k2 = np.sum(v*v, axis=2)
    wf = (source[:, 2:]*weights[:, None, None]).reshape(-1, 27)
    # 方差方向特征只计算一次，b1 依赖通过三节点精确组合。
    feat = features(v) if method != "pre" else None
    out = []
    for b in bnodes:
        if method == "pre":
            var = np.zeros((*k2.shape, 3)); var[...,0] = variance.pre_sigma
        else:
            lb = np.array([(b-2)*(b-4)/8, -b*(b-4)/4, b*(b-2)/8])
            var = feat @ np.einsum("b,bfc->fc", lb, variance.coefficients)
        damping = np.exp(-k2*var[...,0])
        d1 = -k2*var[...,1]
        d2 = .5*(k2*var[...,1])**2-k2*var[...,2]
        out.append(np.array([nw+damping.ravel() @ wf-fixed,
                             (damping*d1).ravel() @ wf, (damping*d2).ravel() @ wf]))
    return np.array(out), fixed


def nodes_for_bin(model, index, level):
    old = model.runtime.low_runtime.b_matrices["standard"]
    key = "B000_k_edges"
    edges = old[key][index].ravel()
    orders=ORDERS[level]
    if model.method=="pre":
        orders=(4,80,1,1,1) if level=="primary" else (6,120,1,1,1)
    cmd = [str(OUT/"build/paware_ir_nodes"), *map(str, orders)]
    completed = subprocess.run(cmd, input=f"{index} "+" ".join(map(str,edges))+"\n",
                               text=True, capture_output=True, check=True)
    values = np.loadtxt(io.StringIO(completed.stdout))
    return values[:,1:].reshape(-1,3,3), values[:,0]


def generate_bin(index, level="primary", method="sto_marisa"):
    destination = TEMPLATES/f"{method}_{level}_bin{index:03d}.npz"
    if destination.exists():
        raise FileExistsError(destination)
    sp, _ = spectrum(); source = Sources(); var = Variance()
    model = CorrectedModel(method)
    nodes_by_degree = {d: np.cos(np.pi*(np.arange(d+1)+.5)/(d+1)) for d in (DEGREE,CHECK_DEGREE)}
    nodes = np.concatenate(list(nodes_by_degree.values()))
    bnodes = .1+7.9*(nodes+1)/2
    v,w = nodes_for_bin(model,index,level)
    total = np.zeros((len(nodes),3,27)); fixed = np.zeros(27)
    for start in range(0,len(v),2048):
        sl = slice(start,min(len(v),start+2048))
        raw = source.evaluate(v[sl],sp,pre=method=="pre")
        part,fix = project_raw(v[sl],w[sl],raw,var,bnodes,method)
        total += part; fixed += fix
    if method == "pre":
        # pre Gaussian 按 FFT marginal；PNG 按原有限盒 continuum，不能混换。
        matrix = model.runtime.low_runtime.b_matrices["standard"]
        geom=SimpleNamespace(weighted_pairs=matrix["B000_k"],shell_edges=matrix["B000_k_edges"])
        nrad,nmu = (4,80) if level=="primary" else (6,120)
        ids,triangles,meta,_ = tree_work(geom,nmu=nmu,nradial=nrad,kfund=QMIN,kmax=.14)
        meta=np.asarray(meta);selected=meta[:,0].astype(int)==list(ids).index(index)
        cv=vectors_from_triangles(np.asarray(triangles)[selected]);cw=.5*meta[selected,1]
        raw=source.evaluate(cv,sp,pre=True)
        other,fix=project_raw(cv,cw,raw,var,bnodes,method)
        keep=[*LCOLS,*QCOLS,18,19]
        total[:,:,keep]=other[:,:,keep];fixed[keep]=fix[keep]
    b=2.2; bp=2*DC*(b-model.p)
    mp={"b1":b,**dict.fromkeys(BNAMES,0.)}
    zero=model.parts(mp);shot=model.parts({**mp,"Bshot_residual":1.})
    legacy_stoch=np.array([shot[n][index]-zero[n][index] for n in ("gaussian","linear_coefficient","quadratic_coefficient")])
    adapter=legacy_stoch*model.nbar/(fixed[[17,18,19]]*np.array([b*b,b*bp,bp*bp]))
    values=np.array([combine(total[i],b,model.p,model.nbar,adapter) for i,b in enumerate(bnodes)])
    coefficients={};start=0
    for degree,x in nodes_by_degree.items():
        chunk=values[start:start+degree+1]
        coefficients[f"degree{degree}"]=np.linalg.solve(np.polynomial.chebyshev.chebvander(x,degree),chunk.reshape(degree+1,-1)).reshape(degree+1,5,3)
        start+=degree+1
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination, **coefficients, fixed_unit=fixed, adapter=adapter,
                        weight_sum=w.sum(), node_count=len(v), integrals=total,
                        producer_sha256=np.array(digest(__file__)), library_sha256=np.array(digest(LIBRARY)))
    print("BIN",method,level,index,"nodes",len(v),"adapter",adapter.tolist(),flush=True)


def pk_nodes():
    path=ROOT.parent/"02_reconstruction_on_pk/inputs/p0_finite_local_png_lattice.npz"
    with np.load(path) as z:
        edges=z["pk_edges"];counts=z["pk_nmodes"]
    file=TEMPLATES/"pk_nodes.npz"
    meta=json.loads(file.with_suffix(".json").read_text())
    assert meta["source_sha256"]==digest(path) and meta["export_sha256"]==digest(file)
    with np.load(file) as z:
        v=z["vectors"];ids=z["bin_ids"]
        assert np.array_equal(z["edges"],edges)
    assert np.array_equal(np.bincount(ids,minlength=12),counts)
    return v,ids,counts


def generate_pk(method):
    dest=TEMPLATES/f"{method}_pk.npz"
    if dest.exists():raise FileExistsError(dest)
    sp,_=spectrum();var=Variance();model=CorrectedModel(method)
    v,ids,counts=pk_nodes();k=np.linalg.norm(v,axis=1)
    full=sp.power(k);nw=sp.nowiggle(k);m=sp.transfer(k);mnw=m*np.sqrt(nw/full);pw=full-nw
    arrays={}
    for degree in (DEGREE,CHECK_DEGREE):
        x=np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1));bs=.1+7.9*(x+1)/2
        rows=[]
        for b in bs:
            bp=2*DC*(b-model.p);sig=var.evaluate(v,b,method)
            d=np.exp(-k*k*sig[:,0]);d1=-k*k*sig[:,1];d2=.5*k**4*sig[:,1]**2-k*k*sig[:,2]
            g_w=b*b*pw;l_w=b*bp*pw/mnw
            zero=np.array([b*b*full,2*b*bp*full/m,bp*bp*full/(m*m)])
            ir=np.array([b*b*nw+d*g_w,2*b*bp*nw/mnw+d*(l_w+d1*g_w),
                          bp*bp*nw/mnw**2+d*(d1*l_w+d2*g_w)])
            rows.append(np.array([np.bincount(ids,weights=r,minlength=12)/counts for r in ir-zero]))
        arrays[f"degree{degree}"]=np.linalg.solve(np.polynomial.chebyshev.chebvander(x,degree),np.asarray(rows).reshape(degree+1,-1)).reshape(degree+1,3,12)
    np.savez_compressed(dest,**arrays)
    print("PK",method,"modes",len(v),flush=True)


class Delta:
    degree=DEGREE
    def __init__(self,method):
        p=TEMPLATES/f"{method}_accepted.npz"
        self.contract=json.loads(p.with_suffix(".json").read_text())
        if self.contract["status"]!="pass":raise RuntimeError("IR 模板尚未通过验收")
        with np.load(p) as z:
            self.bcoef=z["b_primary"];self.bcheck=z["b_check"]
            self.pcoef=z["p_primary"];self.pcheck=z["p_check"]
    def b_design(self,b,validation=False):
        return np.polynomial.chebyshev.chebval(2*(b-.1)/7.9-1,self.bcheck if validation else self.bcoef)
    def p_design(self,b,validation=False):
        return np.polynomial.chebyshev.chebval(2*(b-.1)/7.9-1,self.pcheck if validation else self.pcoef)


def load_delta(method):return Delta(method)


def accept(method):
    """只在独立 b1 插值与 shell 收敛均通过时开放 IR MCMC。"""
    from scipy.linalg import solve_triangular
    model=CorrectedModel(method);idx=model.indices
    stacks={};files={}
    for level in ("primary","fine"):
        for degree in (DEGREE,CHECK_DEGREE):
            arrays=[]
            for i in idx:
                p=TEMPLATES/f"{method}_{level}_bin{i:03d}.npz"
                with np.load(p) as z:
                    if str(z["producer_sha256"])!=digest(__file__):
                        raise RuntimeError(f"生成代码哈希不一致: {p}")
                    arrays.append(z[f"degree{degree}"])
                files[p.name]=digest(p)
            stacks[(level,degree)]=np.stack(arrays,axis=-1)
    with np.load(TEMPLATES/f"{method}_pk.npz") as z:
        pc=z[f"degree{DEGREE}"];ph=z[f"degree{CHECK_DEGREE}"]
    old=OUT.parent/"paware_pb_mcmc_k012_k014_v1"/f"{method}_k0p14"
    with np.load(old/"likelihood.npz") as z:
        chol=z["chol"];centre=z["centre"];transform=z["transform"]
    rng=np.random.default_rng(26260930)
    test=centre+rng.normal(size=(128,7))@transform.T*2
    test=np.vstack((centre,test))
    def vector(coef,params):
        f,b=params[:2]
        d=np.polynomial.chebyshev.chebval(2*(b-.1)/7.9-1,coef)
        return np.einsum("a,j,aji->i",np.r_[1.,params[3:]],[1.,f,f*f],d)
    quadrature=[]
    for row in test:
        difference=vector(stacks[("fine",DEGREE)],row)-vector(stacks[("primary",DEGREE)],row)
        w=solve_triangular(chol,np.r_[np.zeros(12),difference],lower=True)
        quadrature.append((float(max(abs(w))),float(np.linalg.norm(w))))
    wide=np.tile(centre,(201,1))
    wide[:,0]=rng.uniform(-299,299,201);wide[:,1]=rng.uniform(.1001,7.9999,201)
    wide[:,3:5]=rng.uniform([-3.9,-1.9],[5.9,3.9],(201,2))
    interpolation=[]
    for row in wide:
        f,b=row[:2];x=2*(b-.1)/7.9-1
        bd=vector(stacks[("fine",CHECK_DEGREE)],row)-vector(stacks[("fine",DEGREE)],row)
        pd=np.array([1.,f,f*f]) @ (np.polynomial.chebyshev.chebval(x,ph)-np.polynomial.chebyshev.chebval(x,pc))
        interpolation.append(float(max(abs(solve_triangular(chol,np.r_[pd,bd],lower=True)))))
    report={"status":"pending","method":method,"b_indices":idx.tolist(),
        "quadrature_max_whitened_component":max(x[0] for x in quadrature),
        "quadrature_max_whitened_norm":max(x[1] for x in quadrature),
        "interpolation_max_whitened_component":max(interpolation),
        "quadrature_tolerance_component":.05,"quadrature_tolerance_norm":.1,"interpolation_tolerance":1.e-5,
        "production_projection":"fine", "post_orders":ORDERS,
        "pre_orders":{"primary":[4,80],"fine":[6,120]},
        "model":"strict O(Pw), O(fNL^2), Gaussian linear displacement; P and B both resummed",
        "anchor":"corrected inherited fixed-order model + same-node (strict IR - full fixed-order) difference",
        "IR_off":"original full fixed order; strict Sigma=0 still differs by O(Pw^2)",
        "variance_contract":json.loads((TEMPLATES/"variance.json").read_text()),
        "source_correction":model.provenance(),"files":files,
        "shift_noise":"not added; no measured weighted shift-noise spectrum supplied",
        "catalogue_status":"legacy-path comparison, not fixed-f_rec=100 catalogue validation"}
    passed=report["quadrature_max_whitened_component"]<.05 and report["quadrature_max_whitened_norm"]<.1 and max(interpolation)<1.e-5
    report["status"]="pass" if passed else "failed_numerical_gate"
    destination=TEMPLATES/f"{method}_accepted.npz"
    if destination.exists():raise FileExistsError(destination)
    save_json(destination.with_suffix(".json"),report)
    if not passed:
        raise RuntimeError(f"IR 数值验收未通过: {method} {report['quadrature_max_whitened_norm']} {max(interpolation)}")
    np.savez_compressed(destination,b_primary=stacks[("fine",DEGREE)],b_check=stacks[("fine",CHECK_DEGREE)],
                        p_primary=pc,p_check=ph)
    print("ACCEPTED",method,report["quadrature_max_whitened_norm"],max(interpolation),flush=True)


def _job(spec):
    method,level,index=spec
    dest=TEMPLATES/f"{method}_{level}_bin{index:03d}.npz"
    if dest.exists():
        with np.load(dest) as z:
            if str(z["producer_sha256"])!=digest(__file__):raise RuntimeError(f"stale partial: {dest}")
        return str(dest)
    generate_bin(index,level,method)
    return str(dest)


def all_templates(workers):
    if not 1<=workers<=16:raise ValueError("workers must be 1..16")
    for name in ("BASELINE_KERNEL_AUDIT.json","IR_DRESSING_AUDIT.json"):
        if json.loads((OUT/name).read_text())["status"]!="pass":raise RuntimeError(name)
    indices=CorrectedModel("sto_marisa").indices
    tasks=[(method,level,int(i)) for method in ("pre","sto_marisa") for level in ORDERS for i in indices]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_job,tasks):print("DONE",result,flush=True)
    for method in ("pre","sto_marisa"):
        if not (TEMPLATES/f"{method}_pk.npz").exists():generate_pk(method)
        accept(method)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("variance","bin","pk","accept","all"))
    parser.add_argument("--method",choices=("pre","sto_marisa"),default="sto_marisa")
    parser.add_argument("--level",choices=tuple(ORDERS),default="primary")
    parser.add_argument("--index",type=int)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    if args.action=="variance":build_variance()
    elif args.action=="pk":generate_pk(args.method)
    elif args.action=="accept":accept(args.method)
    elif args.action=="all":all_templates(args.workers)
    else:generate_bin(args.index,args.level,args.method)


if __name__=="__main__":main()
