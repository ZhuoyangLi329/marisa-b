#!/usr/bin/env python3
"""模块大纲：只用现有实空间测量，V=1的P0及4+3 P+B重拟合。

读取旧700/750等测量矩阵中的指定100盒，不重构、不测量、不生成理论。
复用已保存的pre与源核修正后的STO+Marisa no-IR模板、原先验和采样器。
缺少同配方的STO+Marisa -100只列明缺项，不替换方法或另产样本。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
from iminuit import Minuit
from scipy.stats import chi2

HERE=Path(__file__).resolve().parents[1]
PART=HERE.parents[1]
sys.path.insert(0,str(HERE/'code'))
import run_paware_pb_mcmc as legacy

OUT=HERE/'outputs/realspace_V1_existing_measurements_20261011_v1'
ENTRY=Path(__file__).resolve()
VENDOR=HERE/'vendor/data'
OLD=HERE/'outputs/paware_pb_mcmc_k012_k014_v1'
CORRECTED=HERE/'outputs/paware_pb_ir_brecfix_k012_k014_v2'
TRUTHS={'fiducial':0.,'LC_p':100.,'LC_m':-100.}
SPEC=[(method,truth,cut) for method,truth in
      [('pre','fiducial'),('pre','LC_p'),('pre','LC_m'),('sto_marisa','fiducial'),('sto_marisa','LC_p')]
      for cut in (None,.08,.14)]
PB_NAMES=tuple(legacy.NAMES)
CPUS=(8,9,10,11,12,13,14)

def read(task_file):return json.loads(Path(task_file).read_text())
def stamp():return datetime.now(timezone.utc).isoformat()
def save(task_file,value):legacy.save_json(Path(task_file),value)
def name(spec):
    method,truth,cut=spec
    return method+'_'+truth+('_P0' if cut is None else f'_PB_4plus3_k{cut:.2f}')
def directory(spec):return OUT/'cases'/name(spec)
def arrays(task_file):
    with np.load(task_file,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
def checked(task_directory):
    meta=read(task_directory/'contract.json');task_file=task_directory/'likelihood.npz'
    if legacy.digest(task_file)!=meta['likelihood_sha256']:raise ValueError('旧模板改变')
    return arrays(task_file),meta

class Case:
    def __init__(self,task_directory):
        self.directory=Path(task_directory);self.meta=read(self.directory/'contract.json')
        task_file=self.directory/'likelihood.npz'
        if legacy.digest(task_file)!=self.meta['likelihood_sha256']:raise ValueError('新似然改变')
        for key,value in arrays(task_file).items():setattr(self,key,value)
        self.names=tuple(self.meta['parameter_names']);self.ndim=len(self.names)
        if self.names!=(PB_NAMES[:3] if self.ndim==3 else PB_NAMES):raise ValueError('参数不一致')
        self.inverse_chol=np.linalg.inv(self.chol)

    def predictions(self,values):
        values=np.atleast_2d(values)
        if self.ndim==7:
            return legacy.basis_prediction(values,self.coefficients,self.p_templates,self.meta['p_halo'])
        f,b,shot=values.T;bp=2*1.686*(b-self.meta['p_halo'])
        return (b[:,None]**2*self.p_templates[0]+(2*f*b*bp)[:,None]*self.p_templates[1]
                +(f*bp)[:,None]**2*self.p_templates[2]+shot[:,None])

    def terms(self,values):
        values=np.atleast_2d(values)
        white=(self.predictions(values)-self.target)@self.inverse_chol.T
        prior=np.zeros(len(values))
        if self.ndim==7:
            whitened=(values[:,[1,3,4,5,6]]-self.prior_mean)@self.prior_whitener.T
            prior=self.meta['prior_weight']*np.sum(whitened**2,axis=1)
        return np.sum(white**2,axis=1),prior

    def objective(self,*values):
        data,prior=self.terms(values);return float(data[0]+prior[0])

    def logp(self,coordinates):
        physical=self.centre+np.atleast_2d(coordinates)@self.transform.T
        valid=np.all(np.isfinite(physical)&(physical>self.bounds[:,0])&(physical<self.bounds[:,1]),axis=1)
        result=np.full(len(physical),-np.inf)
        if valid.any():
            data,prior=self.terms(physical[valid]);result[valid]=-.5*(data+prior)
        return result

def target_rows():
    pre=arrays(VENDOR/'PRE_P0_FNL250_MATRIX.npz')
    bis=arrays(VENDOR/'STANDARD_FNL250_MATRIX.npz')
    sto=arrays(VENDOR/'sto_marisa_matrix.npz')
    rows={};sources=[VENDOR/'PRE_P0_FNL250_MATRIX.npz',VENDOR/'STANDARD_FNL250_MATRIX.npz',VENDOR/'sto_marisa_matrix.npz']
    for truth in TRUTHS:
        pi=np.asarray(pre[truth+'_realizations']);bi=np.asarray(bis[truth+'_realizations'])
        if not np.array_equal(pi[:100],np.arange(100)) or not np.array_equal(bi[:100],np.arange(100)):
            raise ValueError('pre前100的编号不一致')
        rows['pre',truth]=(pre[truth+'_pre_P0'][:100],bis[truth+'_pre_B000'][:100])
    rows['sto_marisa','fiducial']=(sto['fiducial_post_P0'][:100,:12],sto['fiducial_post_B000'][:100])
    # +100读取已生产的p-aware实空间P+B，不能用旧p=1.12生产目标冒充。
    root=legacy.original.OLD_HEAD
    import h5py
    pvalues=[];bvalues=[]
    for real in range(100):
        dest=root/f'real{real:03d}';meta_file=dest/'metadata.json'
        legacy.original.validate_metadata(meta_file,real)
        meta=read(meta_file)
        if meta['manifest']['recon_cellsize']!=8.:raise ValueError('原网格不是8Mpc/h')
        if not (dest/'DONE').exists():raise ValueError('原p-aware测量未完成')
        pfile=dest/'post_pk.npz';bfile=dest/'post_b000.h5'
        with np.load(pfile,allow_pickle=False) as data:pvalues.append(data['pk'][:12].copy())
        with h5py.File(bfile,'r') as data:bvalues.append(data['0_0_0/value'][()].copy())
        sources.extend([meta_file,pfile,bfile])
    rows['sto_marisa','LC_p']=(np.asarray(pvalues),np.asarray(bvalues))
    fid={'pre':(pre['fiducial_pre_P0'],bis['fiducial_pre_B000']),
         'sto_marisa':(sto['fiducial_post_P0'][:,:12],sto['fiducial_post_B000'])}
    for pf,bf in fid.values():
        if pf.shape!=(500,12) or bf.shape!=(500,120):raise ValueError('原Fid500形状不一致')
    return rows,fid,sources

def prepare():
    if OUT.exists():raise ValueError('不覆盖已有拟合')
    rows,fid,sources=target_rows()
    OUT.mkdir(parents=True);(OUT/'cases').mkdir()
    old={}
    for method in ('pre','sto_marisa'):
        src=(OLD/'pre_k0p14' if method=='pre' else CORRECTED/'sto_marisa_noir_k0p14')
        data,meta=checked(src);old[method]=(data,meta,src)
        sources.extend([src/'contract.json',src/'likelihood.npz'])
        pf,bf=fid[method];cols=np.concatenate([pf,bf[:,data['indices']]],axis=1)
        np.testing.assert_allclose(np.cov(cols,rowvar=False,ddof=1),data['covariance']*meta['volume'],rtol=2e-13,atol=1e-7)
    # 固定原9/27个B bins，不改分箱密度。
    indices08=np.array([1,2,3,15,16,17,29,30,42])
    summary=[]
    for spec in SPEC:
        method,truth,cut=spec;olddata,oldmeta,src=old[method]
        idx=np.array([],dtype=int) if cut is None else indices08 if cut==.08 else olddata['indices']
        loc=np.array([int(np.flatnonzero(olddata['indices']==value)[0]) for value in idx],dtype=int)
        pf,bf=fid[method];pt,bt=rows[method,truth]
        cols=np.concatenate([pf,bf[:,idx]],axis=1)
        target=np.r_[pt.mean(axis=0),bt.mean(axis=0)[idx]]
        covariance=np.cov(cols,rowvar=False,ddof=1);hartlap=(500-len(target)-2)/499
        ndim=3 if cut is None else 7;names=PB_NAMES[:ndim];bounds=olddata['bounds'][:ndim].copy()
        frozen=dict(target=target,covariance=covariance,chol=np.linalg.cholesky(covariance)/np.sqrt(hartlap),
            p_templates=olddata['p_templates'],bounds=bounds,centre=olddata['centre'][:ndim].copy())
        if cut is not None:
            frozen.update(coefficients=olddata['coefficients'][...,loc],prior_mean=olddata['prior_mean'],
                          prior_whitener=olddata['prior_whitener'],indices=idx)
        dest=directory(spec);dest.mkdir()
        np.savez(dest/'likelihood.npz',**frozen)
        meta=dict(oldmeta,method=method,truth_fNL=TRUTHS[truth],cosmology=truth,
            statistic='P0-only' if cut is None else 'P+B',kmax_B=0. if cut is None else cut,
            kmax_P=.08,volume=1.,covariance_is_of_mean=False,covariance_multiplier=1.,
            n_target=100,n_fid=500,target_realizations=list(range(100)),
            fiducial_target_overlap=truth=='fiducial',parameter_names=names,
            prior_weight=0. if cut is None else oldmeta['prior_weight'],
            IR_resummation=False,new_reconstruction=False,new_measurement=False,
            original_recon_cellsize_Mpc_h=8.,original_measurement_mesh=256,
            source_case=str(src),hartlap=hartlap,likelihood_sha256=legacy.digest(dest/'likelihood.npz'))
        meta.pop('map',None);save(dest/'contract.json',meta)
        case=Case(dest);candidates=[]
        for f0 in (-150.,-100.,0.,100.,150.):
            center=frozen['centre'].copy();center[0]=f0
            fit=Minuit(case.objective,*center,name=names);fit.errordef=1;fit.strategy=2;fit.tol=1e-7
            for j,key in enumerate(names):
                fit.limits[key]=tuple(bounds[j]);fit.errors[key]=(10.,.1,1000.,.1,.1,.1,.1)[j]
            fit.migrad(ncall=100000);fit.hesse()
            if fit.valid and fit.fmin.has_posdef_covar:candidates.append(fit)
        if not candidates:raise ValueError('MAP/HESSE失败: '+name(spec))
        fit=min(candidates,key=lambda item:item.fval);center=np.array(fit.values)
        frozen.update(centre=center,transform=np.linalg.cholesky(np.array(fit.covariance)))
        np.savez(dest/'likelihood.npz',**frozen)
        data_q,prior_q=case.terms(center)
        meta.update(map=dict(zip(names,center)),map_objective=float(fit.fval),map_data_chi2=float(data_q[0]),
            map_prior_chi2=float(prior_q[0]),minuit_valid=True,
            reference_MINOS={'gof_p_value_data_only':float(chi2.sf(data_q[0],len(target)-ndim))},
            likelihood_sha256=legacy.digest(dest/'likelihood.npz'))
        save(dest/'contract.json',meta)
        row=dict(case=name(spec),method=method,truth_fNL=TRUTHS[truth],statistic=meta['statistic'],kmax_B=cut,
                 best_fNL=center[0],Hessian_sigma_fNL=float(fit.errors['fNL']),
                 posterior_sigma_fNL=None,status='MAP_COMPLETE_MCMC_PENDING')
        summary.append(row);save(OUT/'RESULTS.json',{'rows':summary,'volume':1.,'mean_data_count':100,'covariance_count':500})
        print('MAP',json.dumps(legacy.clean(row)),flush=True)
    sources.extend([ENTRY,Path(legacy.__file__),Path(legacy.original.__file__)])
    save(OUT/'SETTINGS.json',dict(immutable_inputs={str(task_file):legacy.digest(task_file) for task_file in sources},
        cases=[name(spec) for spec in SPEC],CPUs=list(CPUS),controller_CPU=7,
        ensembles=4,walkers=32,initial_steps=6000,max_steps=20000,
        missing_case='STO+Marisa -100: no matching pre-existing realspace measurement',
        volume=1.,C_single500=True,target_mean100=True,p_pre=1.179,p_sto_marisa=1.2822,
        IR=False,reconstruction_or_measurement_jobs=0))

def worker(case_name):
    settings=read(OUT/'SETTINGS.json')
    for task_file,want in settings['immutable_inputs'].items():
        if legacy.digest(Path(task_file))!=want:raise ValueError('原输入改变: '+task_file)
    dest=OUT/'cases'/case_name;case=Case(dest)
    legacy.OUT=OUT/'cases';legacy.Case=Case;legacy.NAMES=case.names;legacy.NDIM=case.ndim
    original_diagnostics=legacy.diagnostics;original_save=legacy.save_json;latest={}
    def diagnostics(actual,chain,accepted):
        diag,flat=original_diagnostics(actual,chain,accepted)
        latest.update(sigma=np.std(flat,axis=0),quantiles=np.quantile(flat,[.025,.16,.5,.84,.975],axis=0))
        return diag,flat
    def save_with_sigma(task_file,result):
        if Path(task_file).name=='RESULTS.json':
            result['posterior_sigma']={key:float(value) for key,value in zip(case.names,latest['sigma'])}
            result['truth_in_68']=bool(latest['quantiles'][1,0]<=case.meta['truth_fNL']<=latest['quantiles'][3,0])
            result['truth_in_95']=bool(latest['quantiles'][0,0]<=case.meta['truth_fNL']<=latest['quantiles'][4,0])
        original_save(task_file,result)
    legacy.diagnostics=diagnostics;legacy.save_json=save_with_sigma
    legacy.sample(case_name)

def collect():
    rows=[]
    for spec in SPEC:
        dest=directory(spec);meta=read(dest/'contract.json')
        row=dict(case=name(spec),method=spec[0],truth_fNL=TRUTHS[spec[1]],statistic=meta['statistic'],kmax_B=spec[2],
            best_fNL=meta['map']['fNL'],Hessian_sigma_fNL=None,posterior_sigma_fNL=None,status='MCMC_RUNNING')
        if (dest/'RESULTS.json').exists():
            result=read(dest/'RESULTS.json');q=result['posterior']['fNL']
            row.update(median_fNL=q['median'],q16=q['q16'],q84=q['q84'],posterior_sigma_fNL=result['posterior_sigma']['fNL'],
                status=result['status'],steps=result['diagnostics']['steps'])
        rows.append(row)
    save(OUT/'RESULTS.json',dict(rows=rows,volume=1.,mean_data_count=100,covariance_count=500,
        covariance_of_mean=False,p_pre=1.179,p_sto_marisa=1.2822,PB_parameters='4+3',P_max=.08,B_max=[.08,.14],
        missing='STO+Marisa fNL=-100 existing matched realspace measurement unavailable'))

def run():
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={7}:raise ValueError('controller须login26 CPU7')
    state=dict(pid=os.getpid(),host=socket.gethostname(),state='PREPARING_FITS_ONLY',started_UTC=stamp())
    def update(**fields):state.update(fields,updated_UTC=stamp());save(OUT/'STATUS.json',state)
    try:
        if not (OUT/'SETTINGS.json').exists():prepare()
        update(state='MCMC_RUNNING',completed_cases=[])
        pending=list(SPEC);active={};finished=[]
        while pending or active:
            for spec,(proc,stream,cpu) in list(active.items()):
                if proc.poll() is None:continue
                stream.close();active.pop(spec)
                if proc.returncode!=0:raise ValueError('拟合失败: '+name(spec))
                finished.append(name(spec));collect()
            occupied={value[2] for value in active.values()}
            for cpu in CPUS:
                if not pending:break
                if cpu in occupied:continue
                spec=pending.pop(0);stream=(directory(spec)/'sampling.log').open('x')
                proc=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(ENTRY),'worker','--case',name(spec)],
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                active[spec]=(proc,stream,cpu)
            update(completed_cases=finished,active=[dict(case=name(spec),pid=value[0].pid,cpu=value[2]) for spec,value in active.items()])
            time.sleep(2)
        collect();results=read(OUT/'RESULTS.json')
        update(state='COMPLETE' if all(row['status']=='converged' for row in results['rows']) else 'FINISHED_WITH_CONVERGENCE_WARNINGS',
               finished_UTC=stamp())
    except BaseException:
        if OUT.exists():update(state='FAILED',error=traceback.format_exc())
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','worker','run','detach'))
    parser.add_argument('--case');args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker(args.case)
    elif args.action=='run':run()
    else:
        if OUT.exists():raise ValueError('不重复启动')
        # 日志只属于本轮必要拟合输出，不另建数据生产、模板或报告目录。
        OUT.parent.mkdir(exist_ok=True)
        with (OUT.parent/(OUT.name+'.log')).open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(ENTRY),'run'],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':proc.pid,'host':socket.gethostname(),'output':str(OUT)}),flush=True)
