"""模块大纲：以实空间已用于p-aware拟合的p=1.2822重跑九组post-RSD。

复用上一轮模型、MAP、独立ensemble后端、原诊断及完整链审计。
只路由新输出，并替换物理p及B的p轴；实际data/C/先验/窗口逐位继承。
先三组P0-only，再两档六组4+4 P+B；登录节点四单核worker。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import json
import signal
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
import run_postP_calibrated_rsd_v1 as core
from post_rsd_noir_polynomial import collapse_physical_p

BASE=core.OUT
BASE_NAMES={i:core.directory(i).name for i in range(1,10)}
OUT=core.PART/'outputs/sto_marisa_realspace_p_rsd_20261011_v1'
REALSPACE=core.PART.parent/'03_bk_model/envclass_b/outputs/paware_pb_ir_brecfix_k012_k014_v2/SUMMARY.json'
REALCAL=core.PART.parent/'02_reconstruction_on_pk/work/quijote_halo_z0p5_sto_marisa_fnlfixed100_freep_calib_v1/FREEP_RESULTS.json'
PHYSICAL_P=1.2822
CPUS=core.CPUS

def directory(index): return OUT/'cases'/BASE_NAMES[index].replace('_postPcal','_realspaceP')

# 适配只路由输出与目录，原模块文件不变；其模型、RNG和诊断完全复用。
core.OUT=OUT
core.directory=directory

def freeze():
    if (OUT/'SETTINGS.json').exists(): core.guard();return
    OUT.mkdir(exist_ok=True)
    deployed=core.read(REALSPACE)
    rows=deployed['old_comparison_rows']
    if not isinstance(rows,list): raise ValueError('需核对实空间结果结构')
    matched=[r for r in rows if r.get('method')=='sto_marisa']
    if not matched or any(r.get('p')!=PHYSICAL_P for r in matched): raise ValueError('实空间实际p不是1.2822')
    source_settings=core.read(BASE/'SETTINGS.json')
    for path,want in source_settings['immutable_inputs'].items():
        if core.digest(path)!=want: raise ValueError('前一轮来源已改变: '+path)
    choice=dict(physical_p=PHYSICAL_P,p_origin='real-space STO+Marisa calibrated p deployed in p-aware fits',
        estimator='fixed calibrated value used in real-space analyses',RSD_p_recalibrated_in_this_run=False,
        realspace_p_source=str(REALSPACE),realspace_p_source_sha256=core.digest(REALSPACE),
        original_realspace_calibration=str(REALCAL),original_realspace_calibration_sha256=core.digest(REALCAL),
        baseline_RSD_p=core.read(BASE/'P_CHOICE.json')['physical_p'],p_uncertainty_propagated=False)
    core.save_json(OUT/'P_CHOICE.json',choice)
    sources=[Path(__file__),REALSPACE,REALCAL,BASE/'SETTINGS.json',BASE/'P_CHOICE.json',OUT/'P_CHOICE.json']
    sources += [Path(p) for p in source_settings['immutable_inputs']]
    tensor=core.arrays(core.TENSOR)
    for index in range(1,10):
        src=BASE/'cases'/BASE_NAMES[index];meta=core.read(src/'contract.json')
        if core.digest(src/'likelihood.npz')!=meta['likelihood_sha256']: raise ValueError('基线似然改变')
        if core.digest(src/'SUMMARY.json')!=core.read(src/'COMPLETION_AUDIT.json')['summary_sha256']:
            raise ValueError('基线摘要改变')
        data=core.arrays(src/'likelihood.npz');preserved={k:v.copy() for k,v in data.items()}
        if index>=4:
            loc=np.array([int(np.flatnonzero(tensor['indices']==i)[0]) for i in data['indices']])
            np.testing.assert_allclose(collapse_physical_p(tensor['coefficients'],p_halo=meta['p_fixed_physical'])[...,loc],
                data['coefficients'],rtol=3e-13,atol=1e-8)
            data['coefficients']=collapse_physical_p(tensor['coefficients'],p_halo=PHYSICAL_P)[...,loc]
        for name in data:
            if name!='coefficients':np.testing.assert_array_equal(data[name],preserved[name])
        for name in list(meta):
            if 'calibration' in name or name.startswith('physical_p_'):meta.pop(name)
        meta.update(name=directory(index).name,p_fixed_physical=PHYSICAL_P,
            previous_fixed_p=choice['baseline_RSD_p'],source_case=str(src),
            p_origin=choice['p_origin'],p_estimator=choice['estimator'],p_uncertainty_propagated=False,
            realspace_p_source=str(REALSPACE),realspace_p_source_sha256=core.digest(REALSPACE),
            RSD_p_recalibrated_in_this_run=False,only_changed_fixed_physical_p=True,
            data_covariance_priors_windows_unchanged=True)
        if index>=4:meta['source_P0_case']=str(directory(1+core.TRUTHS.index(core.SPECS[index][1])))
        d=directory(index);d.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(d/'likelihood.npz',**data);meta['likelihood_sha256']=core.digest(d/'likelihood.npz')
        core.save_json(d/'contract.json',meta)
        case_sources=[src/'contract.json',src/'likelihood.npz',src/'SUMMARY.json',src/'COMPLETION_AUDIT.json',
            OUT/'P_CHOICE.json',REALSPACE,REALCAL,core.TENSOR]
        core.save_json(d/'SOURCES.json',{str(p):core.digest(p) for p in case_sources})
        sources += case_sources+[d/'contract.json',d/'likelihood.npz',d/'SOURCES.json']
    core.save_json(OUT/'SETTINGS.json',dict(created_UTC=core.stamp(),physical_p=PHYSICAL_P,
        case_names=[directory(i).name for i in range(1,10)],case_indices=list(range(1,10)),CPUs=list(CPUS),
        main_mean_target_count=100,C_single500=True,volume_Gpc_h3=1.,P_cut=.08,B_cuts=[.08,.14],IR=False,
        p_origin=choice['p_origin'],p_uncertainty_propagated=False,
        ensembles=4,walkers=32,initial_steps=6000,step_increment=2000,max_steps=64000,
        immutable_inputs={str(p):core.digest(p) for p in dict.fromkeys(sources)}))
    for index in range(1,10):
        core.input_audit(index)
        path=directory(index)/'INPUT_AUDIT.json';audit=core.read(path)
        audit.pop('calibration_block',None);audit.update(model_data_block='P_post',physical_p=PHYSICAL_P,
            p_origin='real-space calibrated p',RSD_p_recalibrated_in_this_run=False)
        core.save_json(path,audit)
    print('NINE_REALSPACE_P_INPUTS_VERIFIED',flush=True)

def sample(index,update):
    core.guard();core.verify_sources(index);case=core.case_at(index);mapmeta=case.initialize();started=time.monotonic()
    for steps in range(6000,64001,2000):
        active=[]
        try:
            for j,cpu in enumerate(CPUS):
                stream=(case.directory/f'worker_{j}_to{steps}.log').open('x')
                proc=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-B','-u',str(Path(__file__).resolve()),
                    '--worker',str(index),'--ensemble',str(j),'--steps',str(steps)],stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT)
                active.append((proc,stream));update(current_case=case.directory.name,current_case_index=index,
                    requested_checkpoint_steps=steps,worker_pids=[p.pid for p,_ in active])
            while any(p.poll() is None for p,_ in active):time.sleep(.5)
            if any(p.returncode!=0 for p,_ in active):raise ValueError('采样worker失败，保留结果')
        finally:
            for proc,stream in active:
                if proc.poll() is None:proc.send_signal(signal.SIGINT);proc.wait(timeout=30)
                stream.close()
        backends,chains,accepted=core.load_chains(case)
        if chains.shape!=(4,steps,32,case.ndim):raise ValueError('四ensemble步数不一致')
        report,flat=core.shared.diagnose(case,chains,accepted,steps)
        report.update(case.meta,chi2_MAP=mapmeta['chi2_MAP'],minuit_valid=mapmeta['minuit_valid'],
            elapsed_seconds=time.monotonic()-started,gof_pvalue=None,
            gof_note='mean100 with C_single forecast; goodness-of-fit p-value not calibrated')
        report=core.normalize_report(index,report);core.save_json(case.directory/'SUMMARY.json',report)
        update(last_checked_steps=steps,last_converged=report['converged'],worker_pids=[])
        print(case.directory.name,steps,report['converged'],report['gates'],flush=True)
        if report['converged']:break
    pick=np.random.default_rng(core.thin_seed(index)).choice(len(flat),min(30000,len(flat)),replace=False)
    np.savez_compressed(case.directory/'POSTERIOR_THIN.npz',samples=flat[pick],parameter_names=case.names)
    return core.completed_audit(index)

def pipeline():
    state=dict(pid=os.getpid(),host=socket.gethostname(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        started_UTC=core.stamp(),state='PREPARING_FIXED_REALSPACE_P',completed_cases=[],physical_p=PHYSICAL_P)
    def update(**values):state.update(values,updated_UTC=core.stamp());core.save_json(OUT/'STATUS.json',state)
    update()
    try:
        freeze();audits=[]
        for index in range(1,10):
            update(state='SAMPLING_P0_ONLY' if index<4 else 'SAMPLING_4PLUS4_PB')
            audits.append(sample(index,update));state['completed_cases'].append(directory(index).name);update()
            if index==3:core.save_json(OUT/'P0_RESULTS.json',dict(physical_p=PHYSICAL_P,
                reports=[core.read(directory(i)/'SUMMARY.json') for i in range(1,4)]))
        core.save_json(OUT/'RESULTS.json',dict(physical_p=PHYSICAL_P,
            reports=[core.read(directory(i)/'SUMMARY.json') for i in range(1,10)],audits=audits,
            all_nine_original_sampling_gates_verified=True))
        update(state='ALL_NINE_CHAINS_VERIFIED_COMPARISON_REPORT_REQUIRED',current_case=None,finished_UTC=core.stamp())
    except BaseException:update(state='FAILED_OR_INTERRUPTED_PRESERVED',error=traceback.format_exc());raise

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--detach',action='store_true');ap.add_argument('--worker',type=int)
    ap.add_argument('--ensemble',type=int);ap.add_argument('--steps',type=int);args=ap.parse_args()
    if socket.gethostname()!='login26':raise ValueError('本轮使用login26')
    if args.worker is not None:core.worker(args.worker,args.ensemble,args.steps)
    elif args.prepare_only:freeze()
    elif args.detach:
        if os.sched_getaffinity(0)!={12}:raise ValueError('controller固定单核12')
        OUT.mkdir(exist_ok=True)
        if (OUT/'LAUNCH.json').exists() or (OUT/'controller.log').exists():raise ValueError('已有启动记录，检查实际进程')
        with (OUT/'controller.log').open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        core.save_json(OUT/'LAUNCH.json',dict(pid=proc.pid,host=socket.gethostname(),CPUs=list(CPUS),created_UTC=core.stamp()))
        print(json.dumps(core.read(OUT/'LAUNCH.json')),flush=True)
    else:pipeline()
