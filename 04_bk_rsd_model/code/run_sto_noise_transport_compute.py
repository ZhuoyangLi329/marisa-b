"""模块大纲：compute shared32/60G，单核按bin并行，内存4GiB/worker最多14。

单独理论任务；不修改正在排队的700盒测量job或其冻结源码。
396节点/88细网格probe全部通过后聚合44-bin；失败保留，不假完成/不自动重启。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
from build_post_rsd_noise_transport_templates import OUT,PART,ROOT,save_json,digest


def capacity(allowed,memory_mib):
    if len(allowed)>32 or not allowed:raise ValueError('compute CPU allocation must be 1..32 logical cores')
    result=min(len(allowed)-1,(int(memory_mib)-4096)//4096)
    if result<1:raise ValueError('need separate orchestration core and >=8GiB allocation')
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('theory template batch requires compute allocation')
    cpus=sorted(os.sched_getaffinity(0));memory=int(os.environ['SLURM_MEM_PER_NODE']);workers=capacity(cpus,memory)
    # parent亦单核；每个child绑定分配内的独立CPU，没有BLAS/FFT多线程。
    os.sched_setaffinity(0,{cpus[-1]})
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'COMPUTE_STATUS.json').exists():raise RuntimeError('prior batch execution exists; verify terminal identity before recovery')
    for name in ('COMPLETION_AUDIT.json','OPERATOR_COEFFICIENTS.npz'):
        if (OUT/name).exists():raise RuntimeError('preserve existing complete/aggregate operator outputs')
    log=OUT/'PLAN_VALIDATION.log'
    command=[sys.executable,'-B','-u',str(PART/'code/build_post_rsd_noise_transport_templates.py')]
    with log.open('xb') as stream:
        subprocess.run(command+['--plan-only'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    cp=OUT/'CONTRACT.json';contract=json.loads(cp.read_text());csha=digest(cp)
    sources={str(PART/'code/run_sto_noise_transport_compute.py'):digest(Path(__file__)),
             str(PART/'code/submit_sto_noise_transport_compute.slurm'):digest(PART/'code/submit_sto_noise_transport_compute.slurm')}
    state=dict(state='RUNNING',host=socket.gethostname(),pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],
        requested_CPU_limit=32,allocation_CPUs=cpus,parent_CPU=cpus[-1],worker_count_limit=workers,
        memory_MiB=memory,reserved_memory_MiB_per_worker=4096,contract_sha256=csha,
        started_UTC=datetime.now(timezone.utc).isoformat(),completed_bins=[],active_workers=[],queue_sources=sources)
    pending=list(contract['indices']);active={};logs={};failure=None
    def update():
        state['active_workers']=[dict(bin_index=index,pid=p.pid,CPU=cpu) for index,(p,cpu) in active.items()]
        state['updated_UTC']=datetime.now(timezone.utc).isoformat();save_json(OUT/'COMPUTE_STATUS.json',state)
    def guard():
        if digest(cp)!=csha:raise ValueError('frozen operator contract changed')
        for file,want in dict(contract['immutable_sources'],**sources).items():
            if digest(file)!=want:raise ValueError('frozen compute/operator source changed: '+file)
    guard();update()
    while pending or active:
        for index,(proc,cpu) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            logs.pop(index).close();del active[index]
            if rc:
                failure=dict(bin_index=index,exit_code=rc);pending.clear()
            else:
                meta=OUT/'bins'/f'bin{index:03d}.json';record=json.loads(meta.read_text())
                if record['contract_sha256']!=csha or digest(meta.with_suffix('.npz'))!=record['coefficients_sha256']:
                    failure=dict(bin_index=index,error='completed worker bin hash drift');pending.clear()
                else:state['completed_bins'].append(index)
        if failure:state['state']='DRAINING_AFTER_WORKER_FAILURE';state['failure']=failure
        while pending and not failure and len(active)<workers:
            guard();index=pending.pop(0);used={cpu for _,cpu in active.values()}
            cpu=next(c for c in cpus[:-1] if c not in used)
            worker_log=OUT/'workers'/f'bin{index:03d}.log';stream=worker_log.open('xb')
            try:
                proc=subprocess.Popen(command+['--bin-index',str(index)],stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,
                    preexec_fn=lambda cpu=cpu:os.sched_setaffinity(0,{cpu}))
            except Exception:stream.close();raise
            active[index]=(proc,cpu);logs[index]=stream
        update()
        if active:time.sleep(1.)
    if failure:
        state['state']='FAILED';update();raise RuntimeError('worker failed; preserved all outputs, no aggregate/C500/MCMC release')
    guard();indices=contract['indices'];rows=[];coefficients=[]
    if sorted(state['completed_bins'])!=sorted(indices):raise ValueError('not all44 original bins completed')
    for index in indices:
        target=OUT/'bins'/f'bin{index:03d}.npz';meta=json.loads(target.with_suffix('.json').read_text())
        if meta['contract_sha256']!=csha or digest(target)!=meta['coefficients_sha256']:
            raise ValueError('post-worker bin aggregate input drift')
        if len(meta['probes'])!=2 or meta['max_probe_scaled_error']>contract['scaled_tolerance']:
            raise ValueError('full original fine-probe gate required')
        with np.load(target,allow_pickle=False) as z:
            if str(z['contract_sha256'])!=csha:raise ValueError('bin tensor marker drift')
            coefficients.append(np.array(z['coefficients']))
        rows.append(meta)
    aggregate=OUT/'OPERATOR_COEFFICIENTS.npz'
    np.savez_compressed(aggregate,coefficients=np.stack(coefficients,axis=-1),indices=indices,
        contract_sha256=np.array(csha),b1_nodes=contract['b1_nodes'],physical_p_nodes=contract['physical_p_nodes'])
    report=dict(status='ALL44_NOISE_TRANSPORT_OPERATOR_TEMPLATES_COMPLETE_NOT_STO_CLOSURE',
        created_UTC=datetime.now(timezone.utc).isoformat(),bins=44,nodes=396,fine_probes=88,rows=rows,
        contract_sha256=csha,coefficients_sha256=digest(aggregate),
        max_probe_scaled_error=max(row['max_probe_scaled_error'] for row in rows),
        maximum_cache_bytes=max(row['cache_peak_bytes'] for row in rows),
        actual_STO_noise_amplitudes_validated=False,formal_estimator_choice_validated=False,MCMC_released=False,IR=False)
    save_json(OUT/'COMPLETION_AUDIT.json',report);state['state']='COMPLETE';update()
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)


if __name__=='__main__':main()
