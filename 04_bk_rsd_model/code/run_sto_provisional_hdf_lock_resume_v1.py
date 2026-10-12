"""模块大纲：核实v2 writer终止后续原HDF；仅加打开锁等待和writer内进度记录。

保持原likelihood/prior/seed/moves/诊断/64k预算；不改变冻结源码或已有样本。
禁止外部实时读写chain.h5；读取SUMMARY/本脚本writer进度不竞争HDF写锁。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import traceback
import unittest
import h5py
import numpy as np
import emcee
import run_sto_post_provisional_eft_mcmc_v2 as route
from hdf_lock_retry import open_with_lock_retry

base=route.base;ROOT=base.ROOT;OUT=ROOT/'provisional_hdf_lock_recovery_v1'


def pipeline():
    if OUT.exists():raise ValueError('inspect and preserve original recovery before any retry')
    old_status=base.read(route.OUT/'STATUS.json')
    if (old_status['state']!='FAILED_PRESERVED_REQUIRES_EXACT_PROCESS_INSPECTION'
            or 'unable to lock file' not in old_status['error']
            or Path('/proc/'+str(old_status['pid'])).exists()):
        raise ValueError('exact original writer must be terminal with verified HDF lock failure')
    parent=base.read(route.OUT/'CONTRACT.json')
    for path,want in parent['immutable_inputs'].items():
        if base.digest(path)!=want:raise ValueError('frozen v2 scientific input drift')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_hdf_lock_retry'))
    if not tests.wasSuccessful() or tests.testsRun!=3:raise ValueError('three native HDF lock tests required')
    checkpoints={}
    for d in (route.OUT/'cases').iterdir():
        h=d/'chain.h5'
        if not h.exists():continue
        mp=base.read(d/'MAP.json');center=np.array(mp['center']);transform=np.array(mp['transform'])
        with np.load(d/'likelihood.npz',allow_pickle=False) as z:bounds=z['bounds']
        with h5py.File(h,'r') as f:
            checkpoints[d.name]={}
            for name in f:
                g=f[name];n=int(g.attrs['iteration']);coordinate=g['chain'][n-1];prob=g['log_prob'][n-1]
                physical=coordinate@transform.T+center
                if (n<=0 or not np.isfinite(coordinate).all() or not np.isfinite(prob).all()
                        or not np.all(physical>bounds[:,0]) or not np.all(physical<bounds[:,1])
                        or 'random_state_0' not in g.attrs):raise ValueError('last32/RNG checkpoint invalid')
                checkpoints[d.name][name]=n
    OUT.mkdir();base.save_json(OUT/'FAILED_ORIGINAL_STATUS.json',old_status)
    paths=[Path(__file__),base.PART/'code/hdf_lock_retry.py',base.PART/'code/test_hdf_lock_retry.py']
    frozen={str(q):base.digest(q) for q in paths}
    base.save_json(OUT/'CONTRACT.json',dict(status='SAME_V2_HDF_CHECKPOINT_RESUME_IO_LOCK_WAIT_ONLY',
        failed_original_pid=old_status['pid'],failed_original_starttime_ticks='151284834',
        original_writer_confirmed_absent=True,initial_iterations=checkpoints,
        frozen_parent_sha256=base.digest(route.OUT/'CONTRACT.json'),runtime_source_sha256=frozen,
        actual_native_lock_tests_passed=3,existing_samples_reset=False,HDF_locks_disabled=False,
        likelihood_covariance_priors_moves_RNG_diagnostics_budget_changed=False,max_steps=64000,
        external_running_HDF_inspection_allowed=False,writer_owned_progress_file='HDF_PROGRESS.json'))
    state=dict(state='RESUMING',pid=os.getpid(),host=socket.gethostname(),CPU_affinity=[13],started_UTC=base.stamp())
    base.save_json(OUT/'STATUS.json',state)
    original_open=emcee.backends.HDFBackend.open;original_save=emcee.backends.HDFBackend.save_step
    counters={}; progress={}; lock_retries=[]
    def opened(self,mode='r'):
        def event(attempt):lock_retries.append(dict(UTC=base.stamp(),mode=mode,attempt=attempt))
        return open_with_lock_retry(lambda:original_open(self,mode),on_retry=event)
    def saved(self,sample,accepted):
        original_save(self,sample,accepted)
        key=(str(self.filename),str(self.name))
        current=counters[key]+1 if key in counters else int(self.iteration)
        counters[key]=current
        if key not in progress or current%2000==0:
            progress[key]=current
            base.save_json(OUT/'HDF_PROGRESS.json',dict(pid=os.getpid(),updated_UTC=base.stamp(),
                progress=[dict(filename=k[0],ensemble=k[1],actual_writer_saved_steps=v) for k,v in progress.items()],
                original_likelihood_priors_unchanged=True,external_HDF_opened=False))
    emcee.backends.HDFBackend.open=opened;emcee.backends.HDFBackend.save_step=saved
    try:
        before=route.setup_route()
        try:base.pipeline()
        finally:base.OUT,base.ProvisionalCase,base.save_json=before
        for path,want in frozen.items():
            if base.digest(path)!=want:raise ValueError('IO recovery runtime source drift')
        state.update(state='ORIGINAL3_PROVISIONAL_CASES_FINISHED_REVIEW_CONVERGENCE',finished_UTC=base.stamp())
    except Exception:
        state.update(state='FAILED_PRESERVED_REQUIRES_EXACT_PROCESS_INSPECTION',error=traceback.format_exc());raise
    finally:
        emcee.backends.HDFBackend.open=original_open;emcee.backends.HDFBackend.save_step=original_save
        base.save_json(OUT/'LOCK_RETRIES.json',lock_retries);base.save_json(OUT/'STATUS.json',state)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('login26 CPU13 single core required')
    if args.detach:
        log=ROOT/'provisional_hdf_lock_recovery_v1.log';launch=ROOT/'provisional_hdf_lock_recovery_v1.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect original recovery before duplicate launch')
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        base.save_json(launch,dict(pid=process.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=base.stamp(),log=str(log)))
        print(json.dumps(base.read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
