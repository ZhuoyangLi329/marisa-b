"""模块大纲：原provisional likelihood不改；只修正boundary初始化，保留v1零步失败。

独立v2链目录；原sampler kernel/seed/收敛门保持。初始affine尺度按既有prior
宽度condition，物理起点反射进prior。没有改变/regularize科学C_single。
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import unittest
import numpy as np
import h5py
import run_sto_post_provisional_eft_mcmc_v1 as base
from provisional_prior_safe_initialization import bounded_affine_transform,physical_initial_cloud

ROOT=base.ROOT;OLD=base.OUT;OUT=ROOT/'diagnostic_provisional_eft12_pb_20261010_v2'


class PriorSafeCase(base.ProvisionalCase):
    def initialize(self):
        cached=OLD/'cases'/self.directory.name/'MAP.json'
        if not (self.directory/'MAP.json').exists() and cached.exists():
            m=base.read(cached)
            if m['likelihood_sha256']!=self.meta['likelihood_sha256']:
                raise ValueError('cannot reuse blind MAP from a different actual likelihood')
            self.center=np.array(m['center']);self.transform=np.array(m['transform'])
            m=dict(m,blind_MAP_reused_from_same_likelihood=str(cached),
                reused_blind_MAP_sha256=base.digest(cached))
        else:m=super().initialize()
        if m.get('prior_safe_initialization'):return m
        base.shared.save_json(self.directory/'MAP_HESSIAN_PROPOSAL_ORIGINAL.json',m)
        transform,conditioning=bounded_affine_transform(np.array(m['transform']),self.bounds)
        self.transform=transform;m=dict(m,transform=transform.tolist(),prior_safe_initialization=True,
            affine_conditioning=conditioning,likelihood_prior_sampler_moves_RNG_diagnostics_unchanged=True)
        base.shared.save_json(self.directory/'MAP.json',m);return m

    def initial(self,seed,ensemble,mapmeta):
        best=np.array(mapmeta['center'])
        good=[r for r in mapmeta['candidates'] if r['chi2']<=mapmeta['chi2_MAP']+10 and r['valid']]
        if ensemble>=2 and good:best=np.array(good[(ensemble-2)%len(good)]['parameters'])
        physical=physical_initial_cloud(best,self.transform,self.bounds,seed)
        points=self.coordinates(physical)
        if not np.isfinite(self.log_prob(points)).all():raise ValueError('actual prior-safe walkers not finite')
        np.testing.assert_allclose(self.physical(points),physical,rtol=2e-9,atol=1e-9)
        return points


def setup_route():
    failed=base.read(OLD/'STATUS.json')
    if failed['state']!='FAILED_PRESERVED_REQUIRES_EXACT_PROCESS_INSPECTION' or Path('/proc/'+str(failed['pid'])).exists():
        raise ValueError('v1 must actually be terminal; no duplicate chains')
    if 'unable to initialize walkers within frozen priors' not in failed['error']:
        raise ValueError('this recovery only addresses the verified zero-step initialization failure')
    for p in (OLD/'cases').glob('*/chain.h5'):
        with h5py.File(p,'r') as f:
            if any(int(f[k].attrs['iteration'])!=0 for k in f):raise ValueError('original actual samples exist; do not duplicate')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_provisional_prior_safe_initialization'))
    if not tests.wasSuccessful() or tests.testsRun!=2:raise ValueError('actual prior-safe initialization tests required')
    old=(base.OUT,base.ProvisionalCase,base.save_json);save=base.save_json
    extra=[Path(__file__),base.PART/'code/provisional_prior_safe_initialization.py',
        base.PART/'code/test_provisional_prior_safe_initialization.py',OLD/'STATUS.json',OLD/'CONTRACT.json']
    frozen={str(p):base.digest(p) for p in extra}
    def write(p,value):
        for q,want in frozen.items():
            if base.digest(q)!=want:raise ValueError('v2 initialization route/source drift')
        if Path(p).name=='CONTRACT.json' and Path(p).parent==OUT:
            value=dict(value)
            value['immutable_inputs'].update(frozen);value.update(
                prior_safe_initialization=True,original_v1_zero_step_failure_preserved=True,
                sampler_moves_RNG_diagnostics_unchanged=True,data_covariance_likelihood_priors_unchanged=True)
        save(p,value)
    base.OUT=OUT;base.ProvisionalCase=PriorSafeCase;base.save_json=write
    return old


def finish_partial_prepare():
    """三case已写完、CASES/parent尚未commit时，只补控制文件；likelihood不重写。

    v2首次prepare的save wrapper误将CASES list转dict，发生在parent commit前。
    逐字节比对v1科学输入与三case，缺任何项或出现实际chain就拒绝恢复。
    """
    if not OUT.exists() or (OUT/'CONTRACT.json').exists():return False
    if any((OUT/'cases'/name/'chain.h5').exists() for name in base.CASES):
        raise ValueError('unexpected actual v2 chain before parent commit')
    previous=base.read(OLD/'CONTRACT.json')
    for p,want in previous['immutable_inputs'].items():
        if base.digest(p)!=want:raise ValueError('original scientific input drift during partial prepare')
    for name in base.CASES:
        for file in ('likelihood.npz','contract.json'):
            if base.digest(OUT/'cases'/name/file)!=base.digest(OLD/'cases'/name/file):
                raise ValueError('partial prepare is not byte-identical to original case')
    base.save_json(OUT/'CASES.json',base.CASES)
    value=dict(previous)
    prefix=str(OLD)+'/'
    value['immutable_inputs']={(str(OUT)+'/'+p[len(prefix):] if p.startswith(prefix) else p):want
        for p,want in previous['immutable_inputs'].items()}
    value['partial_prepare_recovered_without_rewriting_likelihood_or_priors']=True
    base.save_json(OUT/'PREPARE_RECOVERY.json',dict(
        verified_error='save wrapper attempted dict(CASES list) before parent commit',
        scientific_case_bytes_identical=True,existing_likelihood_or_prior_files_overwritten=False,
        sampling_steps_before_recovery=0,created_UTC=base.stamp()))
    base.save_json(OUT/'CONTRACT.json',value)
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={13}:raise ValueError('login26 CPU13 single core required')
    if args.detach:
        log=ROOT/'diagnostic_provisional_eft12_pb_20261010_v2.log';launch=ROOT/'diagnostic_provisional_eft12_pb_20261010_v2.launch.json'
        if log.exists() or launch.exists():raise ValueError('inspect exact original v2 before any recovery')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        base.save_json(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[13],created_UTC=base.stamp(),log=str(log)))
        print(json.dumps(base.read(launch)),flush=True);return
    before=setup_route()
    try:
        recovered=finish_partial_prepare()
        if args.prepare_only:
            if not recovered:base.freeze()
            print('PROVISIONAL_V2_SAME_LIKELIHOOD_PRIORS_FROZEN_NOT_SAMPLED',flush=True)
        else:base.pipeline()
    finally:base.OUT,base.ProvisionalCase,base.save_json=before


if __name__=='__main__':main()
