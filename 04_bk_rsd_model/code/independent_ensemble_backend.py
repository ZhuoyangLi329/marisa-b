"""模块大纲：原emcee moves/RNG的单ensemble文件；仅改变进程/文件布局。

四ensemble不能同时写一个HDF。每个writer独占自己的文件，诊断只在
所有worker退出之后读取。seed、32walkers、Stretch80%/DE20%与原代码一致。
"""
import json
from pathlib import Path
import numpy as np
import emcee
from hdf_lock_retry import open_with_lock_retry


def advance_ensemble(case, filename, ensemble, case_index, steps, progress=None):
    original_open=emcee.backends.HDFBackend.open
    def opened(self,mode='r'):return open_with_lock_retry(lambda:original_open(self,mode))
    emcee.backends.HDFBackend.open=opened
    try:
        backend=emcee.backends.HDFBackend(str(filename),name=f'ensemble_{ensemble}')
        if not backend.initialized:backend.reset(32,case.ndim)
        initial_steps=backend.iteration
        if initial_steps>steps:raise ValueError('worker cannot shorten an existing chain')
        seed=20261001+100003*case_index+1013*ensemble
        sampler=emcee.EnsembleSampler(32,case.ndim,case.log_prob,vectorize=True,backend=backend,
            moves=[(emcee.moves.StretchMove(a=2),.8),(emcee.moves.DEMove(),.2)])
        if initial_steps==0:
            initial=case.initial(seed,ensemble,case.initialize())
            sampler.random_state=np.random.RandomState(seed+19).get_state()
        else:initial=backend.get_last_sample()
        # sampler.sample与原run_mcmc调用同一generator；这里只多了writer内计数。
        for offset,_ in enumerate(sampler.sample(initial,iterations=steps-initial_steps,
                progress=False,skip_initial_state_check=True),1):
            if progress is not None and (offset==1 or (initial_steps+offset)%1000==0):
                progress(initial_steps+offset)
        if progress is not None:progress(backend.iteration)
        return backend.iteration
    finally:emcee.backends.HDFBackend.open=original_open
