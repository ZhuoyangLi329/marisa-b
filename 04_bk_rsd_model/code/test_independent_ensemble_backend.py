"""模块大纲：独立文件与原串行sampler的实际样本/RNG逐位一致；断点也一致。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import emcee
from independent_ensemble_backend import advance_ensemble


class ToyCase:
    ndim=3
    def initialize(self):return {}
    def initial(self,seed,ensemble,meta):return np.random.default_rng(seed).normal(size=(32,3))
    def log_prob(self,x):return -.5*np.sum(x*x,axis=1)


class EnsembleBackendTests(unittest.TestCase):
    def test_exact_original_rng_moves_and_checkpoint(self):
        case=ToyCase()
        with tempfile.TemporaryDirectory(prefix='sto-ensemble-unit-') as taskdir:
            for ensemble in (0,3):
                old=emcee.backends.HDFBackend(str(Path(taskdir)/f'old{ensemble}.h5'),name=f'ensemble_{ensemble}');old.reset(32,3)
                seed=20261001+100003*2+1013*ensemble
                sampler=emcee.EnsembleSampler(32,3,case.log_prob,vectorize=True,backend=old,
                    moves=[(emcee.moves.StretchMove(a=2),.8),(emcee.moves.DEMove(),.2)])
                sampler.random_state=np.random.RandomState(seed+19).get_state()
                sampler.run_mcmc(case.initial(seed,ensemble,{}),40,progress=False,skip_initial_state_check=True)
                path=Path(taskdir)/f'new{ensemble}.h5'
                advance_ensemble(case,path,ensemble,2,17)
                advance_ensemble(case,path,ensemble,2,40)
                new=emcee.backends.HDFBackend(str(path),name=f'ensemble_{ensemble}')
                np.testing.assert_array_equal(old.get_chain(),new.get_chain())
                np.testing.assert_array_equal(old.get_log_prob(),new.get_log_prob())
                np.testing.assert_array_equal(old.accepted,new.accepted)
                np.testing.assert_array_equal(old.random_state[1],new.random_state[1])


if __name__=='__main__':unittest.main()
