"""模块大纲：给v1完整链审计增加与执行器名字无关的实际owner终止门。"""
import os
import socket
import unittest
import numpy as np
import emcee
import audit_sto_latest_pb_inference_v1 as original
from post_rsd_owner_quiescence import owner_quiescent


def main():
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={12}:raise ValueError('login26 CPU12 only')
    tests=unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_post_rsd_owner_quiescence'))
    if not tests.wasSuccessful() or tests.testsRun!=2:raise ValueError('actual owner gate regression required')
    status=original.read(original.OUT/'STATUS.json')
    if not owner_quiescent(status):
        print('NOT_READY_ACTUAL_OWNER_STILL_LIVE_OR_PAUSED_NO_HDF_READ',flush=True);return
    for progress in (original.OUT/'pb_cases').glob('*/PROGRESS_ensemble_*.json'):
        if not owner_quiescent(original.read(progress)):
            print('NOT_READY_ACTUAL_ENSEMBLE_WRITER_STILL_LIVE_NO_HDF_READ',flush=True);return
    if status['state']!='LATEST_COV_ALL3_PB_TRIALS_FINISHED_REVIEW_CONVERGENCE':
        print('NOT_READY_REAL_THREE_CASES_NOT_FINISHED_NO_HDF_READ',flush=True);return
    # 同一最新cohort的calibration/P0也从真实四ensemble回放，不只看converged标签。
    checks=[]
    for index,name in enumerate(original.run.pbase.NAMES):
        directory=original.run.POUT/'cases'/name
        factory=original.run.pbase.CalibrationCase if index==0 else original.run.pbase.POnlyCase
        case=factory(directory);case.initialize();stored=original.read(directory/'SUMMARY.json')
        backends=[emcee.backends.HDFBackend(str(directory/'chain.h5'),name=f'ensemble_{j}') for j in range(4)]
        chains=np.stack([b.get_chain() for b in backends]);accepted=np.stack([b.accepted for b in backends])
        if chains.shape!=(4,stored['steps'],32,3):raise ValueError('actual four32walker controls required')
        replay,_=original.sampler.diagnose(case,chains,accepted,stored['steps'])
        gates=dict(replay['gates'])
        if index==0:gates['physical_p_ESS']=gates.pop('fNL_ESS')
        if gates!=stored['gates'] or replay['converged']!=stored['converged']:
            raise ValueError('control original sampling gate replay mismatch')
        np.testing.assert_allclose(replay['quantiles_2p5_16_50_84_97p5'],
            stored['quantiles_2p5_16_50_84_97p5'],rtol=2e-13,atol=1e-12)
        checks.append(dict(name=name,actual_steps=stored['steps'],converged=bool(replay['converged']),
            original_gates=gates,actual_n_covariance=case.meta['n_covariance']))
    original.save_json(original.OUT/'P_CONTROL_COMPLETION_AUDIT.json',dict(
        status='LATEST402_PHYSICAL_P_AND_THREE_P0_ACTUAL_HDF_GATE_REPLAY',checks=checks,
        all_converged=all(x['converged'] for x in checks),created_UTC=original.stamp(),
        audit_source_sha256=original.digest(__file__),full_STO_noise_closure_certified=False))
    original.completed_audit()


if __name__=='__main__':main()
