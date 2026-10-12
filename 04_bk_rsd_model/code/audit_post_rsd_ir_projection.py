"""模块大纲：IR harmonic/tagged FFT 子系统回归与受限状态保存。"""
from datetime import datetime,timezone
import io
import json
import os
import unittest
from rsd_b000_contract import PART,digest
import test_post_rsd_ir_projection


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('too many CPUs')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_post_rsd_ir_projection))
    out={'status':'IR_HARMONIC_FFT_PROJECTION_TEST_PASS' if result.wasSuccessful() else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'tests_run':result.testsRun,'log':stream.getvalue(),
         'CPU_affinity':sorted(os.sched_getaffinity(0)),
         'scope':'deterministic strict-O(Pw) finite-PNG, synthetic spectrum split, exact small full-shell FFT, LOS=z',
         'full_production_registry_verified':False,'production_parameter_interpolation_verified':False,
         'actual_post_stochastic_validated':False,'MCMC_released':False,
         'hashes':{x:digest(PART/'code'/x) for x in ('post_rsd_displacement.py','post_rsd_ir_harmonics.py',
             'post_rsd_ir_projection.py','test_post_rsd_ir_projection.py','audit_post_rsd_ir_projection.py')}}
    path=PART/'outputs/sto_marisa_post_rsd_mean100_v1/IR_PROJECTION_AUDIT.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
