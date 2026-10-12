"""模块大纲：执行残余长位移六项独立验收并保存受限状态。"""
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import unittest
from rsd_b000_contract import PART,digest
import test_post_rsd_displacement


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('too many CPUs')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_post_rsd_displacement))
    out={'status':'GAUSSIAN_LONG_DISPLACEMENT_TEST_PASS' if result.wasSuccessful() else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'tests_run':result.testsRun,'log':stream.getvalue(),
         'CPU_affinity':sorted(os.sched_getaffinity(0)),
         'scope':'linear-Gaussian long displacement, physical RSD plus frozen same-shift, continuous main image',
         'full_B_IR_implemented':False,'source_wiggle_dressing_verified':False,
         'reconstruction_shift_noise_validated':False,'PNG_higher_cumulants_included':False,
         'hashes':{x:digest(PART/'code'/x) for x in ('post_rsd_displacement.py','test_post_rsd_displacement.py','audit_post_rsd_displacement.py')}}
    path=PART/'outputs/sto_marisa_post_rsd_mean100_v1/DISPLACEMENT_AUDIT.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
