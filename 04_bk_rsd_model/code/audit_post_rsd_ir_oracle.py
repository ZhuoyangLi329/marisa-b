"""模块大纲：有向 IR source oracle 的受限测试；不放行 likelihood。"""
from datetime import datetime,timezone
import io
import json
import os
import unittest
from rsd_b000_contract import PART,digest
import test_post_rsd_ir_oracle


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('too many CPUs')
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_post_rsd_ir_oracle))
    out={'status':'ORIENTED_IR_SOURCE_ORACLE_PASS' if result.wasSuccessful() else 'REQUIRES_INVESTIGATION',
         'created_UTC':datetime.now(timezone.utc).isoformat(),'tests_run':result.testsRun,'log':stream.getvalue(),
         'CPU_affinity':sorted(os.sched_getaffinity(0)),
         'scope':'deterministic LO O(Pw), fNL<=2, unprojected oriented source oracle, Gaussian long modes',
         'production_projection_implemented':False,'post_stochastic_validated':False,'IR_MCMC_run':False,
         'old_native_source_changed_by_this_extension':False,
         'initial_failed_audit':'IR_SOURCE_ORACLE_AUDIT_initial_normalization.json',
         'test_metric_correction':'fixed remainder normalization across wiggle amplitudes; source and rtol=.008 unchanged; independent absolute Gaussian remainder ratio=4',
         'hashes':{x:digest(PART/'code'/x) for x in ('post_rsd_ir_oracle.py','test_post_rsd_ir_oracle.py','audit_post_rsd_ir_oracle.py','post_rsd_displacement.py')}}
    path=PART/'outputs/sto_marisa_post_rsd_mean100_v1/IR_SOURCE_ORACLE_AUDIT.json'
    path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
