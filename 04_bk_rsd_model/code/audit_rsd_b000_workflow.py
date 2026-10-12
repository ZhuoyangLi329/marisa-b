"""模块大纲：独立回归套件；已冻结模板/库 hash；新增源清单；不作科学恢复门。"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import io
import unittest
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import json
from rsd_b000_contract import PART,OUT,digest,save_json,reference

def test_ids(suite):
    return [name for item in suite for name in (test_ids(item) if isinstance(item,unittest.TestSuite) else [item.id()])]

def main():
    suite=unittest.defaultTestLoader.discover(str(PART/'code'),pattern='test_rsd_b000_*.py')
    names=test_ids(suite);stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2,buffer=True).run(suite)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise ValueError('workflow regression tests failed')
    contract=json.loads((OUT/'contract.json').read_text())
    template_root=Path(contract.get('source_output',str(OUT)))
    template_contract_hash=contract.get('source_contract_sha256',digest(OUT/'contract.json'))
    library=PART/'work/build/libmarisa_tree_rsd.so'
    if digest(library)!=contract['source_library_sha256']:raise ValueError('source library changed after contract freeze')
    errors=[]
    for index in reference()['certified_indices']:
        with np.load(template_root/'template_bins_m144'/f'bin{index:03d}.npz') as z:
            if str(z['contract_sha256'])!=template_contract_hash:raise ValueError('template contract drift')
            errors.append(float(z['probe_relative_error']))
    if not np.isfinite(errors).all() or max(errors)>2e-10:raise ValueError('independent polynomial probe failed')
    code=PART/'code';files=set(code.glob('test_rsd_b000_*.py'))|set(code.glob('rsd_b000_*.py'))|set(code.glob('rsd_b000_*.sh'))|set(code.glob('audit_rsd_b000_*.py'))
    files.update(code/n for n in ['audit_rsd_b000_workflow.py','measure_rsd_b000.py','build_rsd_b000_templates.py',
                                  'run_rsd_b000_inference.py','run_rsd_b000_mcmc.py','report_rsd_b000.py',
                                  'submit_rsd_b000_covariance_fix.py','run_rsd_b000_login.py',
                                  'recover_rsd_b000_login.py','rebalance_rsd_login_worker.py'])
    shared=[PART.parent/'01_reconstruction_methods/common/halo_b000_pipeline.py',
            PART.parent/'01_reconstruction_methods/common/quijote_dm_2d_paperbin_runner_tools.py',
            PART.parent/'02_reconstruction_on_pk/code/run_five_method_p0only_mcmc.py']
    value={'status':'PASS_NUMERICAL_AND_PROGRAM_REGRESSION','generated_UTC':datetime.now(timezone.utc).isoformat(),
           'tests_run':result.testsRun,'tests':names,'template_bins':len(errors),
           'independent_b1_probe_max_relative_error':max(errors),'contract_sha256':digest(OUT/'contract.json'),
           'source_library_sha256':digest(library),'new_source_hashes':{str(p.relative_to(PART)):digest(p) for p in sorted(files)},
           'readonly_shared_source_hashes':{str(p):digest(p) for p in shared},
           'scientific_fNL_recovery_claim':False,'old_repository_modified':False,'PNG_generated':False,
           'note':'Synthetic statistical/layout fixtures are temporary and isolated; real-data pilot/covariance/MCMC are separate gates.'}
    save_json(OUT/'WORKFLOW_AUDIT.json',value)
    print(json.dumps({'status':value['status'],'tests':result.testsRun,'max_probe_error':max(errors)}),flush=True)

if __name__=='__main__':main()
