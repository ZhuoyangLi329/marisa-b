"""模块大纲：source/FFT 门、旧产物保护、单核记录；不放行 catalog/MCMC。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import io
import json
import os
import socket
import unittest
from tree_png_rsd_native import PART


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if len(os.sched_getaffinity(0))>8:raise RuntimeError('audit CPU affinity exceeds task limit')
    oldlib=PART/'work/build/libmarisa_tree_rsd.so'
    expected='9a3ced1c69a0224e1738a2d7e4ea3a841b8533ec4a3b4a816d755a9051a0efea'
    if digest(oldlib)!=expected:raise RuntimeError('frozen pre library changed')
    original={
        'code/rsd_b000_fft_projection.py':'da7fe70eeb34b23770897f1c32671ca2acdd025bf9a962a0833ef572098f1c03',
        'outputs/pre_rsd_p0only_mean100_v1/RESULTS.json':'308122b08d10ab85ee45f5124c484b69b22302cd8388b6e3f1a177a8f67cd55f',
        'outputs/pre_rsd_pb_mean100_v1/RESULTS.json':'8abb934915798f1f84f90692240e4990127f1e181670089f7027ab1f90863d0c',
        'outputs/pre_rsd_b000_bonly_mean100_v1/RESULTS.json':'56f99654de8caeaecc9a31f8a7ca58d13828da9f5ae832edbb826bfa47d35335',
    }
    protected={k:dict(expected=v,actual=digest(PART/k)) for k,v in original.items()}
    if any(x['actual']!=x['expected'] for x in protected.values()):
        raise RuntimeError('old frozen result/source changed')
    suite=unittest.defaultTestLoader.discover(str(PART/'code'),pattern='test_post_rsd_*.py')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    paths=[PART/'code'/x for x in ('post_rsd_fixedrec.py','post_rsd_fft_projection.py',
        'test_post_rsd_fixedrec.py','test_post_rsd_projection.py','test_post_rsd_weighted_factorial.py','audit_post_rsd_source.py',
        'post_rsd_displacement.py','test_post_rsd_displacement.py','post_rsd_ir_oracle.py','test_post_rsd_ir_oracle.py',
        'post_rsd_ir_harmonics.py','post_rsd_ir_projection.py','post_rsd_ir_inputs.py','test_post_rsd_ir_projection.py',
        'post_rsd_ir_grid.py','test_post_rsd_ir_grid.py',
        'post_rsd_catalog_identity.py','test_post_rsd_catalog_identity.py',
        'post_rsd_poisson_projection.py','test_post_rsd_poisson_projection.py',
        'post_rsd_catalog.py','test_post_rsd_catalog.py',
        'native/tree_png_rsd.cpp','native/tree_png_rsd.h','build_post_rsd.sh')]
    payload={'status':'SOURCE_TESTS_PASS_PRODUCTION_NOT_RELEASED' if result.wasSuccessful() else 'FAIL',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),
        'CPU_affinity':sorted(os.sched_getaffinity(0)),'numeric_threads':{k:os.environ.get(k) for k in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'test_log':log.getvalue(),
        'old_library_sha256':expected,'new_library_sha256':digest(PART/'work/build/libmarisa_tree_rsd_fixedrec.so'),
        'protected_old_outputs':protected,'sources':{str(p):digest(p) for p in paths},
        'scope':'fixed-catalog source, exact-lattice no-IR and small-bin deterministic LO IR projection, point factorial identity, Gaussian long tensor, exact grid symmetry acceleration, explicit leading Poisson-shift projection; not full stochastic closure',
        'not_yet_validated':['actual reconstruction operator transfer and alias','STO weighted discrete/shift noise',
                             'physical p calibration','independent Fid500 STO inputs','mock recovery','production IR projection and full post stochastic'],
        'catalog_production_released':False,'MCMC_released':False}
    dest=PART/'outputs/sto_marisa_post_rsd_mean100_v1/SOURCE_AUDIT.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(payload,indent=2)+'\n')
    print(log.getvalue());print(json.dumps({k:v for k,v in payload.items() if k not in ('test_log','sources')},indent=2))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
