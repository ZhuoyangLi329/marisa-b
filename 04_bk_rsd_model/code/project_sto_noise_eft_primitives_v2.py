"""模块大纲：v2只修正三种零腿几何；复用唯一v1双网格执行器并显式冻结路由。

v1失败目录/代码不改。原44 registry排除全零(0,0)，full-shell denominator保留。
数值回放不等于实际STO闭合；此执行器没有MCMC路径或formal noise PASS。
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
import project_sto_noise_eft_primitives_v1 as shared
from post_rsd_eft_zero_modes_v2 import project_zero_primitives_all_legs,independent_linear_oracle_all_legs

ROOT=shared.ROOT;OUT=ROOT/'noise_eft_primitives_v2'


def pipeline():
    failed=shared.read(ROOT/'noise_eft_primitives_v1/STATUS.json')
    if failed['state']!='FAILED_PRESERVED' or Path('/proc/'+str(failed['pid'])).exists():
        raise ValueError('prior failed candidate must be actually terminal; never duplicate a live process')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_sto_noise_eft_zero_modes_v2'))
    if not tests.wasSuccessful() or tests.testsRun!=2:raise ValueError('two actual all-zero-leg tests required')
    before=(shared.OUT,shared.independent_linear_oracle,shared.project_zero_primitives,shared.save_json)
    extra=[Path(__file__),shared.PART/'code/post_rsd_eft_zero_modes_v2.py',
        shared.PART/'code/test_sto_noise_eft_zero_modes_v2.py',ROOT/'noise_eft_primitives_v1/STATUS.json']
    frozen={str(p):shared.digest(p) for p in extra}
    original_save=shared.save_json
    def save(path,value):
        for p,want in frozen.items():
            if shared.digest(p)!=want:raise ValueError('explicit v2 route/source drift')
        value=dict(value)
        if Path(path).name=='CONTRACT.json':
            value['immutable_inputs'].update(frozen);value.update(
                zero_atom_route='all3 legs; original44 excludes all-zero shell pair',
                corrected_from='v1 rejected original shell0 because it wrongly assumed no outer zero mode',
                shared_executor_source=str(shared.PART/'code/project_sto_noise_eft_primitives_v1.py'),
                explicit_override_functions=['independent_linear_oracle_all_legs','project_zero_primitives_all_legs'])
        if Path(path).name=='AUDIT.json':value.update(unit_tests_actual_NERSC_passed=11,all_three_zero_legs=True)
        original_save(path,value)
    shared.OUT=OUT;shared.independent_linear_oracle=independent_linear_oracle_all_legs
    shared.project_zero_primitives=project_zero_primitives_all_legs;shared.save_json=save
    try:shared.pipeline()
    finally:shared.OUT,shared.independent_linear_oracle,shared.project_zero_primitives,shared.save_json=before


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--detach',action='store_true');args=ap.parse_args()
    if socket.gethostname()!='login26' or os.sched_getaffinity(0)!={14}:raise ValueError('login26 CPU14 single core required')
    if args.detach:
        log=ROOT/'noise_eft_primitives_v2.log';launch=ROOT/'noise_eft_primitives_v2.launch.json'
        if OUT.exists() or log.exists() or launch.exists():raise ValueError('inspect exact original v2 before any recovery')
        with log.open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        original=shared.save_json;original(launch,dict(pid=proc.pid,host=socket.gethostname(),CPU_affinity=[14],
            created_UTC=shared.stamp(),log=str(log)))
        print(json.dumps(shared.read(launch)),flush=True)
    else:pipeline()


if __name__=='__main__':main()
