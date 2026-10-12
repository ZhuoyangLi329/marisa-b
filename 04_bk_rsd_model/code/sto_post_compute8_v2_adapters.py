"""模块大纲：v2仅修复STOP确认竞态及交接前USR1；原数值/失败v1源码不改。

新控制目录，复用原协调器、批次、atomic box；新增bootstrap均进入冻结源。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import signal
import time
import sto_post_compute8_common as common

OUT=common.ROOT/'compute8_measurement_series_v2'
SUBMIT=common.PART/'code/submit_sto_post_compute8_batch_v2.slurm'
BOX=common.PART/'code/run_sto_post_compute8_box_v2.py'
FILES=[Path(__file__),SUBMIT,BOX,common.PART/'code/run_sto_post_compute8_batch_v2.py',
       common.PART/'code/continue_sto_post_compute8_series_v2.py']


def kernel_state(record):
    return (Path('/proc')/str(record['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]


def confirmed_identity(record, *, stopped=False, timeout=5.):
    if not stopped:
        return common.original_identity(record)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if not common.original_identity(record):
            return False
        if kernel_state(record)=='T':
            return common.original_identity(record,stopped=True)
        time.sleep(.01)
    raise ValueError('same original owner has not acknowledged STOP; preserve it for inspection')


def rewrite_command(args):
    value=list(args)
    if value and value[0]=='sbatch':
        value=[str(SUBMIT) if x==str(common.PART/'code/submit_sto_post_compute8_batch.slurm') else x for x in value]
        if str(SUBMIT) in value:
            value=value[:1]+['--chdir='+str(common.PART),'--licenses=scratch,u2']+value[1:]
    return value


def rewrite_worker(args):
    if isinstance(args,(tuple,list)):
        return [str(BOX) if x==str(common.PART/'code/run_sto_post_compute8_box.py') else x for x in args]
    return args


def protect_worker_preexec(previous):
    # Slurm默认预警送到job steps；忽略只针对USR1，TERM/KILL仍正常生效。
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    if previous is not None:
        previous()


def has_scratch_license(value):
    # Slurm可把数量1显示为scratch或scratch:1；数量0不能当作依赖。
    return any(token.lower() in ('scratch','scratch:1') for token in value.split(','))


def verify_job8(jid):
    raw,v=common.job_fields(jid)
    expected=dict(UserId='lzy(112513)',JobName='sto_measure8_v2',Account='desi',QOS='shared',
        NumCPUs='8',**{'CPUs/Task':'8'},TimeLimit='00:30:00',Command=str(SUBMIT))
    if any(v.get(k)!=want for k,want in expected.items()) or not v.get('ReqTRES','').startswith('cpu=8,mem=14G,'):
        raise ValueError('exact owned v2 eight-core short measurement job required: '+raw)
    if not has_scratch_license(v.get('Licenses','')):
        raise ValueError('actual scratch dependency must be declared, not bypassed')
    if common.command(['scontrol','write','batch_script',str(jid),'-'])!=SUBMIT.read_text():
        raise ValueError('actual stored v2 script differs')
    return v


def bind_coordinator():
    common.OUT=OUT
    import continue_sto_post_compute8_series as original
    original.OUT=OUT;original.verify_job8=verify_job8;original.original_identity=confirmed_identity
    old_command,old_save=original.command,original.save
    original.command=lambda args:old_command(rewrite_command(args))
    def save(path,value):
        if Path(path)==OUT/'CONTRACT.json':
            # 同一frozen对象也被原guard闭包使用，不仅修改保存出的合同副本。
            value['immutable_sources'].update({str(p):common.sha(p) for p in FILES})
            value.update(control_version='v2',STOP_acknowledgement_polled=True,
                early_warning_before_handoff_supported=True,original_numeric_sources_modified=False)
        return old_save(path,value)
    original.save=save
    return original


def bind_batch():
    common.OUT=OUT
    import run_sto_post_compute8_batch as original
    original.OUT=OUT
    warnings=[];previous_enough=original.enough_time
    # 在原main等待HANDOFF之前安装；收到预警后不得开始任何新盒。
    signal.signal(signal.SIGUSR1,lambda *_:warnings.append('early deadline warning'))
    signal.signal(signal.SIGTERM,lambda *_:warnings.append('early administrative termination'))
    original.enough_time=lambda seconds:not warnings and previous_enough(seconds)
    previous_popen=original.subprocess.Popen
    def popen(args,*positional,**kwargs):
        rewritten=rewrite_worker(args)
        if isinstance(rewritten,list) and str(BOX) in rewritten:
            previous=kwargs.get('preexec_fn')
            kwargs['preexec_fn']=lambda:protect_worker_preexec(previous)
        return previous_popen(rewritten,*positional,**kwargs)
    original.subprocess.Popen=popen
    return original,warnings


def bind_box():
    common.OUT=OUT
    import run_sto_post_compute8_box as original
    original.OUT=OUT
    old_save=original.save
    def save(path,value):
        value.update(actual_v2_bootstrap_sha256=common.sha(BOX),control_namespace=str(OUT))
        return old_save(path,value)
    original.save=save
    return original
