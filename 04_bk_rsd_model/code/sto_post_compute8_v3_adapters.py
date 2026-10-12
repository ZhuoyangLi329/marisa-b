"""模块大纲：v3只修正环境加载的nounset；复用v2信号/暂停保护和原数值。

旧v1/v2及其冻结源保留，新会话单独目录，34项源共同冻结。
"""
from pathlib import Path
import sto_post_compute8_v2_adapters as base

OUT=base.common.ROOT/'compute8_measurement_series_v3'
SUBMIT=base.common.PART/'code/submit_sto_post_compute8_batch_v3.slurm'
BOX=base.common.PART/'code/run_sto_post_compute8_box_v3.py'
FILES=list(base.FILES)+[Path(__file__),SUBMIT,BOX,
    base.common.PART/'code/run_sto_post_compute8_batch_v3.py',
    base.common.PART/'code/continue_sto_post_compute8_series_v3.py']


def verify_job8(jid):
    raw,v=base.common.job_fields(jid)
    expected=dict(UserId='lzy(112513)',JobName='sto_measure8_v3',Account='desi',QOS='shared',
        NumCPUs='8',**{'CPUs/Task':'8'},TimeLimit='00:30:00',Command=str(SUBMIT))
    if any(v.get(k)!=want for k,want in expected.items()) or not v.get('ReqTRES','').startswith('cpu=8,mem=14G,'):
        raise ValueError('exact owned v3 eight-core short measurement job required: '+raw)
    if not base.has_scratch_license(v.get('Licenses','')):
        raise ValueError('actual scratch dependency must be declared')
    if base.common.command(['scontrol','write','batch_script',str(jid),'-'])!=SUBMIT.read_text():
        raise ValueError('actual stored v3 script differs')
    return v


def configure():
    base.OUT=OUT;base.SUBMIT=SUBMIT;base.BOX=BOX;base.FILES=FILES
    base.verify_job8=verify_job8;base.common.OUT=OUT
