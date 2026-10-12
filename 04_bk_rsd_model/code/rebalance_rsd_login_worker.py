"""模块大纲：核验本任务 PID/host；只在原八核集合内迁移单核 worker；来源留档。

不暂停/终止进程，不改链、RNG、模型、先验或 MCMC 源码，只调整 OS affinity。
"""
import os
import argparse
import json
import socket
from datetime import datetime,timezone
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-index',type=int,required=True);parser.add_argument('--cpu',type=int,required=True)
    args=parser.parse_args()
    # 此工具本身也是单核，随后进入已有核集合；不加载任何数值计算库。
    part=Path(__file__).resolve().parents[1]
    output=part/'outputs/pre_rsd_b000_bonly_mean100_v1'
    state=json.loads((output/'LOGIN_STATUS.json').read_text())
    launch=json.loads((output/'LOGIN_LAUNCH.json').read_text())
    pool=set(launch['affinity'])
    if socket.gethostname()!=launch['host'] or state['status']!='RUNNING':raise ValueError('not the active runner host')
    if len(pool)>8 or args.cpu not in pool or set(state['affinity'])!=pool:raise ValueError('CPU outside approved pool')
    os.sched_setaffinity(0,{args.cpu})
    rows=[r for r in state['active_workers'] if r['case_index']==args.case_index]
    if len(rows)!=1:raise ValueError('case is no longer an active worker')
    pid=rows[0]['pid'];proc=Path('/proc')/str(pid)
    if proc.stat().st_uid!=os.getuid():raise ValueError('worker UID mismatch')
    stat=proc.joinpath('stat').read_text().rsplit(') ',1)[1].split()
    if int(stat[1])!=state['pid']:raise ValueError('worker is not a child of this runner')
    argv=[x.decode() for x in proc.joinpath('cmdline').read_bytes().split(b'\0') if x]
    expected=str(part/'code/run_rsd_b000_mcmc.py')
    if expected not in argv or argv[argv.index('--case-index')+1]!=str(args.case_index):raise ValueError('worker command mismatch')
    old=sorted(os.sched_getaffinity(pid))
    if not set(old).issubset(pool) or set(os.sched_getaffinity(state['pid']))!=pool:raise ValueError('affinity contract drift')
    os.sched_setaffinity(pid,{args.cpu});actual=sorted(os.sched_getaffinity(pid))
    if actual!=[args.cpu]:raise ValueError('affinity update failed')
    path=output/'CPU_REBALANCE_LOG.json'
    events=json.loads(path.read_text()) if path.exists() else []
    event={'UTC':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),'pid':pid,
           'case_index':args.case_index,'old_allowed_CPUs':old,'new_allowed_CPUs':actual,
           'unchanged_approved_pool':sorted(pool),'maximum_logical_CPUs':len(pool),
           'chain_model_RNG_untouched':True}
    events.append(event)
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(events,indent=2)+'\n');temporary.replace(path)
    print(json.dumps(event),flush=True)

if __name__=='__main__':main()
