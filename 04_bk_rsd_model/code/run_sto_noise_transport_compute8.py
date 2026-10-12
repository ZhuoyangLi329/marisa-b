"""模块大纲：用户指定8核短作业；复用原44-bin噪声模板数值与队列。

只改变申请/CPU上限和审计的实际入口；原32核源码、失败/等待记录保留。
14GiB下沿用4GiB/worker+4GiB共享余量，最多两个原单核worker。
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import run_sto_noise_transport_compute as original


def capacity8(allowed, memory_mib):
    if len(allowed) != 8:
        raise ValueError('exactly eight allowed logical CPUs required')
    return original.capacity(allowed, memory_mib)


def main():
    if os.environ.get('SLURM_CPUS_PER_TASK') != '8':
        raise ValueError('actual user-requested8 CPU Slurm allocation required')
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 8:
        raise ValueError('fewer than eight allocated allowed CPUs')
    os.sched_setaffinity(0, set(available[:8]))
    # capacity使用闭包保存旧函数，不能让capacity8递归调用被替换后的符号。
    previous_capacity, previous_save = original.capacity, original.save_json
    def capacity(allowed, memory):
        if len(allowed) != 8:
            raise ValueError('eight logical CPU maximum')
        return previous_capacity(allowed, memory)
    def save(path, value):
        if Path(path) == original.OUT/'COMPUTE_STATUS.json':
            value['requested_CPU_limit'] = 8
            value['human_override'] = 'compute nodes: eight cores, short walltime'
            value['actual_submission_source'] = str(original.PART/'code/submit_sto_noise_transport_compute8.slurm')
            value['old_submit32_source_is_historical_not_executed'] = True
            for source in (Path(__file__), original.PART/'code/submit_sto_noise_transport_compute8.slurm'):
                value['queue_sources'][str(source)] = original.digest(source)
        return previous_save(path, value)
    original.capacity, original.save_json = capacity, save
    try:
        original.main()
    finally:
        original.capacity, original.save_json = previous_capacity, previous_save


if __name__ == '__main__':
    main()
