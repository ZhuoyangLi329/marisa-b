"""模块大纲：任何执行入口的实际owner退出门；不由STATUS字符串推断终止。"""
from pathlib import Path


def owner_quiescent(status,proc_root=Path('/proc')):
    pid=status.get('pid')
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0:
        raise ValueError('actual positive owner PID required')
    proc=Path(proc_root)/str(pid)
    if not proc.exists():return True
    try:
        suffix=(proc/'stat').read_text().rsplit(')',1)[1].split()
        return suffix[0] in ('Z','X')
    except (OSError,IndexError):
        return not proc.exists()
