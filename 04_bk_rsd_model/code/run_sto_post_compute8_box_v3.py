"""模块大纲：原atomic box数值，v3路由；USR1不结束正在计算的盒。"""
import signal
signal.signal(signal.SIGUSR1,signal.SIG_IGN)
from sto_post_compute8_v3_adapters import configure,base

if __name__=='__main__':
    configure();base.bind_box().main()
