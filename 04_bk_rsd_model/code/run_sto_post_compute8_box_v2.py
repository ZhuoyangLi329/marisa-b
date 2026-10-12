"""模块大纲：只重定向v2控制/staging目录，原target/cov/pilot数值不变。"""
import signal
signal.signal(signal.SIGUSR1,signal.SIG_IGN)
from sto_post_compute8_v2_adapters import bind_box

if __name__=='__main__':
    bind_box().main()
