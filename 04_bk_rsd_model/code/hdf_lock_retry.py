"""模块大纲：只等待HDF文件打开的EAGAIN锁冲突；不关闭锁、不重放采样步骤。"""
import errno
import time


def open_with_lock_retry(opener, *, maximum_wait=10., pause=.05, on_retry=None):
    if maximum_wait <= 0 or pause <= 0:
        raise ValueError('positive bounded lock wait required')
    started=time.monotonic(); attempts=0
    while True:
        try:
            return opener()
        except OSError as error:
            if error.errno not in (errno.EAGAIN,errno.EWOULDBLOCK) or 'lock' not in str(error).lower():
                raise
            attempts+=1
            if on_retry is not None: on_retry(attempts)
            remaining=maximum_wait-(time.monotonic()-started)
            if remaining<=0: raise
            time.sleep(min(pause,remaining))
