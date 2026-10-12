"""模块大纲：复用原batch，交接前也处理USR1，全部新输出在v2控制目录。"""
from sto_post_compute8_v2_adapters import bind_batch

if __name__=='__main__':
    original,_=bind_batch();original.main()
