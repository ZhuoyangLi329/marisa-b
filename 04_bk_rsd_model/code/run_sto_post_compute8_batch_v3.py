"""模块大纲：原batch+v2生命周期保护，仅使用v3控制目录。"""
from sto_post_compute8_v3_adapters import configure,base

if __name__=='__main__':
    configure();original,_=base.bind_batch();original.main()
