#!/usr/bin/env python3
"""marisa_gra 示例入口。

大纲：定位 common -> 选定方法 -> 使用 examples 配置 -> 调用共用生产 pipeline。
输入 CLI 与旧 runner 相同；输出限于本部分 outputs/marisa_gra。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from entrypoint import run_gra

if __name__ == '__main__':
    run_gra('marisa_gra')
