#!/usr/bin/env bash
# 模块大纲：第四部分独立 native 库；共享第三部分 bias 类型；保留旧库。
set -euo pipefail
TASK_PART=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
TASK_CORE=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/marisa_b_core
TASK_REACT=/pscratch/sd/l/lzy/reconstruction-png/codes/marisa_b/external/ACTio-ReACTio
mkdir -p "$TASK_PART/work/build"
g++ -std=c++17 -O3 -fPIC -shared -I"$TASK_CORE/src/marisa_b" \
    -I"$TASK_REACT/reactions" -I"$TASK_REACT/reactions/src" \
    "$TASK_PART/code/native/tree_png_rsd.cpp" -o "$TASK_PART/work/build/libmarisa_tree_rsd.so"
