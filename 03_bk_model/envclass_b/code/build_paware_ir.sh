#!/usr/bin/env bash
# 模块大纲：新输出目录中的独立桥接库；不覆盖旧 native 库或生产结果。
set -euo pipefail
TASK_ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model
TASK_CODE="$TASK_ROOT/envclass_b/code"
TASK_CORE="$TASK_ROOT/marisa_b_core"
TASK_BUILD="$TASK_ROOT/envclass_b/outputs/paware_pb_ir_brecfix_k012_k014_v2/build"
TASK_REACT=/pscratch/sd/l/lzy/reconstruction-png/codes/marisa_b/external/ACTio-ReACTio
mkdir -p "$TASK_BUILD"
FLAGS=(-std=c++17 -O3 -ffunction-sections -fdata-sections -DHAVE_CONFIG_H
    -I"$TASK_CORE/src/marisa_b" -I"$TASK_CORE/src/halo_v1" -I"$TASK_CORE/src/eft_v2"
    -I"$TASK_REACT/reactions" -I"$TASK_REACT/reactions/src")
read -r -a GSL_CFLAGS <<< "$(pkg-config --cflags gsl)"
read -r -a GSL_LIBS <<< "$(pkg-config --libs gsl)"
g++ "${FLAGS[@]}" "${GSL_CFLAGS[@]}" -fPIC -shared "$TASK_CODE/paware_ir_bridge.cpp" \
    "$TASK_CORE/src/marisa_b/tree_png_ir.cpp" "$TASK_CORE/src/marisa_b/post_recon_tree_png_ir.cpp" \
    "${GSL_LIBS[@]}" -o "$TASK_BUILD/libpaware_ir.so"
g++ "${FLAGS[@]}" "${GSL_CFLAGS[@]}" "$TASK_CODE/paware_ir_nodes.cpp" \
    "$TASK_CORE/src/eft_v2/lattice_shell_rule.cpp" -Wl,--gc-sections "${GSL_LIBS[@]}" \
    -o "$TASK_BUILD/paware_ir_nodes"
