#!/usr/bin/env bash
# 模块大纲：显式依赖路径；独立构建目录；tree-only 库、native 回归测试和投影节点工具。
set -euo pipefail
TREE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${REACT_DIR:?Set REACT_DIR to the existing ACTio-ReACTio checkout}"
TREE_BUILD="${TREE_BUILD:-$TREE_ROOT/build/tree_png_ir}"
mkdir -p "$TREE_BUILD"
CXX="${CXX:-g++}"
TREE_INCLUDES=(-I"$TREE_ROOT/src/marisa_b" -I"$TREE_ROOT/src/halo_v1" -I"$TREE_ROOT/src/eft_v2" -I"$REACT_DIR/reactions" -I"$REACT_DIR/reactions/src")
read -r -a TREE_GSL_CFLAGS <<< "$(pkg-config --cflags gsl)"
read -r -a TREE_GSL_LIBS <<< "$(pkg-config --libs gsl)"
TREE_FLAGS=(-std=c++17 -O2 -ffunction-sections -fdata-sections -DHAVE_CONFIG_H)
"$CXX" "${TREE_FLAGS[@]}" "${TREE_INCLUDES[@]}" "${TREE_GSL_CFLAGS[@]}" -fPIC -shared "$TREE_ROOT/src/marisa_b/tree_png_ir.cpp" -o "$TREE_BUILD/libmarisa_tree_png_ir.so"
"$CXX" "${TREE_FLAGS[@]}" "${TREE_INCLUDES[@]}" "${TREE_GSL_CFLAGS[@]}" -fPIC -shared "$TREE_ROOT/src/marisa_b/tree_png_ir.cpp" "$TREE_ROOT/src/marisa_b/post_recon_tree_png_ir.cpp" "${TREE_GSL_LIBS[@]}" -o "$TREE_BUILD/libmarisa_post_tree_png_ir.so"
# 重新编译本任务当前 native 实现，不能用可能漂移的历史 native 二进制作为回归依据。
"$CXX" "${TREE_FLAGS[@]}" "${TREE_INCLUDES[@]}" "${TREE_GSL_CFLAGS[@]}" -c "$TREE_ROOT/src/marisa_b/marisa_b_native.cpp" -o "$TREE_BUILD/native.o"
for TREE_SOURCE in PowerSpectrum Common array Quadrature; do
    "$CXX" "${TREE_FLAGS[@]}" "${TREE_INCLUDES[@]}" "${TREE_GSL_CFLAGS[@]}" -c "$REACT_DIR/reactions/src/$TREE_SOURCE.cpp" -o "$TREE_BUILD/$TREE_SOURCE.o"
done
"$CXX" "${TREE_FLAGS[@]}" "${TREE_INCLUDES[@]}" "${TREE_GSL_CFLAGS[@]}" "$TREE_ROOT/tests/eft_v2/test_tree_png_ir_native.cpp" "$TREE_ROOT/src/marisa_b/tree_png_ir.cpp" "$TREE_BUILD/native.o" "$TREE_BUILD/PowerSpectrum.o" "$TREE_BUILD/Common.o" "$TREE_BUILD/array.o" "$TREE_BUILD/Quadrature.o" -Wl,--gc-sections "${TREE_GSL_LIBS[@]}" -o "$TREE_BUILD/test_tree_png_ir_native"
"$CXX" "${TREE_FLAGS[@]}" "${TREE_INCLUDES[@]}" "${TREE_GSL_CFLAGS[@]}" "$TREE_ROOT/src/marisa_b/tree_ir_lattice_nodes.cpp" "$TREE_ROOT/src/eft_v2/lattice_shell_rule.cpp" -Wl,--gc-sections "${TREE_GSL_LIBS[@]}" -o "$TREE_BUILD/tree_ir_lattice_nodes"
"$TREE_BUILD/test_tree_png_ir_native"
