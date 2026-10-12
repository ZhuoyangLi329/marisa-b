#!/usr/bin/env bash
# 由示范脚本 source：复用已存在的 cosmodesi 环境，不安装依赖。
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
