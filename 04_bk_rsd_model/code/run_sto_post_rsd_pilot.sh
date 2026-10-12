#!/usr/bin/env bash
# 模块大纲：沿用 cosmodesi；单线程 CPU pilot；调用方负责硬 affinity <=8。
set -eo pipefail
cd /pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
source ../01_reconstruction_methods/common/nersc_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
exec python -B -u code/run_sto_post_rsd_pilot.py "$@"
