#!/usr/bin/env bash
#SBATCH --job-name=sto_simultaneous8_v2
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@960
#SBATCH --licenses=scratch,u2
#SBATCH --chdir=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
#SBATCH --output=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model/outputs/sto_marisa_post_rsd_mean100_v1/simultaneous_compute8_v2_%j.log

# 模块大纲：原环境、单线程数值；逐盒共享锁允许与login同时运行。
set -euo pipefail
cd /pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
set +u
source ../01_reconstruction_methods/common/nersc_env.sh
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=threads --mpi=none --network=no_vni \
    python -B -u code/run_sto_post_simultaneous_v2.py compute
