#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@7200
#SBATCH --job-name=sm_post_rsd_cpu32
#SBATCH --output=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model/outputs/sto_marisa_post_rsd_mean100_v1/compute32_migration/slurm_%j.log

# 模块大纲：沿用旧STO+Marisa的CPU/shared/desi；32核预算、内存限额下9盒并发。
set -euo pipefail
TASK_PART=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
cd "$TASK_PART"
source ../01_reconstruction_methods/common/nersc_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
# 单个Slurm rank带32核CPU集合；每个worker继承环境且只绑定自己的一个核。
srun --ntasks=1 --cpus-per-task=32 --cpu-bind=threads --mpi=none --network=no_vni python -B -u code/run_sto_post_compute_queue.py --workers 32
# 所有盒子完成且parent真实退出后才可汇总；不是部分协方差或MCMC放行。
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=threads --mpi=none --network=no_vni python -B -u code/assemble_sto_post_measurements.py
