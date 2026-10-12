#!/usr/bin/env bash
#SBATCH --job-name=sto_noise8_short
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=00:20:00
#SBATCH --output=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model/outputs/sto_marisa_post_rsd_mean100_v1/compute8_noise_%j.log

# 模块大纲：仅8核、14GiB、20分钟；复用原44-bin数值，保护旧作业源码。
set -euo pipefail
cd /pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
source ../01_reconstruction_methods/common/nersc_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=threads --mpi=none --network=no_vni \
    python -B -u code/run_sto_noise_transport_compute8.py
