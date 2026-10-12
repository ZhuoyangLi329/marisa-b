#!/usr/bin/env bash
#SBATCH --job-name=sto_ir_match8
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@240
#SBATCH --licenses=scratch,u2
#SBATCH --chdir=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
#SBATCH --output=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model/outputs/sto_marisa_post_rsd_mean100_v1/compute8_ir_matching_%j.log

# 模块大纲：只申请8核，重B投影单路，保护原测量数值与checkpoint。
set -euo pipefail
set +u
source ../01_reconstruction_methods/common/nersc_env.sh
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=threads --mpi=none --network=no_vni \
    python -B -u code/run_sto_ir_matching_compute8.py
