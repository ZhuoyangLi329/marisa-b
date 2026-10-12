#!/usr/bin/env bash
#SBATCH --job-name=sto_post_noise44
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=01:00:00
# 只做44-bin noise transport理论模板；不修改700盒测量作业或主MCMC。
set -eo pipefail
cd /pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
source ../01_reconstruction_methods/common/nersc_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
srun -n1 -c32 --cpu-bind=threads --mpi=none --network=no_vni \
    python -B -u code/run_sto_noise_transport_compute.py
