#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=paware_pb_audit

# 在四组采样正常退出后复核；若未收敛，汇总必须保留未收敛标识。
set -eo pipefail
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/code/audit_paware_pb_mcmc.py
