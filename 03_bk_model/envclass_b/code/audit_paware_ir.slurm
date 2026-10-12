#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=paware_ir_audit
# 模块大纲：全部六组任务成功结束后只读核验链，输出独立新汇总。
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
/global/homes/l/lzy/anaconda3/envs/desilike/bin/python -B -u \
    /pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b/code/audit_paware_pb_ir_mcmc.py
