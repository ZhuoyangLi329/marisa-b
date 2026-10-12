#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --job-name=paware_ir_mcmc
# 模块大纲：每个 case 一个 CPU；HDF 可续跑；先验收对应模型再采样。
set -euo pipefail
TASK_ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b
TASK_OUT="$TASK_ROOT/outputs/paware_pb_ir_brecfix_k012_k014_v2"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
TASK_PY=/global/homes/l/lzy/anaconda3/envs/desilike/bin/python
TASK_CASE="$1"
"$TASK_PY" -B -u "$TASK_ROOT/code/run_paware_pb_ir_mcmc.py" sample --case "$TASK_CASE" \
    > "$TASK_OUT/$TASK_CASE/sampling.log" 2>&1
