#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=paware_ir_prepare
# 模块大纲：计算节点生产模板；双精度/独立 b1 网格验收；成功后才冻结四个 IR case。
set -euo pipefail
TASK_ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b
TASK_OUT="$TASK_ROOT/outputs/paware_pb_ir_brecfix_k012_k014_v2"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
TASK_PY=/global/homes/l/lzy/anaconda3/envs/desilike/bin/python
"$TASK_PY" -B -u "$TASK_ROOT/code/paware_pb_ir_templates.py" all --workers 16
for TASK_METHOD in pre sto_marisa; do
    "$TASK_PY" -B -u "$TASK_ROOT/code/run_paware_pb_ir_mcmc.py" prepare --method "$TASK_METHOD" --ir
done
