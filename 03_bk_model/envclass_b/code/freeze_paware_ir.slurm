#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --job-name=paware_ir_freeze
# 模块大纲：模板已经通过验收后恢复冻结步骤；不重建、不覆盖模板和已有 case。
set -euo pipefail
TASK_ROOT=/pscratch/sd/l/lzy/reconstuction-png-v2/03_bk_model/envclass_b
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
TASK_PY=/global/homes/l/lzy/anaconda3/envs/desilike/bin/python
for TASK_METHOD in pre sto_marisa; do
    "$TASK_PY" -B -u "$TASK_ROOT/code/run_paware_pb_ir_mcmc.py" prepare --method "$TASK_METHOD" --ir
done
