#!/usr/bin/env bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=marisa_pre_rsd
# 模块大纲：单 CPU 计算节点；独立审计先行；真实 pre-RSD 样本模型诊断。
set -euo pipefail
TASK_PART=/pscratch/sd/l/lzy/reconstuction-png-v2/04_bk_rsd_model
TASK_PY=/global/homes/l/lzy/anaconda3/envs/desilike/bin/python
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$TASK_PY" -B -m pytest -q -p no:cacheprovider "$TASK_PART/work/reference_package/tests/python/test_tree_png_rsd_reference.py"
"$TASK_PY" -B -u "$TASK_PART/code/audit_tree_rsd.py"
"$TASK_PY" -B -u "$TASK_PART/code/test_periodic_rsd.py"
"$TASK_PY" -B -u "$TASK_PART/code/test_quijote_strict_triangle_fft_estimator.py"
"$TASK_PY" -B -u "$TASK_PART/code/run_pre_rsd_model_test.py"
"$TASK_PY" -B -u "$TASK_PART/code/plot_pre_rsd_diagnostic.py"
